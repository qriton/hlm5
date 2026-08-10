"""Fisher metric fitting and certified class-mixture energy dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class LineSearchError(RuntimeError):
    """Raised when a non-stationary row cannot satisfy the registered search."""


@dataclass(frozen=True)
class AlignedModel:
    """Closed-form Fisher/LDA model fitted on frozen encoder states."""

    alpha: float
    global_mean: torch.Tensor
    projection: torch.Tensor
    centroids: torch.Tensor
    within_eigenvalues: torch.Tensor
    discriminant_eigenvalues: torch.Tensor


@dataclass(frozen=True)
class CentroidControls:
    """Unaligned raw and common-mode-centered centroid controls."""

    global_mean: torch.Tensor
    raw_centroids: torch.Tensor
    centered_centroids: torch.Tensor


@dataclass(frozen=True)
class SettleTrace:
    """Per-row diagnostics for exact Euclidean energy descent."""

    energies: torch.Tensor
    step_lengths: torch.Tensor
    gradient_norms: torch.Tensor
    directional_terms: torch.Tensor
    armijo_residuals: torch.Tensor
    halvings: torch.Tensor
    constrained_stationary: torch.Tensor


def unit(value: torch.Tensor, dim: int = -1, eps: float = 1e-15) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm <= eps).any()):
        raise ValueError("cannot normalize zero or non-finite tensor")
    return value / norm


def _validate_fit_inputs(
    features: torch.Tensor, targets: torch.Tensor, class_count: int
) -> None:
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ValueError("features/targets have incompatible shapes")
    if features.dtype != torch.float64 or features.device.type != "cpu":
        raise ValueError("registered fit requires CPU torch.float64 features")
    if targets.dtype != torch.int64 or targets.device.type != "cpu":
        raise ValueError("registered targets require CPU torch.int64")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("fit features are non-finite")
    if (
        class_count < 2
        or int(targets.min()) != 0
        or int(targets.max()) != class_count - 1
    ):
        raise ValueError("targets do not span registered classes")
    if torch.unique(targets).numel() != class_count:
        raise ValueError("one or more classes are absent")


def _class_means(
    features: torch.Tensor, targets: torch.Tensor, class_count: int
) -> torch.Tensor:
    means = []
    for class_index in range(class_count):
        class_rows = features[targets == class_index]
        if class_rows.shape[0] == 0:
            raise ValueError(f"class {class_index} is empty")
        means.append(class_rows.mean(dim=0))
    return torch.stack(means)


def _canonicalize_eigenvector_signs(vectors: torch.Tensor) -> torch.Tensor:
    canonical = vectors.clone()
    for column in range(canonical.shape[1]):
        vector = canonical[:, column]
        pivot = int(torch.argmax(torch.abs(vector)))
        if float(vector[pivot]) < 0.0:
            canonical[:, column] = -vector
    return canonical


def fit_aligned_model(
    features: torch.Tensor,
    targets: torch.Tensor,
    class_count: int,
    alpha: float,
) -> AlignedModel:
    """Fit the registered shrinkage Fisher projection and LDA centroids."""

    _validate_fit_inputs(features, targets, class_count)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")

    global_mean = features.mean(dim=0)
    raw_class_means = _class_means(features, targets, class_count)
    within = torch.zeros((features.shape[1], features.shape[1]), dtype=features.dtype)
    for class_index in range(class_count):
        residual = features[targets == class_index] - raw_class_means[class_index]
        within += residual.T @ residual
    within /= float(features.shape[0])
    covariance_scale = torch.trace(within) / float(features.shape[1])
    if not bool(torch.isfinite(covariance_scale)) or float(covariance_scale) <= 0.0:
        raise ValueError("within-class covariance scale is not positive")
    regularized = (1.0 - alpha) * within
    regularized += (
        alpha * covariance_scale * torch.eye(features.shape[1], dtype=features.dtype)
    )
    within_values, within_vectors = torch.linalg.eigh(regularized)
    if float(within_values.min()) <= 0.0:
        raise ValueError("regularized within covariance is not positive definite")
    inverse_sqrt = within_vectors @ torch.diag(within_values.rsqrt()) @ within_vectors.T

    centered_means = raw_class_means - global_mean
    between = centered_means.T @ centered_means / float(class_count)
    discriminant_matrix = inverse_sqrt @ between @ inverse_sqrt
    discriminant_matrix = 0.5 * (discriminant_matrix + discriminant_matrix.T)
    discriminant_values, discriminant_vectors = torch.linalg.eigh(discriminant_matrix)
    rank = class_count - 1
    top_vectors = discriminant_vectors[:, -rank:]
    top_vectors = _canonicalize_eigenvector_signs(top_vectors)
    projection = top_vectors.T @ inverse_sqrt
    projected = (features - global_mean) @ projection.T
    centroids = _class_means(projected, targets, class_count)
    if not bool(torch.isfinite(projection).all()) or not bool(
        torch.isfinite(centroids).all()
    ):
        raise ValueError("aligned model contains non-finite values")
    return AlignedModel(
        alpha=float(alpha),
        global_mean=global_mean,
        projection=projection,
        centroids=centroids,
        within_eigenvalues=within_values,
        discriminant_eigenvalues=discriminant_values[-rank:],
    )


def transform(features: torch.Tensor, model: AlignedModel) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != model.global_mean.numel():
        raise ValueError("feature shape is incompatible with aligned model")
    return (features - model.global_mean) @ model.projection.T


def aligned_scores(states: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Equal-prior LDA scores, equivalent to negative squared distance."""

    if states.ndim != 2 or centroids.ndim != 2 or states.shape[1] != centroids.shape[1]:
        raise ValueError("state/centroid shapes are incompatible")
    offsets = 0.5 * torch.sum(centroids * centroids, dim=1)
    return states @ centroids.T - offsets


def fit_centroid_controls(
    features: torch.Tensor, targets: torch.Tensor, class_count: int
) -> CentroidControls:
    _validate_fit_inputs(features, targets, class_count)
    global_mean = features.mean(dim=0)
    raw_centroids = unit(_class_means(features, targets, class_count))
    centered_features = unit(features - global_mean)
    centered_centroids = unit(_class_means(centered_features, targets, class_count))
    return CentroidControls(global_mean, raw_centroids, centered_centroids)


def centroid_control_scores(
    features: torch.Tensor, controls: CentroidControls
) -> dict[str, torch.Tensor]:
    raw = unit(features) @ controls.raw_centroids.T
    centered = unit(features - controls.global_mean) @ controls.centered_centroids.T
    return {"raw_centroid": raw, "centered_centroid": centered}


def mixture_energy(
    states: torch.Tensor, centroids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scores = aligned_scores(states, centroids)
    return 0.5 * torch.sum(states * states, dim=1) - temperature * torch.logsumexp(
        scores / temperature, dim=1
    )


def negative_energy_gradient(
    states: torch.Tensor, centroids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    probabilities = torch.softmax(
        aligned_scores(states, centroids) / temperature, dim=1
    )
    return probabilities @ centroids - states


def decision_boundary_radius(
    states: torch.Tensor, targets: torch.Tensor, centroids: torch.Tensor
) -> float:
    """Median nearest positive true-class boundary distance on correct fit rows."""

    scores = aligned_scores(states, centroids)
    predictions = torch.argmax(scores, dim=1)
    correct = predictions == targets
    if not bool(correct.any()):
        raise ValueError("no correctly classified fit rows for trust radius")
    row_indices = torch.arange(states.shape[0], dtype=torch.int64)
    true_scores = scores[row_indices, targets]
    true_centroids = centroids[targets]
    difference_norms = torch.linalg.vector_norm(
        true_centroids.unsqueeze(1) - centroids.unsqueeze(0), dim=2
    )
    margins = true_scores.unsqueeze(1) - scores
    distances = margins / difference_norms.clamp_min(1e-15)
    distances[row_indices, targets] = torch.inf
    nearest = torch.min(distances, dim=1).values
    valid = correct & torch.isfinite(nearest) & (nearest > 0.0)
    if not bool(valid.any()):
        raise ValueError("no positive fit boundary distances")
    radius = torch.quantile(nearest[valid], 0.5, interpolation="linear")
    if not bool(torch.isfinite(radius)) or float(radius) <= 0.0:
        raise ValueError("trust radius is not finite and positive")
    return float(radius)


def project_to_ball(
    origins: torch.Tensor, proposals: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = proposals - origins
    norms = torch.linalg.vector_norm(delta, dim=1)
    outside = norms > radius
    scale = (radius / norms.clamp_min(1e-15)).clamp_max(1.0)
    projected = origins + delta * scale.unsqueeze(1)
    return projected, outside


def constrained_settle(
    initial_states: torch.Tensor,
    centroids: torch.Tensor,
    radius: float,
    *,
    temperature: float,
    steps: int,
    initial_step_radius_fraction: float,
    armijo_c: float,
    max_halvings: int,
    stationary_tolerance: float,
    energy_tolerance: float,
    cap_tolerance: float,
) -> tuple[torch.Tensor, SettleTrace]:
    """Run registered vectorized descent inside an input-anchored Euclidean ball."""

    if initial_states.ndim != 2 or centroids.ndim != 2:
        raise ValueError("states and centroids must be matrices")
    if initial_states.shape[1] != centroids.shape[1]:
        raise ValueError("state/centroid dimensions differ")
    if radius <= 0.0 or steps < 0:
        raise ValueError("radius must be positive and steps non-negative")

    origins = initial_states
    states = initial_states.clone()
    energies = [mixture_energy(states, centroids, temperature)]
    step_columns: list[torch.Tensor] = []
    gradient_columns: list[torch.Tensor] = []
    directional_columns: list[torch.Tensor] = []
    residual_columns: list[torch.Tensor] = []
    halving_columns: list[torch.Tensor] = []
    stationary_columns: list[torch.Tensor] = []
    batch = states.shape[0]
    base_step = radius * initial_step_radius_fraction

    for _ in range(steps):
        current_energy = energies[-1]
        negative_gradient = negative_energy_gradient(states, centroids, temperature)
        gradient_norm = torch.linalg.vector_norm(negative_gradient, dim=1)
        active = gradient_norm > stationary_tolerance
        direction = negative_gradient / gradient_norm.clamp_min(
            stationary_tolerance
        ).unsqueeze(1)
        displacement_from_origin = states - origins
        at_boundary = (
            torch.linalg.vector_norm(displacement_from_origin, dim=1)
            >= radius - cap_tolerance
        )
        points_outward = torch.sum(displacement_from_origin * direction, dim=1) > 0.0
        energy_gradient = -negative_gradient

        accepted = ~active
        chosen_states = states.clone()
        chosen_energy = current_energy.clone()
        chosen_step = torch.zeros(batch, dtype=states.dtype)
        chosen_directional = torch.zeros(batch, dtype=states.dtype)
        chosen_residual = torch.zeros(batch, dtype=states.dtype)
        chosen_halvings = torch.zeros(batch, dtype=torch.int64)
        constrained_stationary = ~active

        for halving in range(max_halvings + 1):
            pending = ~accepted
            if not bool(pending.any()):
                break
            trial_step = base_step / float(2**halving)
            trial = states + trial_step * direction
            trial, _ = project_to_ball(origins, trial, radius)
            displacement = trial - states
            actual_step = torch.linalg.vector_norm(displacement, dim=1)
            no_movement = actual_step <= stationary_tolerance
            directional = torch.sum(energy_gradient * displacement, dim=1)
            trial_energy = mixture_energy(trial, centroids, temperature)
            residual = trial_energy - (current_energy + armijo_c * directional)
            valid_descent = (directional < 0.0) & (residual <= energy_tolerance)
            take_stationary = pending & no_movement & at_boundary & points_outward
            take_descent = pending & (~no_movement) & valid_descent
            take = take_stationary | take_descent

            chosen_states = torch.where(take_descent.unsqueeze(1), trial, chosen_states)
            chosen_energy = torch.where(take_descent, trial_energy, chosen_energy)
            chosen_step = torch.where(take_descent, actual_step, chosen_step)
            chosen_directional = torch.where(
                take_descent, directional, chosen_directional
            )
            chosen_residual = torch.where(take_descent, residual, chosen_residual)
            chosen_halvings = torch.where(
                take, torch.full_like(chosen_halvings, halving), chosen_halvings
            )
            constrained_stationary = constrained_stationary | take_stationary
            accepted = accepted | take

        if not bool(accepted.all()):
            failed = torch.nonzero(~accepted, as_tuple=False).flatten().tolist()
            raise LineSearchError(f"line search exhausted for batch rows {failed}")
        states = chosen_states
        energies.append(chosen_energy)
        step_columns.append(chosen_step)
        gradient_columns.append(gradient_norm)
        directional_columns.append(chosen_directional)
        residual_columns.append(chosen_residual)
        halving_columns.append(chosen_halvings)
        stationary_columns.append(constrained_stationary)

    def stack(columns: list[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if columns:
            return torch.stack(columns, dim=1)
        return torch.empty((batch, 0), dtype=dtype)

    trace = SettleTrace(
        energies=torch.stack(energies, dim=1),
        step_lengths=stack(step_columns, states.dtype),
        gradient_norms=stack(gradient_columns, states.dtype),
        directional_terms=stack(directional_columns, states.dtype),
        armijo_residuals=stack(residual_columns, states.dtype),
        halvings=stack(halving_columns, torch.int64),
        constrained_stationary=stack(stationary_columns, torch.bool).to(torch.bool),
    )
    return states, trace


def independent_matched_settle(
    initial_states: torch.Tensor,
    centroids: torch.Tensor,
    radius: float,
    *,
    temperature: float,
    steps: int,
    initial_step_radius_fraction: float,
    armijo_c: float,
    max_halvings: int,
    stationary_tolerance: float,
    energy_tolerance: float,
    cap_tolerance: float,
) -> torch.Tensor:
    """Separately structured implementation of the same registered optimizer."""

    anchors = initial_states
    current = initial_states.clone()
    base_step = radius * initial_step_radius_fraction
    centroid_offsets = 0.5 * torch.sum(centroids**2, dim=1)
    for _ in range(steps):
        logits = current @ centroids.T - centroid_offsets
        scaled_logits = logits / temperature
        shifted_logits = (
            scaled_logits - torch.max(scaled_logits, dim=1, keepdim=True).values
        )
        weights = torch.exp(shifted_logits)
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        downhill = weights @ centroids - current
        norms = torch.sqrt(torch.sum(downhill**2, dim=1))
        moving = norms > stationary_tolerance
        directions = downhill / norms.clamp_min(stationary_tolerance).unsqueeze(1)
        old_energy = 0.5 * torch.sum(current**2, dim=1)
        old_energy -= temperature * torch.logsumexp(logits / temperature, dim=1)
        gradient = -downhill
        anchor_delta = current - anchors
        at_boundary = (
            torch.sqrt(torch.sum(anchor_delta**2, dim=1)) >= radius - cap_tolerance
        )
        points_out = torch.sum(anchor_delta * directions, dim=1) > 0.0
        done = ~moving
        next_state = current.clone()
        for halving in range(max_halvings + 1):
            waiting = ~done
            if not bool(waiting.any()):
                break
            trial = current + (base_step / float(2**halving)) * directions
            delta = trial - anchors
            delta_norm = torch.sqrt(torch.sum(delta**2, dim=1))
            outside = delta_norm > radius
            boundary = anchors + radius * delta / delta_norm.clamp_min(1e-15).unsqueeze(
                1
            )
            trial = torch.where(outside.unsqueeze(1), boundary, trial)
            move = trial - current
            move_norm = torch.sqrt(torch.sum(move**2, dim=1))
            no_move = move_norm <= stationary_tolerance
            directional = torch.sum(gradient * move, dim=1)
            trial_logits = trial @ centroids.T - centroid_offsets
            trial_energy = 0.5 * torch.sum(trial**2, dim=1)
            trial_energy -= temperature * torch.logsumexp(
                trial_logits / temperature, dim=1
            )
            armijo = (
                trial_energy <= old_energy + armijo_c * directional + energy_tolerance
            )
            take_stationary = waiting & no_move & at_boundary & points_out
            take_descent = waiting & (~no_move) & (directional < 0.0) & armijo
            take = take_stationary | take_descent
            next_state = torch.where(take_descent.unsqueeze(1), trial, next_state)
            done = done | take
        if not bool(done.all()):
            failed = torch.nonzero(~done, as_tuple=False).flatten().tolist()
            raise LineSearchError(f"independent line search exhausted for {failed}")
        current = next_state
    return current
