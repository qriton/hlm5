"""Class-balanced spherical attractor dynamics for the I1 gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class LineSearchError(RuntimeError):
    """Raised when a non-stationary row cannot satisfy the registered search."""


@dataclass(frozen=True)
class SettleTrace:
    """Per-row diagnostics from a registered constrained settle."""

    energies: torch.Tensor
    step_angles: torch.Tensor
    gradient_norms: torch.Tensor
    directional_terms: torch.Tensor
    armijo_residuals: torch.Tensor
    halvings: torch.Tensor
    constrained_stationary: torch.Tensor


def unit(value: torch.Tensor, dim: int = -1, eps: float = 1e-15) -> torch.Tensor:
    """Normalize along ``dim`` and fail on a zero or non-finite norm."""

    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    if not bool(torch.isfinite(norm).all()):
        raise ValueError("cannot normalize non-finite tensor")
    if bool((norm <= eps).any()):
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def memory_energy(
    states: torch.Tensor,
    prototypes: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Evaluate the registered mean degree-``degree + 1`` energy."""

    activations = torch.relu(states @ prototypes.T)
    return -(activations.pow(degree + 1).mean(dim=-1) / float(degree + 1))


def negative_riemannian_gradient(
    states: torch.Tensor,
    prototypes: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Return the negative energy gradient projected onto the unit sphere."""

    activations = torch.relu(states @ prototypes.T)
    field = activations.pow(degree) @ prototypes / float(prototypes.shape[0])
    return field - states * (states * field).sum(dim=-1, keepdim=True)


def angular_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Stable row-wise spherical distance for unit tensors."""

    cosine = (left * right).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def sphere_exp(
    states: torch.Tensor,
    unit_tangents: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Spherical exponential map for row-wise unit tangent directions."""

    angle_column = angles.unsqueeze(-1)
    proposal = torch.cos(angle_column) * states + torch.sin(angle_column) * unit_tangents
    return unit(proposal)


def sphere_log(
    origins: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-wise tangent log-map displacements and angular distances."""

    cosine = (origins * targets).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(cosine)
    tangent = targets - cosine.unsqueeze(-1) * origins
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    safe_norm = tangent_norm.clamp_min(eps)
    displacement = tangent * (angles / safe_norm).unsqueeze(-1)
    displacement = torch.where(
        (angles <= eps).unsqueeze(-1),
        torch.zeros_like(displacement),
        displacement,
    )
    return displacement, angles


def project_to_cap(
    origins: torch.Tensor,
    proposals: torch.Tensor,
    radius: float,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project proposals outside a spherical cap onto its geodesic boundary."""

    distances = angular_distance(origins, proposals)
    outside = distances > radius
    if not bool(outside.any()):
        return proposals, outside

    cosine = (origins * proposals).sum(dim=-1).clamp(-1.0, 1.0)
    tangent = proposals - cosine.unsqueeze(-1) * origins
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    safe_tangent = tangent / tangent_norm.clamp_min(eps)
    boundary = torch.cos(torch.as_tensor(radius, dtype=origins.dtype)) * origins
    boundary = boundary + torch.sin(torch.as_tensor(radius, dtype=origins.dtype)) * safe_tangent
    boundary = unit(boundary)
    projected = torch.where(outside.unsqueeze(-1), boundary, proposals)
    return projected, outside


def class_degree_scores(
    states: torch.Tensor,
    prototypes: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Mean degree-``degree`` score over equal prototype support per intent."""

    if prototypes.ndim != 3:
        raise ValueError("prototypes must have shape [classes, supports, dim]")
    flat = prototypes.reshape(-1, prototypes.shape[-1])
    scores = torch.relu(states @ flat.T).pow(degree)
    return scores.reshape(states.shape[0], prototypes.shape[0], prototypes.shape[1]).mean(dim=-1)


def centroid_scores(states: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Cosine scores for unit states and unit class centroids."""

    return states @ centroids.T


def constrained_settle(
    initial_states: torch.Tensor,
    flat_prototypes: torch.Tensor,
    radius: float,
    *,
    degree: int,
    steps: int,
    initial_step_radius_fraction: float,
    armijo_c: float,
    max_halvings: int,
    stationary_tolerance: float,
    energy_tolerance: float,
    cap_tolerance: float,
) -> tuple[torch.Tensor, SettleTrace]:
    """Run the registered vectorized constrained Riemannian descent."""

    if initial_states.ndim != 2 or flat_prototypes.ndim != 2:
        raise ValueError("states and prototypes must be matrices")
    if initial_states.shape[1] != flat_prototypes.shape[1]:
        raise ValueError("state/prototype dimensions differ")
    if not 0.0 < radius < torch.pi / 2:
        raise ValueError("registered radius must lie in (0, pi/2)")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    origins = initial_states
    states = initial_states.clone()
    energy_columns = [memory_energy(states, flat_prototypes, degree)]
    step_columns: list[torch.Tensor] = []
    gradient_columns: list[torch.Tensor] = []
    directional_columns: list[torch.Tensor] = []
    residual_columns: list[torch.Tensor] = []
    halving_columns: list[torch.Tensor] = []
    stationary_columns: list[torch.Tensor] = []

    batch = states.shape[0]
    base_angle = float(radius) * initial_step_radius_fraction

    for _ in range(steps):
        current_energy = energy_columns[-1]
        negative_gradient = negative_riemannian_gradient(states, flat_prototypes, degree)
        gradient_norm = torch.linalg.vector_norm(negative_gradient, dim=-1)
        active = gradient_norm > stationary_tolerance
        unit_direction = negative_gradient / gradient_norm.clamp_min(stationary_tolerance).unsqueeze(-1)
        energy_gradient = -negative_gradient
        distance_from_origin = angular_distance(origins, states)
        at_cap_boundary = distance_from_origin >= radius - cap_tolerance
        points_outward = (origins * unit_direction).sum(dim=-1) < 0.0

        accepted = ~active
        chosen_states = states.clone()
        chosen_energy = current_energy.clone()
        chosen_angle = torch.zeros(batch, dtype=states.dtype)
        chosen_directional = torch.zeros(batch, dtype=states.dtype)
        chosen_residual = torch.zeros(batch, dtype=states.dtype)
        chosen_halvings = torch.zeros(batch, dtype=torch.int64)
        constrained_stationary = ~active

        for halving in range(max_halvings + 1):
            pending = ~accepted
            if not bool(pending.any()):
                break

            trial_angles = torch.full(
                (batch,),
                base_angle / float(2**halving),
                dtype=states.dtype,
            )
            trial = sphere_exp(states, unit_direction, trial_angles)
            trial, _ = project_to_cap(origins, trial, radius)
            displacement, actual_angle = sphere_log(states, trial)
            directional = (energy_gradient * displacement).sum(dim=-1)
            trial_energy = memory_energy(trial, flat_prototypes, degree)
            no_movement = actual_angle <= stationary_tolerance
            descent = directional < 0.0
            residual = trial_energy - (current_energy + armijo_c * directional)
            valid_descent = descent & (residual <= energy_tolerance)
            take_stationary = pending & no_movement & at_cap_boundary & points_outward
            take_descent = pending & (~no_movement) & valid_descent
            take = take_stationary | take_descent

            chosen_states = torch.where(
                take_descent.unsqueeze(-1), trial, chosen_states
            )
            chosen_energy = torch.where(take_descent, trial_energy, chosen_energy)
            chosen_angle = torch.where(take_descent, actual_angle, chosen_angle)
            chosen_directional = torch.where(
                take_descent, directional, chosen_directional
            )
            chosen_residual = torch.where(take_descent, residual, chosen_residual)
            chosen_halvings = torch.where(
                take,
                torch.full_like(chosen_halvings, halving),
                chosen_halvings,
            )
            constrained_stationary = constrained_stationary | take_stationary
            accepted = accepted | take

        if not bool(accepted.all()):
            failed = torch.nonzero(~accepted, as_tuple=False).flatten().tolist()
            raise LineSearchError(f"line search exhausted for batch rows {failed}")

        states = chosen_states
        energy_columns.append(chosen_energy)
        step_columns.append(chosen_angle)
        gradient_columns.append(gradient_norm)
        directional_columns.append(chosen_directional)
        residual_columns.append(chosen_residual)
        halving_columns.append(chosen_halvings)
        stationary_columns.append(constrained_stationary)

    def stack_or_empty(columns: list[torch.Tensor], *, integer: bool = False) -> torch.Tensor:
        if columns:
            return torch.stack(columns, dim=1)
        dtype = torch.int64 if integer else states.dtype
        return torch.empty((batch, 0), dtype=dtype)

    trace = SettleTrace(
        energies=torch.stack(energy_columns, dim=1),
        step_angles=stack_or_empty(step_columns),
        gradient_norms=stack_or_empty(gradient_columns),
        directional_terms=stack_or_empty(directional_columns),
        armijo_residuals=stack_or_empty(residual_columns),
        halvings=stack_or_empty(halving_columns, integer=True),
        constrained_stationary=stack_or_empty(stationary_columns).to(torch.bool),
    )
    return states, trace


def independent_matched_settle(
    initial_states: torch.Tensor,
    flat_prototypes: torch.Tensor,
    radius: float,
    *,
    degree: int,
    steps: int,
    initial_step_radius_fraction: float,
    armijo_c: float,
    max_halvings: int,
    stationary_tolerance: float,
    energy_tolerance: float,
    cap_tolerance: float,
) -> torch.Tensor:
    """Separately structured implementation of the registered state optimizer."""

    anchors = initial_states
    current = initial_states.clone()
    base_angle = radius * initial_step_radius_fraction

    for _ in range(steps):
        dot = current @ flat_prototypes.T
        positive = torch.clamp_min(dot, 0.0)
        field = positive.pow(degree) @ flat_prototypes / flat_prototypes.shape[0]
        downhill = field - current * torch.sum(current * field, dim=1, keepdim=True)
        norms = torch.sqrt(torch.sum(downhill * downhill, dim=1))
        moving = norms > stationary_tolerance
        directions = downhill / torch.clamp_min(norms, stationary_tolerance).unsqueeze(1)
        old_energy = -positive.pow(degree + 1).mean(dim=1) / (degree + 1)
        gradient = -downhill
        distance_from_anchor = torch.acos(
            torch.sum(anchors * current, dim=1).clamp(-1.0, 1.0)
        )
        at_boundary = distance_from_anchor >= radius - cap_tolerance
        points_out = torch.sum(anchors * directions, dim=1) < 0.0

        done = ~moving
        next_state = current.clone()

        for halving in range(max_halvings + 1):
            waiting = ~done
            if not bool(waiting.any()):
                break
            angle = base_angle / float(2**halving)
            trial = unit(torch.cos(torch.tensor(angle, dtype=current.dtype)) * current + torch.sin(torch.tensor(angle, dtype=current.dtype)) * directions)

            anchor_dot = torch.sum(anchors * trial, dim=1).clamp(-1.0, 1.0)
            anchor_angle = torch.acos(anchor_dot)
            beyond = anchor_angle > radius
            anchor_tangent = trial - anchor_dot.unsqueeze(1) * anchors
            anchor_tangent = anchor_tangent / torch.linalg.vector_norm(
                anchor_tangent, dim=1, keepdim=True
            ).clamp_min(1e-15)
            boundary = unit(
                torch.cos(torch.tensor(radius, dtype=current.dtype)) * anchors
                + torch.sin(torch.tensor(radius, dtype=current.dtype)) * anchor_tangent
            )
            trial = torch.where(beyond.unsqueeze(1), boundary, trial)

            local_dot = torch.sum(current * trial, dim=1).clamp(-1.0, 1.0)
            local_angle = torch.acos(local_dot)
            local_tangent = trial - local_dot.unsqueeze(1) * current
            local_norm = torch.linalg.vector_norm(local_tangent, dim=1)
            displacement = local_tangent * (
                local_angle / local_norm.clamp_min(1e-15)
            ).unsqueeze(1)
            displacement = torch.where(
                (local_angle <= 1e-15).unsqueeze(1),
                torch.zeros_like(displacement),
                displacement,
            )
            directional = torch.sum(gradient * displacement, dim=1)
            trial_positive = torch.clamp_min(trial @ flat_prototypes.T, 0.0)
            trial_energy = -trial_positive.pow(degree + 1).mean(dim=1) / (degree + 1)
            no_move = local_angle <= stationary_tolerance
            armijo = trial_energy <= old_energy + armijo_c * directional + energy_tolerance
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
