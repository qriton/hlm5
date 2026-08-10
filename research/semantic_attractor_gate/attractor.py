"""Certified polynomial attractor dynamics for the HLM5 S1 routing gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class LineSearchError(RuntimeError):
    """Raised when a finite Armijo step cannot be found."""


@dataclass(frozen=True)
class SettleTrace:
    final_state: torch.Tensor
    energies: torch.Tensor
    step_sizes: torch.Tensor
    gradient_norms: torch.Tensor
    armijo_residuals: torch.Tensor
    halvings: torch.Tensor


def unit(x: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Normalize the final axis without changing dtype or device."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def require_unit_float64(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    tolerance: float = 1e-10,
) -> None:
    if queries.ndim != 2 or keys.ndim != 2:
        raise ValueError("queries and keys must both have shape [rows, dim]")
    if queries.shape[1] != keys.shape[1]:
        raise ValueError("queries and keys must share the same embedding dimension")
    if queries.dtype != torch.float64 or keys.dtype != torch.float64:
        raise TypeError("registered dynamics require torch.float64")
    if queries.device.type != "cpu" or keys.device.type != "cpu":
        raise ValueError("registered dynamics require CPU tensors")
    if not bool(torch.isfinite(queries).all()) or not bool(torch.isfinite(keys).all()):
        raise ValueError("queries and keys must be finite")
    query_error = torch.abs(queries.norm(dim=-1) - 1.0).max()
    key_error = torch.abs(keys.norm(dim=-1) - 1.0).max()
    if float(max(query_error, key_error)) > tolerance:
        raise ValueError("queries and keys must be unit normalized")


def polynomial_energy(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    degree: int = 5,
) -> torch.Tensor:
    """Evaluate E(q) = -sum(relu(Kq) ** (d + 1)) / (d + 1)."""
    if degree < 1:
        raise ValueError("degree must be positive")
    overlaps = queries @ keys.T
    return -torch.relu(overlaps).pow(degree + 1).sum(dim=-1) / float(degree + 1)


def descent_direction(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    degree: int = 5,
) -> torch.Tensor:
    """Return the negative Riemannian gradient on the unit sphere."""
    overlaps = queries @ keys.T
    attraction = torch.relu(overlaps).pow(degree) @ keys
    radial = (queries * attraction).sum(dim=-1, keepdim=True)
    return attraction - queries * radial


def static_cosine_predictions(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = queries @ keys.T
    return scores.argmax(dim=-1), scores.max(dim=-1).values


def static_degree_predictions(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    degree: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.relu(queries @ keys.T).pow(degree)
    return scores.argmax(dim=-1), scores.max(dim=-1).values


def certified_settle(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    degree: int = 5,
    steps: int = 8,
    initial_step: float = 1.0,
    armijo_c: float = 1e-4,
    max_halvings: int = 24,
    stationary_tolerance: float = 1e-12,
    energy_tolerance: float = 1e-12,
) -> SettleTrace:
    """Run deterministic batched Riemannian descent with per-row line search."""
    require_unit_float64(queries, keys)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0.0 < armijo_c < 1.0:
        raise ValueError("armijo_c must lie strictly between zero and one")
    if initial_step <= 0.0 or max_halvings < 0:
        raise ValueError("invalid line-search configuration")

    state = queries.clone()
    energy = polynomial_energy(state, keys, degree=degree)
    energy_rows = [energy.clone()]
    step_rows: list[torch.Tensor] = []
    norm_rows: list[torch.Tensor] = []
    residual_rows: list[torch.Tensor] = []
    halving_rows: list[torch.Tensor] = []

    for _ in range(steps):
        direction = descent_direction(state, keys, degree=degree)
        norm_sq = direction.square().sum(dim=-1)
        gradient_norm = norm_sq.sqrt()
        stationary = gradient_norm <= stationary_tolerance

        next_state = state.clone()
        next_energy = energy.clone()
        accepted_step = torch.zeros_like(energy)
        accepted_halvings = torch.zeros(
            energy.shape,
            dtype=torch.int64,
            device=energy.device,
        )
        accepted = stationary.clone()
        trial_step = torch.full_like(energy, float(initial_step))

        for halving in range(max_halvings + 1):
            pending = ~accepted
            if not bool(pending.any()):
                break

            candidate = unit(
                state[pending] + trial_step[pending, None] * direction[pending]
            )
            candidate_energy = polynomial_energy(candidate, keys, degree=degree)
            bound = (
                energy[pending]
                - armijo_c * trial_step[pending] * norm_sq[pending]
                + energy_tolerance
            )
            local_ok = torch.isfinite(candidate_energy) & (candidate_energy <= bound)
            pending_indices = torch.nonzero(pending, as_tuple=False).flatten()
            accepted_indices = pending_indices[local_ok]
            rejected_indices = pending_indices[~local_ok]

            if accepted_indices.numel() > 0:
                next_state[accepted_indices] = candidate[local_ok]
                next_energy[accepted_indices] = candidate_energy[local_ok]
                accepted_step[accepted_indices] = trial_step[accepted_indices]
                accepted_halvings[accepted_indices] = halving
                accepted[accepted_indices] = True
            if rejected_indices.numel() > 0:
                trial_step[rejected_indices] *= 0.5

        if not bool(accepted.all()):
            failed = torch.nonzero(~accepted, as_tuple=False).flatten().tolist()
            raise LineSearchError(f"Armijo line search exhausted for rows {failed[:16]}")

        armijo_residual = next_energy - (
            energy - armijo_c * accepted_step * norm_sq
        )
        state = next_state
        energy = next_energy
        energy_rows.append(energy.clone())
        step_rows.append(accepted_step)
        norm_rows.append(gradient_norm)
        residual_rows.append(armijo_residual)
        halving_rows.append(accepted_halvings)

    batch = queries.shape[0]
    empty_float = torch.empty((batch, 0), dtype=torch.float64)
    empty_int = torch.empty((batch, 0), dtype=torch.int64)
    return SettleTrace(
        final_state=state,
        energies=torch.stack(energy_rows, dim=1),
        step_sizes=torch.stack(step_rows, dim=1) if step_rows else empty_float,
        gradient_norms=torch.stack(norm_rows, dim=1) if norm_rows else empty_float,
        armijo_residuals=(
            torch.stack(residual_rows, dim=1) if residual_rows else empty_float
        ),
        halvings=torch.stack(halving_rows, dim=1) if halving_rows else empty_int,
    )


def independent_matched_settle(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    degree: int = 5,
    steps: int = 8,
    initial_step: float = 1.0,
    armijo_c: float = 1e-4,
    max_halvings: int = 24,
    stationary_tolerance: float = 1e-12,
    energy_tolerance: float = 1e-12,
) -> torch.Tensor:
    """Independent cache-side implementation of the registered optimizer."""
    require_unit_float64(queries, keys)
    result = queries.clone()

    def energy_fn(rows: torch.Tensor) -> torch.Tensor:
        similarities = torch.einsum("bd,kd->bk", rows, keys)
        terms = torch.clamp_min(similarities, 0.0).pow(degree + 1)
        return -torch.sum(terms, dim=1) / float(degree + 1)

    current_energy = energy_fn(result)
    for _ in range(steps):
        similarities = torch.einsum("bd,kd->bk", result, keys)
        weights = torch.clamp_min(similarities, 0.0).pow(degree)
        pull = torch.einsum("bk,kd->bd", weights, keys)
        tangent = pull - result * torch.sum(result * pull, dim=1, keepdim=True)
        norm_sq = torch.sum(tangent * tangent, dim=1)
        done = torch.sqrt(norm_sq) <= stationary_tolerance
        accepted = done.clone()
        eta = torch.full_like(current_energy, initial_step)
        updated = result.clone()
        updated_energy = current_energy.clone()

        for _halving in range(max_halvings + 1):
            pending = ~accepted
            if not bool(pending.any()):
                break
            trial = result[pending] + eta[pending, None] * tangent[pending]
            trial = trial / torch.linalg.vector_norm(trial, dim=1, keepdim=True)
            trial_energy = energy_fn(trial)
            limit = (
                current_energy[pending]
                - armijo_c * eta[pending] * norm_sq[pending]
                + energy_tolerance
            )
            ok = torch.isfinite(trial_energy) & (trial_energy <= limit)
            indices = torch.nonzero(pending, as_tuple=False).flatten()
            good = indices[ok]
            bad = indices[~ok]
            if good.numel() > 0:
                updated[good] = trial[ok]
                updated_energy[good] = trial_energy[ok]
                accepted[good] = True
            if bad.numel() > 0:
                eta[bad] /= 2.0

        if not bool(accepted.all()):
            failed = torch.nonzero(~accepted, as_tuple=False).flatten().tolist()
            raise LineSearchError(
                f"independent Armijo line search exhausted for rows {failed[:16]}"
            )
        result = updated
        current_energy = updated_energy

    return result
