# Saturated conduction test bed

Cherry et al. (2024, §3.2) coronal Gaussian diffusing along the field. ρ = 10⁻¹² kg m⁻³,
T = 10⁴ + 9.9×10⁵ exp(−x²/(10⁵ m)²), x ∈ [−1, 1] Mm (periodic), Spitzer κ₀ = 10⁻¹¹, field along x. It is pure
conduction (hydro fluxes off), 250 cells, 2D y-invariant, run to t = 3.2 s with max_cfl = 0.5 (S ≈ 50–200). At this
density the initial Spitzer flux at the front exceeds the Cowie & McKee limit 5φ√(p/ρ)·p by 10–100×, so
saturation controls the early evolution.

- **Scan:** φ ∈ {off, 1.1, 0.3, 0.1}.
- **Methods:**
  - STS;
  - the old cell-centred HTC: `fluid_type: hypertconly`, since its flux lives in the flux vector, which
    `zero_hydro_flux` would remove;
  - the old HTC again at max_cfl = 0.01;
  - hybrid2: `s_max` 10 with the limiter, and with no substeps;
  - hybrid2 1D, at φ = 0.3, to exercise the 1D code path.
- **Reference:** an independent 1D solution of the saturated parabolic equation in `testbed.py`, using the
  explicit/STS face flux along the field: κ_f the mean of the two cells, the upwind pressure, the face-mean density,
  F = f·F_Spitzer with f = q_sat/(q_sat + |F_Spitzer|). SSP-RK2 at 0.3 dx²/χ_max, at 8× resolution, block-averaged
  onto the Mosscap cells. The 4×-against-8× difference is 2e-5 (no saturation) to 9e-4 (φ = 0.1).

```bash
python testbed.py run          # all Mosscap runs (~1 s each; ~10 s for old HTC at cfl 0.01), compacted to npz
python testbed.py reference    # 1D references (~5 min, 8 processes)
python testbed.py analyse      # table + saturated_conduction.png
```

## Results (October 2026, branch `hybrid2-conduction`)

Worst L1(T)/ΔT₀ over t ≤ 3.2 s against the reference, with the peak T and the front position (T above 10% of the
peak excess) relative to the reference at t = 3.2 s:

| | off | φ = 1.1 | φ = 0.3 | φ = 0.1 |
|---|---|---|---|---|
| hybrid2, s_max 10, limiter | 5.7e-4 | 4.9e-4 | 1.4e-3 | 2.4e-3 |
| hybrid2, no substeps | 1.5e-2 | 1.6e-2 | 1.0e-2 | 6.2e-3 |
| STS | 6.6e-4 | 1.9e-3 | 3.7e-3 (front +2.5%) | 4.8e-3 (front +7.3%) |
| old HTC, same CFL | 4.5e-2 (peak 1.48 at t = 0.8) | 2.5e-2 | 9.9e-3 | 3.8e-3 |
| old HTC, max_cfl 0.01 (50× the steps) | 2.0e-4 | 5.0e-4 | 7.2e-4 | 1.0e-3 |
| hybrid2 1D | | | 1.1e-3 | |

- **At the same (hydro) step, hybrid2 with substeps is the most accurate** at every φ: peak within 0.7% and the
  front within the cell size, except +3.6% at φ = 0.1, t = 3.2 s.
- **STS's front runs ahead** as saturation strengthens. Its spatial operator is the reference's, so this is
  time-integration error from large RKL2 supersteps.
- **The old HTC at the same step** has the Cattaneo front artefacts of its unlimited centred scheme (double peaks
  at early times). It is accurate only at max_cfl = 0.01, i.e. S ≈ 1.
- **Without substeps (S ≈ 100+), hybrid2's error is 3–10× larger** but stays bounded, with no overshoot beyond 7%.
- **Energy:** conserved to ≤ 1e-13 by every method.
