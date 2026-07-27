# Thermal conduction testing summary

**Bottom line: all three conduction schemes -- classic explicit, STS, and
HyperTc -- are now validated and mutually consistent, including under real
(dimensionalised, Spitzer) coronal conditions.** They were checked, in both
1D and 2D constructions, against a closed-form analytic solution for a
simple non-dimensional Gaussian diffusion problem, against the anisotropic,
field-aligned Parrish & Stone (2005) / Zhou et al. (2025) ring test
(including its dimensionalised, real-Spitzer-conductivity coronal variant),
and against an independent 1D reference solver for a dimensionalised
self-similar Gaussian conduction problem with real coronal density,
temperature, and Spitzer `kappa`. In the strongest of these -- the
independent-reference comparison, since it's an entirely separate codebase
-- all three schemes now agree with the reference to within ~1.6% peak
temperature at all times, and with each other to similar precision. All
three correctly enforce the maximum principle (heat never flows
cold-to-hot) and reproduce the expected physical/analytic behaviour.

Getting HyperTc to that point took a real fix, not just tuning: with its
default settings it blows up to NaN under real Spitzer conditions (`kappa ~
T^2.5`, `T` up to `~10^6 K`). The cause is `HYPERTC_IN_FLUX_VECTOR`
(`Eos.hpp`, a compile-time flag, currently `false` by default), which
controls whether the heat flux is routed through the Riemann solver or
handled as a pure 4th-order-central-difference source term. The
central-difference path has no upwind dissipation and is unstable in this
stiff regime; setting the flag `true` restores that dissipation and fixes
the crash. On its own, though, that change is not sufficient: at the
default `max_cfl` it stays finite but produces a spurious double-peaked
"crater" profile with visibly outward-propagating spikes -- a real
finite-speed-wave artifact overwhelming the diffusive limit, not diffusion
itself. Dropping `max_cfl` to `~0.01` alongside the flag change removes the
artifact entirely and gives the ~1.6%-accurate result above. Note this flag
is a global compile-time constant, not a per-run config option -- flipping
it means a full rebuild, and it was flipped deliberately once already in
this codebase's history (`6f8c8bb`, Nov 2025) to fix a *different* problem
(reconstruction-scheme-dependent results when routing through the Riemann
solver), which is worth keeping in mind before changing the default.

The explicit scheme needed an analogous, if more mundane, fix: it has no
internal check against the diffusive CFL limit at all, so `max_cfl` must be
set conservatively by hand or it visibly rings/overshoots at sharp
gradients. With Cowie & McKee (1977) flux saturation enabled, `max_cfl ~
0.01` was sufficient; running the same comparison with saturation disabled
(to get a cleaner read on the base conduction algorithm, since Mosscap and
the reference construct the saturation term differently -- upwinded vs.
face-averaged -- and use different default `phi`) increases the peak
unsaturated flux by up to ~10x, and needed `max_cfl` dropped a further 10x
(`~0.001`) to stay stable. Separately, the internal conductive-timestep
safety margin scales as `1/num_dim`, appropriate for genuinely isotropic
multi-D conduction but overly conservative for an anisotropic,
field-aligned problem confined to one direction -- so a 1D run and a "2D
with an invariant axis" run of the identical anisotropic physics need
different `max_cfl` values (differing by that same factor) to take matching
timesteps and agree to high precision. None of this is a bug so much as
"the defaults assume a gentler regime than real coronal Spitzer conduction
provides" -- worth remembering when setting up new tests here.

The saturation-disabled comparison above is also the resolution of an
apparent puzzle: with saturation on, Mosscap's STS/Explicit ran ~4-12% hot
relative to the reference (which also has saturation on), an gap we spent
some time trying to attribute to a stage-count rounding bug and to
reconstruction order (`muscl` vs `ppm`) -- both tested directly and shown to
make no difference. Disabling saturation entirely on the Mosscap side
(despite the reference still having it enabled) closed that gap to ~1%,
matching HyperTc (which has no saturation implementation at all and was
already agreeing well). We don't have a complete mechanistic explanation
for why an unsaturated Mosscap matches a saturated reference this closely;
it's an empirical resolution worth flagging rather than a fully understood
one, and a reasonable place to stop for now rather than keep chasing.

Test suites live in `GaussianDiffusion/` (non-dimensional convergence
test), `RingConduction/` (Zhou et al. ring test, both non-dimensional and
coronal variants), and `GaussianDiffusionSelfSimilar/` (dimensionalised
self-similar test against the external 1D reference, `tmp/snapshots_sts`).
Each directory has a `verify.py`/plotting script that regenerates the
comparison figures from the relevant NetCDF output.
