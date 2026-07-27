"""
Compare Mosscap's 2D (y-invariant) self-similar Gaussian conduction test
against the 1D reference python code in
/workspaces/Mosscap/tmp/cherry_bifrost_self_similar. Same physical setup:
rho0=1e-12 kg/m^3, T=1e4+9.9e5*exp(-x^2/(1e5)^2), domain x in [-1e6,1e6] m,
kappa0=1e-11 W/m/K^(7/2) Spitzer, periodic BCs, max_time=3.2s. Grids are
identical (dx=8000m, same ghost padding), so no interpolation is needed for
the x-axis comparison.

NOTE: the reference snapshots now point at tmp/snapshots_sts -- the original
tmp/cherry_bifrost_self_similar/snapshots turned out to have been run with
*implicit* conduction, not STS, despite the name. tmp/snapshots_sts is the
genuine STS reference.

HyperTc (gss_hypertc.yaml) requires two changes from its naive defaults to
behave here: HYPERTC_IN_FLUX_VECTOR=true (Eos.hpp, a compile-time flag --
routes the heat flux through the Riemann solver rather than a pure
central-difference source term) and max_cfl=0.01. Without the flux-vector
change it blows up to NaN; with the flux-vector change alone (default
max_cfl) it stays finite but develops a spurious double-peaked/crater
profile with outward-propagating spikes -- a real finite-speed-wave
artifact, not diffusion. Both changes together give a smooth, physically
sane profile that in fact matches the reference *better* than STS/Explicit
here (~1% vs ~4-12%), despite HyperTc having no flux-saturation
implementation at all.
"""
import glob

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AVG_MASS = 1.0
ION_FRAC = 1.0
K_B = 1.380649e-23
P_MASS = 1.6737830080950003e-27

CASES = {
    "STS (2D)": "output_gss_sts",
    "Explicit (2D)": "output_gss_explicit",
    "STS (1D)": "output_gss_sts_1d",
    "Explicit (1D)": "output_gss_explicit_1d",
    "HyperTc": "output_gss_hypertc",
}
COLORS = {
    "STS (2D)": "C0", "Explicit (2D)": "C1",
    "STS (1D)": "C2", "Explicit (1D)": "C3",
    "HyperTc": "C4",
    "1D reference": "k",
}
LINESTYLES = {
    "STS (2D)": "--", "Explicit (2D)": "--",
    "STS (1D)": ":", "Explicit (1D)": ":",
    "HyperTc": "-.",
}

REF_DIR = "/workspaces/Mosscap/tmp/snapshots_sts"


def load_case(prefix, basedir="."):
    # NOTE: glob must match exactly 5 trailing digits -- "output_gss_explicit_*.nc"
    # would also match "output_gss_explicit_1d_00000.nc" and silently mix the
    # two runs' frames together.
    files = sorted(glob.glob(f"{basedir}/{prefix}_[0-9][0-9][0-9][0-9][0-9].nc"))
    if not files:
        raise FileNotFoundError(f"No output files found for prefix {prefix!r}")
    times, press, rho = [], [], []
    x = None
    for f in files:
        ds = xr.open_dataset(f)
        ipre = ds.attrs["ipre"]
        irho = ds.attrs["irho"]
        j0 = ds["W"].shape[2] // 2
        press.append(ds["W"].values[ipre, 0, j0, :])
        rho.append(ds["W"].values[irho, 0, j0, :])
        times.append(float(ds["time"].values.item()))
        x = ds["x"].values
        ds.close()
    order = np.argsort(times)
    times = np.array(times)[order]
    press = np.array(press)[order]
    rho = np.array(rho)[order]
    keep = np.concatenate(([True], np.diff(times) > 1e-12))
    return dict(times=times[keep], x=x, press=press[keep], rho=rho[keep])


def load_ref(basedir=REF_DIR):
    files = sorted(glob.glob(f"{basedir}/snap_*.nc"))
    times, T, x = [], [], None
    for f in files:
        ds = xr.open_dataset(f)
        W = ds["W"].values  # (eq=4, x): Rho, Vel, Pres, IonE
        y = ds["y"].values
        rho, pres = W[0], W[2]
        n = rho / P_MASS
        T.append(pres / ((1.0 + y) * n * K_B))
        times.append(float(ds.attrs["time"]))
        x = ds["x"].values
        ds.close()
    order = np.argsort(times)
    times = np.array(times)[order]
    T = np.array(T)[order]
    keep = np.concatenate(([True], np.diff(times) > 1e-12))
    return dict(times=times[keep], x=x, T=T[keep])


def pres_to_T(pres, rho):
    n_baryon = rho / (AVG_MASS * P_MASS)
    return pres / ((1.0 + ION_FRAC) * n_baryon * K_B)


def main():
    data = {name: load_case(prefix) for name, prefix in CASES.items()}
    ref = load_ref()
    for name, d in data.items():
        print(f"{name}: {len(d['times'])} frames, t in [{d['times'][0]:.2f}, {d['times'][-1]:.2f}]")
    print(f"1D reference: {len(ref['times'])} frames, t in [{ref['times'][0]:.2f}, {ref['times'][-1]:.2f}]")

    n_times = len(ref["times"])

    # --- 1. Profile comparison at each output time ---
    fig1, axes = plt.subplots(1, n_times, figsize=(3.6 * n_times, 4.2), sharey=False)
    for col in range(n_times):
        ax = axes[col]
        t_target = ref["times"][col]
        ax.plot(ref["x"] / 1e3, ref["T"][col], color=COLORS["1D reference"], lw=2.5, alpha=0.5,
                 label="1D reference")
        for name, d in data.items():
            i = int(np.argmin(np.abs(d["times"] - t_target)))
            T = pres_to_T(d["press"][i], d["rho"][i])
            ax.plot(d["x"] / 1e3, T, color=COLORS[name], ls=LINESTYLES[name], label=name)
        ax.set_xlabel("x [km]")
        ax.set_title(f"t={t_target:.2f} s")
        if col == 0:
            ax.set_ylabel("Temperature [K]")
            ax.legend(fontsize=8)
    fig1.suptitle("Self-similar Gaussian conduction: Mosscap (2D, y-invariant) vs 1D reference")
    fig1.tight_layout()
    fig1.savefig("gss_verify_profiles.png", dpi=140)
    print("Saved gss_verify_profiles.png")

    # --- 2. Error vs reference over time ---
    # NOTE: max-abs-pointwise error is dominated by the steep Gaussian flanks,
    # where a sub-cell-scale horizontal offset between the two solutions
    # produces a large *vertical* difference despite the curves looking
    # (and being) very close -- not a meaningful measure of agreement here.
    # Peak-value error is a much cleaner summary for this profile shape.
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, d in data.items():
        peak_errs, pointwise_errs = [], []
        for i, t in enumerate(d["times"]):
            j = int(np.argmin(np.abs(ref["times"] - t)))
            if abs(ref["times"][j] - t) > 1e-6:
                peak_errs.append(np.nan)
                pointwise_errs.append(np.nan)
                continue
            T = pres_to_T(d["press"][i], d["rho"][i])
            peak_errs.append(np.abs(T.max() - ref["T"][j].max()) / ref["T"][j].max())
            pointwise_errs.append(np.max(np.abs(T - ref["T"][j])) / (ref["T"][j].max() - ref["T"][j].min() + 1e-30))
        ax2.plot(d["times"], peak_errs, marker="o", color=COLORS[name], label=name)
        ax3.plot(d["times"], pointwise_errs, marker="o", color=COLORS[name], label=name)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("|T_peak,mosscap - T_peak,ref| / T_peak,ref")
    ax2.set_title("Peak-temperature relative error")
    ax2.legend(fontsize=8)
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("max|T_mosscap - T_ref| / (T_ref range)")
    ax3.set_yscale("log")
    ax3.set_title("Pointwise max error (flank-shift dominated, see note)")
    fig2.tight_layout()
    fig2.savefig("gss_verify_error.png", dpi=140)
    print("Saved gss_verify_error.png")

    # --- 3. Print summary table ---
    print("\n=== Peak temperature at each output time ===")
    print(f"  {'t':>6s}  {'1D ref':>10s}", "  ".join(f"{n:>10s}" for n in data))
    for col in range(n_times):
        t_target = ref["times"][col]
        row = f"  {t_target:6.2f}  {ref['T'][col].max():10.4e}"
        for name, d in data.items():
            i = int(np.argmin(np.abs(d["times"] - t_target)))
            T = pres_to_T(d["press"][i], d["rho"][i])
            row += f"  {T.max():10.4e}"
        print(row)


if __name__ == "__main__":
    main()
