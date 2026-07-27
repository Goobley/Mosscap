"""
Validate the ring_conduction test (Parrish & Stone 2005 style anisotropic
conduction ring) across the classic-explicit / STS / hyperbolic conduction
implementations. Checks:
  1. Heat never flows from cold to hot (global T stays within [T_outer, T_inner]).
  2. The three implementations produce matching temporal evolution.
"""
import glob

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# NOTE: these match RingConduction.cpp's hardcoded non-coronal defaults --
# not exposed via yaml for this problem, so they're not read from config.
RHO0 = 1.0
T_OUTER = 10.0
T_INNER = 12.0
LENGTH_SCALE = 1.0
GAMMA = 1.6666667
ION_FRAC = 1.0
AVG_MASS = 1.0
K_B = 1.380649e-23
P_MASS = K_B  # forced by RingConduction.cpp's non-coronal branch

CASES = {
    "STS": "output_ring_validate_sts",
    "Explicit": "output_ring_validate_explicit",
    "HyperTc": "output_ring_validate_hypertc",
}

# Matches the snapshot times shown in Zhou et al. 2025, ApJ 978, 72.
TARGET_TIMES = [0.0, 50.0, 100.0, 400.0]


def nearest_idx(times, t):
    return int(np.argmin(np.abs(times - t)))


def load_case(prefix, basedir="."):
    files = sorted(glob.glob(f"{basedir}/{prefix}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No output files found for prefix {prefix!r}")
    times, press = [], []
    x = y = None
    for f in files:
        ds = xr.open_dataset(f)
        ipre = ds.attrs["ipre"]
        press.append(ds["W"].values[ipre, 0, :, :])  # (y, x)
        times.append(float(ds["time"].values.item()))
        x = ds["x"].values
        y = ds["y"].values
        ds.close()
    order = np.argsort(times)
    times = np.array(times)[order]
    press = np.array(press)[order]
    keep = np.concatenate(([True], np.diff(times) > 1e-12))
    return dict(times=times[keep], x=x, y=y, press=press[keep])


def pres_to_T(pres):
    n_baryon = RHO0 / (AVG_MASS * P_MASS)
    return pres / ((1.0 + ION_FRAC) * n_baryon * K_B)


def radial_theta_grid(x, y):
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)
    Theta = np.where(Theta < 0, Theta + 2 * np.pi, Theta)
    return R, Theta


def main():
    data = {name: load_case(prefix) for name, prefix in CASES.items()}
    for name, d in data.items():
        print(f"{name}: {len(d['times'])} frames, t in [{d['times'][0]:.2f}, {d['times'][-1]:.2f}]")

    colors = {"STS": "C0", "Explicit": "C1", "HyperTc": "C2"}

    # --- 1. Maximum-principle / "no cold-to-hot" check ---
    print("\n=== Bounds check: T should stay within [T_outer, T_inner] = "
          f"[{T_OUTER}, {T_INNER}] ===")
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    violations = {}
    for name, d in data.items():
        T = pres_to_T(d["press"])
        tmin = T.reshape(len(d["times"]), -1).min(axis=1)
        tmax = T.reshape(len(d["times"]), -1).max(axis=1)
        ax1.plot(d["times"], tmin, color=colors[name], ls="--", label=f"{name} min")
        ax1.plot(d["times"], tmax, color=colors[name], ls="-", label=f"{name} max")
        over = tmax - T_INNER
        under = T_OUTER - tmin
        violations[name] = (over.max(), under.max())
        print(f"  {name:10s}: max(T) - T_inner = {over.max():+.3e} K   "
              f"T_outer - min(T) = {under.max():+.3e} K")
    ax1.axhline(T_OUTER, color="k", lw=0.8, alpha=0.5)
    ax1.axhline(T_INNER, color="k", lw=0.8, alpha=0.5)
    ax1.set_xlabel("t")
    ax1.set_ylabel("Temperature")
    ax1.set_title("Global T extrema vs time (dashed=min, solid=max)")
    ax1.legend(fontsize=7, ncol=2)
    fig1.tight_layout()
    fig1.savefig("verify_extrema.png", dpi=140)
    print("Saved verify_extrema.png")

    any_violation = any(o > 1e-6 * T_INNER or u > 1e-6 * T_INNER for o, u in violations.values())
    print("RESULT:", "VIOLATION DETECTED" if any_violation else "no maximum-principle violation (heat did not flow cold->hot)")

    # --- 1b. Explicit minimum-temperature report at each target time ---
    print(f"\n=== Minimum temperature in domain at each target time (background = {T_OUTER}) ===")
    for name, d in data.items():
        T = pres_to_T(d["press"])
        vals = ", ".join(
            f"t={t_target:.0f}: {T[nearest_idx(d['times'], t_target)].min():.6f}"
            for t_target in TARGET_TIMES
        )
        print(f"  {name:10s}: {vals}")

    # --- 2. Cross-scheme temporal-evolution comparison ---
    ref = data["STS"]
    R, Theta = radial_theta_grid(ref["x"], ref["y"])

    # Snapshot at the times shown in Zhou et al. 2025 (0, 50, 100, 400), using
    # the nearest available output frame to each (should be exact, since
    # delta_t=50 puts an output boundary right on every target time).
    show_idx = {name: [nearest_idx(d["times"], t) for t in TARGET_TIMES] for name, d in data.items()}
    for name, d in data.items():
        actual = [d["times"][i] for i in show_idx[name]]
        if any(abs(a - t) > 1e-6 for a, t in zip(actual, TARGET_TIMES)):
            print(f"  WARNING: {name} snapshot times {actual} don't exactly match targets {TARGET_TIMES}")

    # Each time slice (column) gets its own colour scale, shared across the 3
    # schemes in that column: the actual dynamic range shrinks by ~an order of
    # magnitude between t=0 (10-12) and t=400 (~10-10.15), so a single shared
    # scale across all columns would wash out the later, more relaxed snapshots.
    col_vmin, col_vmax = [], []
    for col in range(len(TARGET_TIMES)):
        vals = [pres_to_T(data[name]["press"][show_idx[name][col]]) for name in data]
        col_vmin.append(min(v.min() for v in vals))
        col_vmax.append(max(v.max() for v in vals))

    fig2, axes = plt.subplots(len(data), len(TARGET_TIMES), figsize=(3.4 * len(TARGET_TIMES), 3.1 * len(data)),
                               squeeze=False, constrained_layout=True)
    im_by_col = {}
    for row, (name, d) in enumerate(data.items()):
        T = pres_to_T(d["press"])
        for col, fi in enumerate(show_idx[name]):
            ax = axes[row, col]
            im = ax.pcolormesh(d["x"], d["y"], T[fi], vmin=col_vmin[col], vmax=col_vmax[col], cmap="inferno", shading="auto")
            im_by_col[col] = im
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"t={d['times'][fi]:.1f}")
            if col == 0:
                ax.set_ylabel(name)
            ax.text(0.03, 0.03, f"min={T[fi].min():.4f}", transform=ax.transAxes,
                    fontsize=6, color="white", va="bottom", ha="left")
    for col in range(len(TARGET_TIMES)):
        fig2.colorbar(im_by_col[col], ax=axes[:, col], shrink=0.7, label="Temperature")
    fig2.suptitle("Ring conduction: temperature field evolution (cf. Zhou et al. 2025)")
    fig2.savefig("verify_fields.png", dpi=140)
    print("Saved verify_fields.png")

    # --- 3. Azimuthal profile at fixed radius, at each target time: does the
    # ring homogenise in theta the same way across schemes? ---
    r_ring = 0.6 * LENGTH_SCALE
    theta_samples = np.linspace(0, 2 * np.pi, 361)
    xs = r_ring * np.cos(theta_samples)
    ys = r_ring * np.sin(theta_samples)

    fig3, axes3 = plt.subplots(1, len(TARGET_TIMES), figsize=(4.2 * len(TARGET_TIMES), 4), sharey=True)
    for col, t_target in enumerate(TARGET_TIMES):
        ax = axes3[col]
        for name, d in data.items():
            fi = nearest_idx(d["times"], t_target)
            T = pres_to_T(d["press"])
            interp = RegularGridInterpolator((d["y"], d["x"]), T[fi], bounds_error=False, fill_value=None)
            profile = interp(np.column_stack([ys, xs]))
            ax.plot(np.degrees(theta_samples), profile, color=colors[name], label=name)
        ax.axvline(180 - 15, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.axvline(180 + 15, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.set_xlabel("theta [deg]")
        ax.set_title(f"t={t_target:.0f}")
        if col == 0:
            ax.set_ylabel("Temperature")
            ax.legend(fontsize=8)
    fig3.suptitle(f"Azimuthal profile at r={r_ring} (cf. Zhou et al. 2025)")
    fig3.tight_layout()
    fig3.savefig("verify_azimuthal.png", dpi=140)
    print("Saved verify_azimuthal.png")

    # --- 4. Cross-field leakage: T just outside the ring's radial extent ---
    print("\n=== Cross-field leakage: mean T outside the ring's radial band (r<0.4 or r>0.8) ===")
    outside_mask = (R < 0.4 * LENGTH_SCALE) | (R > 0.8 * LENGTH_SCALE)
    for name, d in data.items():
        T = pres_to_T(d["press"])
        mean_outside = T[:, outside_mask].mean(axis=1)
        vals = ", ".join(f"t={t_target:.0f}: {mean_outside[nearest_idx(d['times'], t_target)]:.6f}" for t_target in TARGET_TIMES)
        print(f"  {name:10s}: {vals}  (background={T_OUTER})")

    # --- 5. Pairwise cross-scheme agreement, at each target time ---
    print("\n=== Pairwise max |T_a - T_b| at each target time ===")
    names = list(data.keys())
    for t_target in TARGET_TIMES:
        print(f"  t={t_target:.0f}:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                di, dj = data[names[i]], data[names[j]]
                fi, fj = nearest_idx(di["times"], t_target), nearest_idx(dj["times"], t_target)
                a = pres_to_T(di["press"][fi])
                b = pres_to_T(dj["press"][fj])
                diff = np.abs(a - b)
                print(f"    {names[i]:10s} vs {names[j]:10s}: max={diff.max():.4e} K  (dT={T_INNER - T_OUTER})")


if __name__ == "__main__":
    main()
