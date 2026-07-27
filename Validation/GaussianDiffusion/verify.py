"""
Compare gaussian_diffusion output against the analytic 1D Gaussian-diffusion
solution, for each of the classic-explicit / STS / hyperbolic conduction
configs in this directory.
"""
import glob

import numpy as np
import xarray as xr
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_B = 1.380649e-23
P_MASS_DEFAULT = 1.6737830080950003e-27

CASES = {
    "STS": "aligned_sts.yaml",
    "Explicit": "aligned_explicit.yaml",
    "HyperTc": "aligned_hypertc.yaml",
}


def get_cfg(cfg, path, default=None):
    node = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_case(yaml_path, basedir="."):
    with open(f"{basedir}/{yaml_path}") as f:
        cfg = yaml.safe_load(f)
    prefix = get_cfg(cfg, "output.name")
    files = sorted(glob.glob(f"{basedir}/{prefix}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No output files found for prefix {prefix!r} -- run mosscap --config {yaml_path} first.")

    times, press = [], []
    x = None
    for f in files:
        ds = xr.open_dataset(f)
        ipre = ds.attrs["ipre"]
        press.append(ds["W"].values[ipre, 0, :, :])
        times.append(float(ds["time"].values.item()))
        x = ds["x"].values
        ds.close()

    order = np.argsort(times)
    times = np.array(times)[order]
    press = np.array(press)[order]
    # NOTE: main.cpp writes once when the loop's periodic-output condition
    # fires, then unconditionally once more after the loop exits -- these
    # coincide whenever the run ends exactly on an output boundary.
    keep = np.concatenate(([True], np.diff(times) > 1e-12))
    return dict(cfg=cfg, times=times[keep], x=x, press=press[keep])


def analytic_params(cfg):
    gamma = get_cfg(cfg, "eos.gamma", 1.4)
    y = get_cfg(cfg, "eos.ion_frac", 1.0)
    avg_mass = get_cfg(cfg, "eos.avg_mass", 1.0)
    rho0 = get_cfg(cfg, "problem.base_density", 5e-12)
    T0 = get_cfg(cfg, "problem.base_temperature", 1e6)
    dT = get_cfg(cfg, "problem.delta_temperature", 0.5 * T0)
    kappa0 = get_cfg(cfg, "problem.kappa0", 0.01)
    dimensionless = get_cfg(cfg, "problem.dimensionless", False)
    p_mass = K_B if dimensionless else P_MASS_DEFAULT
    x_start = get_cfg(cfg, "grid.x_start", 0.0)
    x_dim = get_cfg(cfg, "grid.x_dim", 1.0)
    sigma0 = get_cfg(cfg, "problem.sigma0", 0.1 * x_dim)
    x0 = get_cfg(cfg, "problem.center_x", x_start + 0.5 * x_dim)

    # rho * cv, with e_int = P / (gamma - 1) = rho * (1+y) * k_B * T / ((gamma-1) * avg_mass * p_mass)
    cv = (1.0 + y) * K_B / ((gamma - 1.0) * avg_mass * p_mass)
    D = kappa0 / (rho0 * cv)
    return dict(gamma=gamma, y=y, avg_mass=avg_mass, rho0=rho0, T0=T0, dT=dT,
                sigma0=sigma0, x0=x0, D=D, p_mass=p_mass)


def analytic_T(x, t, p):
    sigma_t2 = p["sigma0"] ** 2 + 2.0 * p["D"] * t
    amp = p["dT"] * (p["sigma0"] / np.sqrt(sigma_t2))
    return p["T0"] + amp * np.exp(-((x - p["x0"]) ** 2) / (2.0 * sigma_t2))


def analytic_pres(x, t, p):
    T = analytic_T(x, t, p)
    n_baryon = p["rho0"] / (p["avg_mass"] * p["p_mass"])
    return n_baryon * (1.0 + p["y"]) * K_B * T


def main():
    data = {name: load_case(path) for name, path in CASES.items()}
    params = {name: analytic_params(d["cfg"]) for name, d in data.items()}

    for name, d in data.items():
        p = params[name]
        print(f"{name}: {len(d['times'])} frames, D={p['D']:.4e}, sigma0={p['sigma0']:.4f}, x0={p['x0']:.4f}")

    ref = data["STS"]
    ymid = ref["press"].shape[1] // 2
    x = ref["x"]

    # --- profile overlay at a few times ---
    n_show = min(len(d["times"]) for d in data.values())
    show_idx = sorted(set(int(round(f)) for f in np.linspace(0, n_show - 1, 5)))

    colors = {"STS": "C0", "Explicit": "C1", "HyperTc": "C2"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    for name, d in data.items():
        p = params[name]
        for k, fi in enumerate(show_idx):
            t = d["times"][fi]
            row = d["press"][fi, ymid, :]
            ax.plot(x, row, color=colors[name], alpha=0.35 + 0.65 * k / (len(show_idx) - 1), lw=1.3,
                    label=name if fi == show_idx[-1] else None)
    for fi in show_idx:
        t = ref["times"][fi]
        ax.plot(x, analytic_pres(x, t, params["STS"]), "k--", lw=1.0, alpha=0.7,
                label="analytic" if fi == show_idx[0] else None)
    ax.set_xlabel("x")
    ax.set_ylabel("Pressure")
    ax.set_title("Profile evolution (mid-y row): sim (solid) vs analytic (dashed)")
    ax.legend(fontsize=8)

    # --- relative L2 error vs time, normalised by the perturbation amplitude ---
    ax = axes[1]
    summary = {}
    for name, d in data.items():
        p = params[name]
        errs = []
        for fi in range(len(d["times"])):
            t = d["times"][fi]
            row = d["press"][fi, ymid, :]
            exact = analytic_pres(x, t, p)
            rel_l2 = np.sqrt(np.mean((row - exact) ** 2)) / p["dT"]
            errs.append(rel_l2)
        errs = np.array(errs)
        summary[name] = errs
        ax.plot(d["times"], errs, marker="o", ms=3, label=name, color=colors[name])
    ax.set_xlabel("t")
    ax.set_ylabel("relative L2 error (normalised by dT)")
    ax.set_yscale("log")
    ax.set_title("Deviation from analytic solution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("verify_profiles_and_error.png", dpi=140)
    print("Saved verify_profiles_and_error.png")

    print("\nFinal-time relative L2 error (normalised by dT):")
    for name, errs in summary.items():
        print(f"  {name:10s} t={data[name]['times'][-1]:.4f}: {errs[-1]:.4e}")

    # --- cross-scheme agreement (pairwise max abs difference in Pres) ---
    print("\nPairwise max |Pres_a - Pres_b| across all frames/cells (should be small vs dT):")
    names = list(data.keys())
    n_common = min(len(d["times"]) for d in data.values())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = data[names[i]], data[names[j]]
            diff = np.abs(a["press"][:n_common] - b["press"][:n_common])
            print(f"  {names[i]:10s} vs {names[j]:10s}: max={diff.max():.4e}  (dT={params[names[i]]['dT']:.3g})")


if __name__ == "__main__":
    main()
