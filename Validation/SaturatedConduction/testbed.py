"""Saturated-conduction test bed (after Cherry et al. 2024, sec. 3.2): a coronal Gaussian diffusing along the field.

rho = 1e-12 kg m^-3, T = 1e4 + 9.9e5 exp(-x^2 / (1e5 m)^2), x in [-1e6, 1e6] m (periodic), Spitzer kappa0 = 1e-11,
field along x, hydro fluxes off (pure conduction), 2D y-invariant (and one 1D run). At this density the initial
Spitzer flux at the front exceeds the Cowie & McKee limit q_sat = 5 phi sqrt(p / rho) p by ~10-100x, so saturation
controls the early evolution. Scan: phi in {off, 1.1, 0.3, 0.1}; methods: STS, the old cell-centred HTC, hybrid2
(substeps at s_max = 10, limiter on) and hybrid2 without substeps.

Reference: an independent 1D solution of the saturated parabolic equation (reference()), with the face flux of
Mosscap's explicit/STS operator along the field: kappa_f = mean of the two cells, upwind pressure, face-mean density,
F = f F_spitzer, f = q_sat / (q_sat + |F_spitzer|). SSP-RK2 at 0.3 dx^2 / chi_max, on 8x the Mosscap resolution
(and 4x, to estimate its own error), block-averaged onto the Mosscap cells.

usage (from this directory):
    python testbed.py run [names...]     # Mosscap runs (all by default), outputs compacted to npz
    python testbed.py reference          # 1D references for every phi
    python testbed.py analyse            # table + figures
"""
import copy
import glob
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = "/home/cmo/Mosscap/build_cpu/mosscap"
N, NY, L = 250, 16, 2.0e6
T_END, DT_OUT = 3.2, 0.4
RHO, T_BG, DT0, SIGMA0, KAPPA0, GAMMA = 1.0e-12, 1.0e4, 9.9e5, 70710.67811865475, 1.0e-11, 1.6666667
PHIS = {"nosat": None, "phi1.1": 1.1, "phi0.3": 0.3, "phi0.1": 0.1}
METHODS = ("sts", "old_htc", "old_htc_cfl0.01", "h2", "h2_nosub")


def base_config(name, num_dim=2):
    c = {
        "simulation": {"num_dim": num_dim, "fluid_type": "mhd", "zero_hydro_flux": True},
        "grid": {"num_ghost": 3, "x": N, "x_dim": L, "x_start": -L / 2},
        "boundary": {"xs": "periodic", "xe": "periodic"},
        "eos": {"type": "ideal", "gamma": GAMMA, "ion_frac": 1.0, "mass_per_h": 1.0},
        "scheme": {"riemann_solver": "hll", "reconstruction": "muscl"},
        "timestep": {"scheme": "ssprk3", "max_cfl": 0.5, "max_time": T_END},
        "problem": {"name": "gaussian_diffusion", "axis": 0, "b0": 1.0e-8, "base_density": RHO,
                    "base_temperature": T_BG, "delta_temperature": DT0, "sigma0": SIGMA0},
        "output": {"name": name, "delta_t": DT_OUT, "single_file": False,
                   "variables": {"primitive": True, "conserved": True}},
        "system": {"mem_pool_gb": 2.0},
    }
    if num_dim > 1:
        c["grid"].update(y=NY, y_dim=L * NY / N, y_start=-0.5 * L * NY / N)
        c["boundary"].update(ys="periodic", ye="periodic")
    return c


def configs():
    out = {}
    for tag, phi in PHIS.items():
        for m in METHODS:
            name = f"{m}_{tag}"
            c = base_config(name)
            sat = phi is not None
            if m.startswith("old_htc"):
                # its heat flux travels in the flux vector, which zero_hydro_flux would remove: hypertconly instead
                c["simulation"] = {"num_dim": 2, "fluid_type": "hypertconly", "hypertc_kappa": KAPPA0,
                                   "hypertc_spitzer": True, "hypertc_limit": sat, "hypertc_saturation_phi": phi or 1.1}
                if m.endswith("cfl0.01"):  # the setting the earlier validation needed for a smooth profile
                    c["timestep"]["max_cfl"] = 0.01
            else:
                tc = {"enable": True, "anisotropic": True, "spitzer": True, "kappa0": KAPPA0, "saturate": sat,
                      "saturation_phi": phi or 1.1, "method": "sts" if m == "sts" else "hybrid2"}
                if m != "sts":
                    tc["hybrid2"] = {"s_max": 10.0 if m == "h2" else 0.0, "limit": True}
                c["sources"] = {"thermal_conduction": tc}
            out[name] = c
    # the 1D code path of hybrid2
    c = base_config("h2_1d_phi0.3", num_dim=1)
    c["sources"] = {"thermal_conduction": {"enable": True, "anisotropic": True, "spitzer": True, "kappa0": KAPPA0,
                                           "saturate": True, "saturation_phi": 0.3, "method": "hybrid2",
                                           "hybrid2": {"s_max": 10.0, "limit": True}}}
    out["h2_1d_phi0.3"] = c
    return out


def compact(name):
    import xarray as xr
    files = sorted(glob.glob(os.path.join(HERE, f"{name}_[0-9][0-9][0-9][0-9][0-9].nc")))
    snaps = {}
    for f in files:
        d = xr.open_dataset(f)
        Q = d["Q"].values[:, 0]
        W = d["W"].values[:, 0]
        ip, ir = d.attrs["ipre"], d.attrs["irho"]
        j = Q.shape[1] // 2 if Q.ndim == 3 else None
        # internal energy from Q (W in an output is from the last RK stage); v = 0 here
        mag = (Q[6] ** 2 + Q[7] ** 2 + Q[8] ** 2) / (2.0 * 4.0e-7 * np.pi)
        e = Q[4] - mag
        p = (GAMMA - 1.0) * e
        rho = W[ir]
        if j is not None:
            p, rho, e = p[j], rho[j], e[j]
        else:
            p, rho, e = p[0], rho[0], e[0]
        t = float(d["time"].values.item())
        snaps[f"p_{t:.3f}"] = p[3:-3]
        snaps[f"rho_{t:.3f}"] = rho[3:-3]
        d.close()
    np.savez_compressed(os.path.join(HERE, f"{name}.npz"), **snaps)
    for f in files:
        os.remove(f)


def run(names):
    cfgs = configs()
    for name in names or cfgs:
        yml = os.path.join(HERE, f"{name}.yaml")
        yaml.safe_dump(cfgs[name], open(yml, "w"), sort_keys=False)
        env = dict(os.environ, OMP_NUM_THREADS="12", OMP_PROC_BIND="spread", OMP_PLACES="threads")
        r = subprocess.run(["taskset", "-c", "0-11", BIN, "--config", yml], cwd=HERE, env=env,
                           stdout=open(os.path.join(HERE, f"{name}.log"), "w"), stderr=subprocess.STDOUT)
        compact(name)
        steps = [l for l in open(os.path.join(HERE, f"{name}.log")) if l.startswith("* t")]
        print(f"{name}: exit {r.returncode}, {steps[-1].strip() if steps else 'no output'}", flush=True)


# --- 1D reference ----------------------------------------------------------------------------------------------
def gas_constant():
    """p / (rho T) of the Mosscap ideal EOS used here, from a run's initial state (avoids re-deriving its constants)."""
    d = np.load(os.path.join(HERE, "sts_nosat.npz"))
    p, rho = d["p_0.000"], d["rho_0.000"]
    x = -L / 2 + (np.arange(N) + 0.5) * L / N
    T = T_BG + DT0 * np.exp(-0.5 * x ** 2 / SIGMA0 ** 2)
    return float(np.median(p / (rho * T)))


def reference_1d(phi, refine, R):
    n = N * refine
    dx = L / n
    x = -L / 2 + (np.arange(n) + 0.5) * dx
    C = RHO * R / (GAMMA - 1.0)  # rho c_v
    T = T_BG + DT0 * np.exp(-0.5 * x ** 2 / SIGMA0 ** 2)

    def rate(T):
        K = KAPPA0 * T ** 2.5
        Tm = np.roll(T, 1)
        kf = 0.5 * (K + np.roll(K, 1))
        F = -kf * (T - Tm) / dx  # flux through the lower face of each cell
        if phi is not None:
            p = RHO * R * T
            pm = np.roll(p, 1)
            pup = np.where(F > 0, pm, np.where(F < 0, p, 0.5 * (p + pm)))
            s = 5.0 * phi * np.sqrt(pup / RHO) * pup
            F = F * s / (s + np.abs(F))
        return (F - np.roll(F, -1)) / (dx * C)

    out, t = {0.0: T.copy()}, 0.0
    targets = [DT_OUT * (k + 1) for k in range(int(round(T_END / DT_OUT)))]
    for tt in targets:
        while t < tt * (1 - 1e-12):
            chi = KAPPA0 * T.max() ** 2.5 / C
            dt = min(0.3 * dx ** 2 / chi, tt - t)
            T1 = T + dt * rate(T)
            T = 0.5 * (T + T1 + dt * rate(T1))
            t += dt
        out[tt] = T.reshape(N, refine).mean(axis=1)  # block-average onto the Mosscap cells
    return out


def _ref_job(a):
    tag, refine, R = a
    res = reference_1d(PHIS[tag], refine, R)
    np.savez_compressed(os.path.join(HERE, f"ref_{tag}_x{refine}.npz"), **{f"T_{t:.3f}": v for t, v in res.items()})
    return tag, refine


def reference():
    R = gas_constant()
    jobs = [(tag, r, R) for tag in PHIS for r in (4, 8)]
    with Pool(len(jobs)) as pl:
        for tag, r in pl.imap_unordered(_ref_job, jobs):
            print(f"reference {tag} x{r} done", flush=True)


# --- analysis ---------------------------------------------------------------------------------------------------
def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = gas_constant()
    x = -L / 2 + (np.arange(N) + 0.5) * L / N
    times = [DT_OUT * (k + 1) for k in range(int(round(T_END / DT_OUT)))]
    T_of = lambda d, t: d[f"p_{t:.3f}"] / (d[f"rho_{t:.3f}"] * R)
    front = lambda T: float(np.max(np.abs(x[T > T_BG + 0.1 * (T.max() - T_BG)])))
    lines = []
    fig, ax = plt.subplots(2, len(PHIS), figsize=(5 * len(PHIS), 8), constrained_layout=True)
    for k, tag in enumerate(PHIS):
        ref = np.load(os.path.join(HERE, f"ref_{tag}_x8.npz"))
        ref4 = np.load(os.path.join(HERE, f"ref_{tag}_x4.npz"))
        sc = DT0
        refconv = max(np.mean(np.abs(ref4[f"T_{t:.3f}"] - ref[f"T_{t:.3f}"])) / sc for t in times)
        lines.append(f"\n== {tag}: reference x4 vs x8 L1/dT0 (worst t) {refconv:.1e}")
        lines.append(f"{'method':10s} " + " ".join(f"t={t:<6g}" for t in times[1::2]) + " | worst L1/dT0 | "
                     "peak T / ref (t = 0.8, 3.2) | front / ref (t = 0.8, 3.2) | energy drift")
        for t, ls in ((0.8, "--"), (3.2, "-")):
            ax[0, k].plot(x / 1e6, ref[f"T_{t:.3f}"] / 1e6, "k", ls=ls, lw=2, label=f"reference t={t:g}" if ls == "-" else None)
        for m, col in zip(METHODS + ("h2_1d",), ("C0", "C3", "C5", "C2", "C1", "C4")):
            name = f"{m}_{tag}"
            path = os.path.join(HERE, f"{name}.npz")
            if not os.path.exists(path):
                continue
            d = np.load(path)
            have = [t for t in times if f"p_{t:.3f}" in d.files]
            if len(have) < len(times):
                lines.append(f"{m:10s} FAILED (outputs to t = {have[-1] if have else 0:g})")
                continue
            errs = [np.mean(np.abs(T_of(d, t) - ref[f"T_{t:.3f}"])) / sc for t in times]
            pk = [T_of(d, t).max() / ref[f"T_{t:.3f}"].max() for t in (0.8, 3.2)]
            fr = [front(T_of(d, t)) / front(ref[f"T_{t:.3f}"]) for t in (0.8, 3.2)]
            E0 = d["p_0.000"].sum()
            drift = d[f"p_{times[-1]:.3f}"].sum() / E0 - 1
            lines.append(f"{m:10s} " + " ".join(f"{e:.1e}  " for e in errs[1::2]) + f" | {max(errs):.1e} | "
                         f"{pk[0]:.3f}, {pk[1]:.3f} | {fr[0]:.3f}, {fr[1]:.3f} | {drift:+.1e}")
            for t, ls in ((0.8, "--"), (3.2, "-")):
                ax[0, k].plot(x / 1e6, T_of(d, t) / 1e6, color=col, ls=ls, lw=1, label=m if ls == "-" else None)
            ax[1, k].semilogy(times, errs, "o-", color=col, label=m)
        ax[0, k].set(title=f"{tag}: T along the field (dashed t = 0.8 s, solid 3.2 s)", xlabel="x [Mm]",
                     ylabel="T [MK]", xlim=(-0.6, 0.6))
        ax[1, k].set(title=f"{tag}: L1(T) / dT0 against the reference", xlabel="t [s]")
        ax[0, k].legend(fontsize=7)
        ax[1, k].legend(fontsize=7)
    fig.savefig(os.path.join(HERE, "saturated_conduction.png"), dpi=90)
    print("\n".join(lines))
    print("wrote saturated_conduction.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"run": lambda: run(sys.argv[2:]), "reference": reference, "analyse": analyse}[cmd]()
