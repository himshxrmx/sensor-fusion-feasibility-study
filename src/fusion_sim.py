"""
2D multi-sensor tracking simulation for the Saarthi sensor-fusion feasibility study.

What it does
------------
Simulates camera, radar, LiDAR and ultrasonic measurements of one target in four Indian-road scenarios
and five operating conditions, tracks the target with an extended Kalman filter (constant-velocity model,
asynchronous sequential updates, chi-square gating, radar clutter) and compares five sensor configurations.
Every (scenario, condition, configuration) cell is a Monte Carlo of N runs, vectorised over runs.

Sensor models use typical datasheet-level numbers (bearing / range noise, detection probability,
maximum range) and simple weather and lighting effects. They are assumptions, listed in SENSORS below
and in the report, not measurements of a specific product.

Outputs: results/fusion_results.csv, results/experiments.json, figures/*.png
Run:     python src/fusion_sim.py            (about 1 minute on a laptop)
"""
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
SEED = 2026
N_RUNS = 200
DEG = np.pi / 180

# ------------------------------------------------------------------ sensor models (assumptions)
# rate Hz, base detection probability, range noise (a + b*r), bearing noise, max range by target type
SENSORS = {
    "camera": dict(rate=30, pd=0.95, sr_a=0.20, sr_b=0.030, sth=0.10 * DEG, rdot=False,
                   rmax=dict(car=100, ped=60, bike=80, truck=120)),
    "radar": dict(rate=20, pd=0.92, sr_a=0.15, sr_b=0.0, sth=1.0 * DEG, srd=0.10, rdot=True,
                  rmax=dict(car=160, ped=50, bike=90, truck=200)),
    "lidar": dict(rate=10, pd=0.97, sr_a=0.03, sr_b=0.0, sth=0.10 * DEG, rdot=False,
                  rmax=dict(car=120, ped=60, bike=90, truck=150)),
    "ultrasonic": dict(rate=20, pd=0.95, sr_a=0.02, sr_b=0.01, sth=None, rdot=False,
                       rmax=dict(car=5, ped=5, bike=5, truck=5)),
}
# condition effects: multipliers on detection probability, noise, max range; clutter probability per scan
CONDITIONS = {
    "clear day":      dict(camera=(1.00, 1.0, 1.0, 0.01), radar=(1.00, 1.0, 1.0, 0.04), lidar=(1.00, 1.0, 1.00, 0.00), ultrasonic=(1.0, 1.0, 1.0, 0.0)),
    "night":          dict(camera=(0.84, 1.6, 0.6, 0.02), radar=(1.00, 1.0, 1.0, 0.04), lidar=(1.00, 1.0, 1.00, 0.00), ultrasonic=(1.0, 1.0, 1.0, 0.0)),
    "heavy rain":     dict(camera=(0.89, 1.3, 0.8, 0.03), radar=(0.95, 1.1, 0.9, 0.12), lidar=(0.80, 2.0, 0.70, 0.05), ultrasonic=(0.95, 1.2, 1.0, 0.0)),
    "dense fog":      dict(camera=(0.63, 2.0, 0.4, 0.03), radar=(1.00, 1.0, 1.0, 0.06), lidar=(0.50, 3.0, 0.35, 0.08), ultrasonic=(1.0, 1.0, 1.0, 0.0)),
    "camera blinded": dict(camera=(0.05, 2.0, 0.5, 0.05), radar=(1.00, 1.0, 1.0, 0.04), lidar=(1.00, 1.0, 1.00, 0.00), ultrasonic=(1.0, 1.0, 1.0, 0.0)),
}
CONFIGS = {
    "Camera only": ["camera"],
    "Radar only": ["radar"],
    "LiDAR only": ["lidar"],
    "Camera + radar": ["camera", "radar", "ultrasonic"],
    "Camera + radar + LiDAR": ["camera", "radar", "lidar", "ultrasonic"],
}
PED_RADAR_PD = 0.75  # pedestrians have a small radar cross-section


# ------------------------------------------------------------------ scenarios (ego frame: x ahead, y left)
def sc_cut_in(t):        # car 25 m ahead in the next lane, 3 m/s slower, cuts in over 3 s
    vy = np.where((t > 1.0) & (t < 4.0), -1.17, 0.0)
    y = 3.5 - 1.17 * np.clip(t - 1.0, 0, 3.0)
    return np.stack([25 - 3.0 * t, y, -3.0 * np.ones_like(t), vy], -1)


def sc_pedestrian(t):    # pedestrian crossing 35 m ahead while ego drives at 40 km/h
    return np.stack([35 - 11.1 * t, -5 + 1.4 * t, -11.1 * np.ones_like(t), 1.4 * np.ones_like(t)], -1)


def sc_bike(t):          # motorbike weaving 40 m ahead, 2 m/s slower
    return np.stack([40 - 2.0 * t, 2 * np.sin(0.8 * t), -2.0 * np.ones_like(t), 1.6 * np.cos(0.8 * t)], -1)


def sc_truck(t):         # stalled truck in lane, ego at 80 km/h, starts 150 m away
    return np.stack([150 - 22.2 * t, 0.5 * np.ones_like(t), -22.2 * np.ones_like(t), np.zeros_like(t)], -1)


SCENARIOS = {
    "S1 car cut-in": dict(f=sc_cut_in, T=8.0, kind="car", q=1.5),
    "S2 pedestrian crossing": dict(f=sc_pedestrian, T=2.8, kind="ped", q=1.0),
    "S3 motorbike weaving": dict(f=sc_bike, T=10.0, kind="bike", q=2.0),
    "S4 stalled truck at 80 km/h": dict(f=sc_truck, T=5.5, kind="truck", q=0.5),
}
GATE = {1: 8.8, 2: 11.6, 3: 14.2}   # chi-square 99.7% gates
DROP_AFTER_S = 1.5
GOOD_ERR_M = 1.0


def h_polar(X, with_rdot, range_only=False):
    px, py, vx, vy = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    r = np.sqrt(px ** 2 + py ** 2) + 1e-9
    n = len(X)
    if range_only:
        H = np.zeros((n, 1, 4)); H[:, 0, 0], H[:, 0, 1] = px / r, py / r
        return r[:, None], H
    th = np.arctan2(py, px)
    m = 3 if with_rdot else 2
    H = np.zeros((n, m, 4))
    H[:, 0, 0], H[:, 0, 1] = px / r, py / r
    H[:, 1, 0], H[:, 1, 1] = -py / r ** 2, px / r ** 2
    z = [r, th]
    if with_rdot:
        dot = px * vx + py * vy
        z.append(dot / r)
        H[:, 2, 0] = (vx * r ** 2 - px * dot) / r ** 3
        H[:, 2, 1] = (vy * r ** 2 - py * dot) / r ** 3
        H[:, 2, 2], H[:, 2, 3] = px / r, py / r
    return np.stack(z, -1), H


def run_cell(scn, cond, sensors, n=N_RUNS, seed=SEED, cam_delay=0.0, radar_bias=0.0, bias_correction=False):
    rng = np.random.default_rng(seed)
    sc = SCENARIOS[scn]
    kind, T, q = sc["kind"], sc["T"], sc["q"]
    events = [(round(k / 10, 6), "eval") for k in range(int(T * 10) + 1)]
    for s in sensors:
        rate = SENSORS[s]["rate"]
        events += [(round(k / rate + 0.001 * (1 + list(SENSORS).index(s)), 6), s) for k in range(int(T * rate))]
    events.sort()
    X = np.zeros((n, 4)); P = np.tile(np.eye(4) * 100, (n, 1, 1))
    init = np.zeros(n, bool); last_upd = np.full(n, -9.0); n_upd = np.zeros(n, int)
    conf_dist = np.full(n, np.nan); drops = np.zeros(n, int)
    bias_est = np.zeros(n); bias_samples = [[] for _ in range(n)] if bias_correction else None
    pos_err, vel_err, nees, good, ever = [], [], [], [], []
    t_last = 0.0
    for t, typ in events:
        dt = t - t_last
        if dt > 0:
            F = np.eye(4); F[0, 2] = F[1, 3] = dt
            G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
            Q = G @ G.T * q ** 2
            X = X @ F.T
            P = F @ P @ F.T + Q
            t_last = t
        truth = sc["f"](np.array([t]))[0]
        stale = init & (t - last_upd > DROP_AFTER_S)
        drops += stale; init &= ~stale
        if typ == "eval":
            e = X - truth
            pe = np.hypot(e[:, 0], e[:, 1])
            conf = init & (n_upd >= 3)
            pos_err.append(np.where(conf, pe, np.nan)); vel_err.append(np.where(conf, np.hypot(e[:, 2], e[:, 3]), np.nan))
            Pinv = np.linalg.inv(P)
            nees.append(np.where(conf, np.einsum("ni,nij,nj->n", e, Pinv, e), np.nan))
            good.append(conf & (pe < GOOD_ERR_M)); ever.append(conf)
            continue
        S_ = SENSORS[typ]
        pd_mul, noise_mul, range_mul, clutter = CONDITIONS[cond][typ]
        tz = sc["f"](np.array([t - (cam_delay if typ == "camera" else 0.0)]))[0]
        r_true = np.hypot(tz[0], tz[1]); th_true = np.arctan2(tz[1], tz[0])
        rmax = S_["rmax"][kind] * range_mul
        if typ == "ultrasonic" and abs(th_true) > 35 * DEG:
            continue
        base_pd = PED_RADAR_PD if (typ == "radar" and kind == "ped") else S_["pd"]
        fade = np.clip((rmax - r_true) / (0.3 * rmax), 0, 1)       # detection fades over the last 30% of range
        pd = base_pd * pd_mul * fade if r_true > (0.3 if typ == "ultrasonic" else 0.5) else 0.0
        det = rng.random(n) < pd
        sr = (S_["sr_a"] + S_["sr_b"] * r_true) * noise_mul
        range_only = typ == "ultrasonic"
        if range_only:
            z = (r_true + rng.normal(0, sr, n))[:, None]; Rm = np.array([[sr ** 2]])
        else:
            sth = S_["sth"] * noise_mul
            cols = [r_true + rng.normal(0, sr, n), th_true + rng.normal(0, sth, n) + (radar_bias if typ == "radar" else 0.0)]
            diag = [sr ** 2, sth ** 2]
            if S_["rdot"]:
                cols.append((tz[0] * tz[2] + tz[1] * tz[3]) / r_true + rng.normal(0, S_["srd"] * noise_mul, n)); diag.append((S_["srd"] * noise_mul) ** 2)
            z = np.stack(cols, -1); Rm = np.diag(diag)
            if typ == "radar" and bias_correction:
                z[:, 1] -= bias_est
        m = z.shape[1]
        # ---- update initialised tracks
        if init.any():
            zp, H = h_polar(X, S_["rdot"], range_only)
            S = H @ P @ H.transpose(0, 2, 1) + Rm
            Sinv = np.linalg.inv(S)
            y = z - zp
            if not range_only:
                y[:, 1] = (y[:, 1] + np.pi) % (2 * np.pi) - np.pi
            d2 = np.einsum("ni,nij,nj->n", y, Sinv, y)
            d2 = np.where(det, d2, np.inf)
            # clutter: a ghost / multipath return near the real object (up to 4 sigma away in each measured
            # dimension). It is placed around the TRUE target, so a track that has drifted away is not fed by it.
            has_cl = rng.random(n) < clutter
            rdot_true = (tz[0] * tz[2] + tz[1] * tz[3]) / r_true
            z_true = np.array([r_true] if range_only else [r_true, th_true] + ([rdot_true] if S_["rdot"] else []))
            zc = z_true + rng.uniform(-4, 4, (n, m)) * np.sqrt(np.diag(Rm)) + (np.array([1.0] + [0.0] * (m - 1)) * rng.uniform(-1.5, 1.5, (n, 1)))
            yc = zc - zp
            if not range_only:
                yc[:, 1] = (yc[:, 1] + np.pi) % (2 * np.pi) - np.pi
            d2c = np.where(has_cl, np.einsum("ni,nij,nj->n", yc, Sinv, yc), np.inf)
            use_cl = d2c < d2
            y = np.where(use_cl[:, None], yc, y); d2 = np.minimum(d2, d2c)
            ok = init & (d2 < GATE[m])
            if bias_correction and typ == "radar":
                for i in np.where(ok & ~use_cl)[0]:
                    if len(bias_samples[i]) < 20:          # learn the offset from the first 20 radar innovations, then freeze it
                        bias_samples[i].append(y[i, 1])
                        if len(bias_samples[i]) == 20:
                            bias_est[i] = np.mean(bias_samples[i])
            K = P @ H.transpose(0, 2, 1) @ Sinv
            Xn = X + np.einsum("nij,nj->ni", K, y)
            Pn = (np.eye(4) - K @ H) @ P
            X = np.where(ok[:, None], Xn, X); P = np.where(ok[:, None, None], Pn, P)
            last_upd = np.where(ok, t, last_upd); n_upd += ok
            newly = ok & (n_upd == 3) & np.isnan(conf_dist)
            conf_dist = np.where(newly, np.hypot(truth[0], truth[1]), conf_dist)
        # ---- start new tracks from a detection (not from range-only ultrasonic)
        start = (~init) & det & (not range_only)
        if start.any():
            r, th = z[start, 0], z[start, 1]
            X[start, 0], X[start, 1] = r * np.cos(th), r * np.sin(th)
            J = np.zeros((start.sum(), 2, 2))
            J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1] = np.cos(th), -r * np.sin(th), np.sin(th), r * np.cos(th)
            Pc = J @ np.diag([Rm[0, 0], Rm[1, 1]]) @ J.transpose(0, 2, 1)
            Pnew = np.tile(np.eye(4) * 400.0, (start.sum(), 1, 1))
            Pnew[:, :2, :2] = Pc
            if S_["rdot"]:
                X[start, 2], X[start, 3] = z[start, 2] * np.cos(th), z[start, 2] * np.sin(th)
                Pnew[:, 2, 2] = Pnew[:, 3, 3] = 25.0
            else:
                X[start, 2:] = 0.0
            P[start] = Pnew
            init |= start; last_upd = np.where(start, t, last_upd); n_upd = np.where(start, 1, n_upd)
    pe, ve, ne = np.array(pos_err), np.array(vel_err), np.array(nees)
    g, ev = np.array(good), np.array(ever)
    with np.errstate(all="ignore"):
        return dict(pos_rmse=float(np.sqrt(np.nanmean(pe ** 2))) if np.isfinite(pe).any() else np.nan,
                    vel_rmse=float(np.sqrt(np.nanmean(ve ** 2))) if np.isfinite(ve).any() else np.nan,
                    pos_err_median=float(np.nanmedian(pe)) if np.isfinite(pe).any() else np.nan,
                    pos_err_p95=float(np.nanpercentile(pe, 95)) if np.isfinite(pe).any() else np.nan,
                    good_track_pct=float(100 * g.mean()), tracked_pct=float(100 * ev.mean()),
                    nees_median=float(np.nanmedian(ne)) if np.isfinite(ne).any() else np.nan,
                    first_confirmed_range_m=float(np.nanmedian(conf_dist)) if np.isfinite(conf_dist).any() else np.nan,
                    never_confirmed_pct=float(100 * np.isnan(conf_dist).mean()),
                    track_drops_per_run=float(drops.mean()),
                    _pos_err=pe)


def main():
    t0 = time.time()
    rows = []
    for scn in SCENARIOS:
        for cond in CONDITIONS:
            for cfg, sens in CONFIGS.items():
                r = run_cell(scn, cond, sens)
                r.pop("_pos_err")
                rows.append(dict(scenario=scn, condition=cond, config=cfg, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    df = pd.DataFrame(rows)
    df.to_csv("results/fusion_results.csv", index=False)
    exp = {"runs_per_cell": N_RUNS, "cells": len(df), "total_runs": N_RUNS * len(df), "seed": SEED}

    # summary tables
    piv_rmse = df.pivot_table(index="config", columns="condition", values="pos_rmse", aggfunc="mean").reindex(index=list(CONFIGS), columns=list(CONDITIONS))
    piv_good = df.pivot_table(index="config", columns="condition", values="good_track_pct", aggfunc="mean").reindex(index=list(CONFIGS), columns=list(CONDITIONS))
    exp["mean_pos_rmse_m"] = piv_rmse.round(3).to_dict()
    exp["mean_good_track_pct"] = piv_good.round(1).to_dict()
    clear_good = piv_good["clear day"]
    exp["robustness_ratio_good_track"] = (piv_good.div(clear_good, axis=0)).round(3).to_dict()
    truck = df[df.scenario.str.startswith("S4")].pivot_table(index="config", columns="condition", values="first_confirmed_range_m").reindex(index=list(CONFIGS), columns=list(CONDITIONS))
    exp["truck_first_confirmed_range_m"] = truck.round(1).to_dict()
    v = 22.2
    exp["stopping_distance_80kmh_m"] = {"dry (0.7 g), 1.0 s reaction": round(v * 1.0 + v ** 2 / (2 * 0.7 * 9.81), 1),
                                        "wet (0.45 g), 1.0 s reaction": round(v * 1.0 + v ** 2 / (2 * 0.45 * 9.81), 1),
                                        "AEB only (0.8 g), 0.3 s system delay": round(v * 0.3 + v ** 2 / (2 * 0.8 * 9.81), 1)}

    # ---- time synchronisation experiment: camera frames stamped late, in camera + radar fusion
    sync = []
    for scn in ["S4 stalled truck at 80 km/h", "S1 car cut-in"]:
        for d in [0, 0.02, 0.05, 0.10, 0.15]:
            r1 = run_cell(scn, "clear day", ["camera"], cam_delay=d)
            r2 = run_cell(scn, "clear day", ["camera", "radar", "ultrasonic"], cam_delay=d)
            sync.append(dict(scenario=scn, delay_ms=int(d * 1000), camera_only_rmse=round(r1["pos_rmse"], 3), fused_rmse=round(r2["pos_rmse"], 3)))
    exp["time_sync"] = sync

    # ---- calibration experiment: radar azimuth misalignment, with and without online correction
    calib = []
    for b in [0, 0.5, 1.0, 2.0, 3.0]:
        r_raw = run_cell("S1 car cut-in", "clear day", ["radar"], radar_bias=b * DEG)
        r_fused = run_cell("S1 car cut-in", "night", ["camera", "radar", "ultrasonic"], radar_bias=b * DEG)
        r_corr = run_cell("S1 car cut-in", "night", ["camera", "radar", "ultrasonic"], radar_bias=b * DEG, bias_correction=True)
        calib.append(dict(bias_deg=b, lateral_offset_at_50m_m=round(50 * np.tan(b * DEG), 2), radar_only_rmse=round(r_raw["pos_rmse"], 3),
                          fused_night_rmse=round(r_fused["pos_rmse"], 3), fused_night_rmse_with_online_correction=round(r_corr["pos_rmse"], 3)))
    exp["calibration"] = calib

    # ---- compute cost of the tracker
    Xs = np.array([[30.0, 1.0, -5.0, 0.0]]); Ps = np.eye(4)[None] * 2
    reps = 3000
    t1 = time.perf_counter()
    for _ in range(reps):
        zp, H = h_polar(Xs, True)
        S = H @ Ps @ H.transpose(0, 2, 1) + np.eye(3) * 0.01
        K = Ps @ H.transpose(0, 2, 1) @ np.linalg.inv(S)
        Xs2 = Xs + np.einsum("nij,nj->ni", K, np.zeros((1, 3)))
        Ps2 = (np.eye(4) - K @ H) @ Ps
    one = (time.perf_counter() - t1) / reps * 1e6
    nb = 100
    Xb = np.tile(Xs, (nb, 1)); Pb = np.tile(Ps, (nb, 1, 1))
    t1 = time.perf_counter()
    for _ in range(300):
        zp, H = h_polar(Xb, True)
        S = H @ Pb @ H.transpose(0, 2, 1) + np.eye(3) * 0.01
        K = Pb @ H.transpose(0, 2, 1) @ np.linalg.inv(S)
        Pb2 = (np.eye(4) - K @ H) @ Pb
    batch = (time.perf_counter() - t1) / 300 * 1e6
    exp["tracker_compute_us"] = {"single_update_python_numpy": round(one, 1), "100_tracks_one_radar_scan_vectorised": round(batch, 1)}
    exp["runtime_s"] = round(time.time() - t0, 1)
    json.dump(exp, open("results/experiments.json", "w"), indent=2)

    # ------------------------------------------------------------------ figures
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.16
    xs = np.arange(len(CONDITIONS))
    for i, cfg in enumerate(CONFIGS):
        vals = piv_good.loc[cfg].values
        ax.bar(xs + (i - 2) * w, vals, w, label=cfg, color=cols[i])
    ax.set_xticks(xs); ax.set_xticklabels(list(CONDITIONS))
    ax.set_ylabel("Time with a good track (%)\n(error < 1 m, mean of 4 scenarios)"); ax.set_ylim(0, 105)
    ax.set_title(f"Share of time the target is tracked within 1 m ({N_RUNS} Monte Carlo runs per bar per scenario)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1)); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig6_good_track_by_condition.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    data = piv_rmse.values.astype(float)
    im = ax.imshow(np.minimum(data, 3.0), cmap="RdYlGn_r", vmin=0, vmax=3.0, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "-" if np.isnan(data[i, j]) else f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(CONDITIONS))); ax.set_xticklabels(list(CONDITIONS)); ax.set_yticks(range(len(CONFIGS))); ax.set_yticklabels(list(CONFIGS))
    ax.set_title("Position RMSE of confirmed tracks (m), mean of 4 scenarios")
    fig.colorbar(im, ax=ax, label="m (capped at 3)"); fig.tight_layout(); fig.savefig("figures/fig7_rmse_heatmap.png", dpi=160); plt.close(fig)

    # example trajectory: motorbike weaving at night, camera only vs fused (single run)
    sc = SCENARIOS["S3 motorbike weaving"]
    tt = np.arange(0, sc["T"], 0.05); tr = sc["f"](tt)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(tr[:, 0], tr[:, 1], "k-", lw=2, label="true path")
    for cfg, c in [("Camera only", cols[0]), ("Camera + radar + LiDAR", cols[4])]:
        est = np.array(_run_single("S3 motorbike weaving", "night", CONFIGS[cfg], 11, []) or [[np.nan, np.nan]])
        ax.plot(est[:, 0], est[:, 1], ".", ms=3, color=c, label=f"{cfg} estimate (night)")
    ax.invert_xaxis(); ax.set_xlabel("distance ahead (m)"); ax.set_ylabel("lateral (m)")
    ax.set_title("S3 motorbike weaving at night: one simulated run"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig8_bike_trajectory.png", dpi=160); plt.close(fig)

    # truck detection distance vs stopping distance
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, cfg in enumerate(CONFIGS):
        ax.bar(xs + (i - 2) * w, truck.loc[cfg].fillna(0).values, w, label=cfg, color=cols[i])
    sd = exp["stopping_distance_80kmh_m"]
    ax.axhline(sd["dry (0.7 g), 1.0 s reaction"], color="k", ls="--", lw=1); ax.text(4.45, sd["dry (0.7 g), 1.0 s reaction"] + 3, "stopping distance, dry", ha="right", fontsize=8)
    ax.axhline(sd["wet (0.45 g), 1.0 s reaction"], color="k", ls=":", lw=1); ax.text(4.45, sd["wet (0.45 g), 1.0 s reaction"] + 3, "stopping distance, wet", ha="right", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(list(CONDITIONS)); ax.set_ylabel("range when track confirmed (m)")
    ax.set_title("S4 stalled truck at 80 km/h: distance at which a confirmed track exists (median)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1)); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig9_truck_detection_range.png", dpi=160); plt.close(fig)

    # worked example: covariance ellipses at 30 m
    r0 = 30.0
    cam = np.diag([(0.2 + 0.03 * r0) ** 2, (r0 * 0.1 * DEG) ** 2]); rad = np.diag([0.15 ** 2, (r0 * 1.0 * DEG) ** 2])
    fused = np.linalg.inv(np.linalg.inv(cam) + np.linalg.inv(rad))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for C, c, lab in [(cam, cols[0], "camera"), (rad, cols[1], "radar"), (fused, cols[4], "fused")]:
        sx, sy = 2 * np.sqrt(C[0, 0]), 2 * np.sqrt(C[1, 1])
        ax.add_patch(Ellipse((r0, 0), 2 * sx, 2 * sy, fill=False, color=c, lw=2, label=f"{lab}: ±{sx / 2:.2f} m range, ±{sy / 2:.2f} m lateral"))
    ax.set_xlim(r0 - 3.5, r0 + 3.5); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("range (m)"); ax.set_ylabel("lateral (m)"); ax.set_title("Worked example: 2-sigma uncertainty of one car at 30 m")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout(); fig.savefig("figures/fig10_worked_example_ellipses.png", dpi=160); plt.close(fig)
    exp["worked_example_30m"] = {"camera_sigma_range_m": round(float(np.sqrt(cam[0, 0])), 3), "camera_sigma_lateral_m": round(float(np.sqrt(cam[1, 1])), 3),
                                 "radar_sigma_range_m": round(float(np.sqrt(rad[0, 0])), 3), "radar_sigma_lateral_m": round(float(np.sqrt(rad[1, 1])), 3),
                                 "fused_sigma_range_m": round(float(np.sqrt(fused[0, 0])), 3), "fused_sigma_lateral_m": round(float(np.sqrt(fused[1, 1])), 3)}

    # sync + calibration figure
    sdf, cdf = pd.DataFrame(sync), pd.DataFrame(calib)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    for scn, mk in [("S4 stalled truck at 80 km/h", "o"), ("S1 car cut-in", "s")]:
        d = sdf[sdf.scenario == scn]
        a1.plot(d.delay_ms, d.camera_only_rmse, mk + "--", label=f"camera only, {scn[:2]}")
        a1.plot(d.delay_ms, d.fused_rmse, mk + "-", label=f"camera + radar, {scn[:2]}")
    a1.set_xlabel("camera timestamp error (ms)"); a1.set_ylabel("position RMSE (m)"); a1.set_title("Time synchronisation error"); a1.legend(fontsize=7); a1.grid(alpha=0.3)
    a2.plot(cdf.bias_deg, cdf.radar_only_rmse, "o--", label="radar only, clear")
    a2.plot(cdf.bias_deg, cdf.fused_night_rmse, "s-", label="camera + radar, night")
    a2.plot(cdf.bias_deg, cdf.fused_night_rmse_with_online_correction, "^-", label="camera + radar, night, online correction")
    a2.set_xlabel("radar azimuth misalignment (deg)"); a2.set_ylabel("position RMSE (m)"); a2.set_title("Calibration drift, S1 cut-in"); a2.legend(fontsize=7); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig11_sync_and_calibration.png", dpi=160); plt.close(fig)
    json.dump(exp, open("results/experiments.json", "w"), indent=2)
    print(piv_good.round(1).to_string()); print(piv_rmse.round(2).to_string()); print(truck.round(0).to_string())
    print(json.dumps({k: exp[k] for k in ["time_sync", "calibration", "tracker_compute_us", "worked_example_30m", "stopping_distance_80kmh_m", "runtime_s"]}, indent=1))


def _run_single(scn, cond, sensors, seed, out):
    rng = np.random.default_rng(seed)
    sc = SCENARIOS[scn]; kind, T, q = sc["kind"], sc["T"], sc["q"]
    events = [(round(k / 10, 6), "eval") for k in range(int(T * 10) + 1)]
    for s in sensors:
        rate = SENSORS[s]["rate"]
        events += [(round(k / rate + 0.001 * (1 + list(SENSORS).index(s)), 6), s) for k in range(int(T * rate))]
    events.sort()
    X = np.zeros((1, 4)); P = np.eye(4)[None] * 100; init = False; t_last = 0.0; nu = 0
    for t, typ in events:
        dt = t - t_last
        if dt > 0:
            F = np.eye(4); F[0, 2] = F[1, 3] = dt
            G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
            X = X @ F.T; P = F @ P @ F.T + G @ G.T * q ** 2; t_last = t
        if typ == "eval":
            if init and nu >= 3:
                out.append(X[0, :2].copy())
            continue
        S_ = SENSORS[typ]; pd_mul, noise_mul, range_mul, _ = CONDITIONS[cond][typ]
        tz = sc["f"](np.array([t]))[0]; r_true = np.hypot(tz[0], tz[1]); th_true = np.arctan2(tz[1], tz[0])
        if typ == "ultrasonic":
            continue
        rmax = S_["rmax"][kind] * range_mul
        pd = S_["pd"] * pd_mul * np.clip((rmax - r_true) / (0.3 * rmax), 0, 1)
        if rng.random() > pd:
            continue
        sr = (S_["sr_a"] + S_["sr_b"] * r_true) * noise_mul; sth = S_["sth"] * noise_mul
        z = [r_true + rng.normal(0, sr), th_true + rng.normal(0, sth)]; Rd = [sr ** 2, sth ** 2]
        if S_["rdot"]:
            z.append((tz[0] * tz[2] + tz[1] * tz[3]) / r_true + rng.normal(0, S_["srd"])); Rd.append(S_["srd"] ** 2)
        z = np.array(z)[None]; Rm = np.diag(Rd)
        if not init:
            X[0, 0], X[0, 1] = z[0, 0] * np.cos(z[0, 1]), z[0, 0] * np.sin(z[0, 1]); X[0, 2:] = 0
            P = np.eye(4)[None] * 400; P[0, 0, 0] = P[0, 1, 1] = max(sr ** 2, (r_true * sth) ** 2); init = True; nu = 1
            continue
        zp, H = h_polar(X, S_["rdot"])
        S = H @ P @ H.transpose(0, 2, 1) + Rm; y = z - zp; y[:, 1] = (y[:, 1] + np.pi) % (2 * np.pi) - np.pi
        K = P @ H.transpose(0, 2, 1) @ np.linalg.inv(S)
        X = X + np.einsum("nij,nj->ni", K, y); P = (np.eye(4) - K @ H) @ P; nu += 1
    return out


if __name__ == "__main__":
    main()
