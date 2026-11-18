"""
Builds the Week 2 feasibility study (.docx) from the simulation results and the reusable text blocks.
Run from the week folder:  python src/fusion_sim.py  then  python src/build_report.py
"""
import json
import sys

import pandas as pd

sys.path.insert(0, "src")
from doclib import Report, section  # noqa: E402

OUT = "Week2_Sensor_Fusion_Feasibility_Study.docx"
GH = "https://github.com/himshxrmx/sensor-fusion-feasibility-study"
B = json.load(open("content/base_text.json", encoding="utf-8"))
E = json.load(open("results/experiments.json"))
df = pd.read_csv("results/fusion_results.csv")
IMG = ["images/fig1_fusion_levels.png", "images/fig2_architecture.png", "images/fig3_simulation_pipeline.png",
       "images/fig4_processing_flowchart.png", "images/fig5_sensor_coverage.png"]
CONDS = ["clear day", "night", "heavy rain", "dense fog", "camera blinded"]
CFGS = ["Camera only", "Radar only", "LiDAR only", "Camera + radar", "Camera + radar + LiDAR"]


def fix(blocks, repl):
    def f(x):
        if isinstance(x, str):
            for a, b in repl.items():
                x = x.replace(a, b)
            return x
        if isinstance(x, list):
            return [f(i) for i in x]
        return x
    return [f(b) for b in blocks]


B = fix(B, {"Figure 5 in Appendix A shows": "The coverage diagram in Appendix A shows",
            "Figure 2 shows the proposed architecture": "Figure 2 shows the proposed architecture"})
good = E["mean_good_track_pct"]; rmse = E["mean_pos_rmse_m"]; truck = E["truck_first_confirmed_range_m"]
rr = E["robustness_ratio_good_track"]; sd = E["stopping_distance_80kmh_m"]; we = E["worked_example_30m"]
g = lambda cfg, cond: good[cond][cfg]  # noqa: E731
full, cr, cam = "Camera + radar + LiDAR", "Camera + radar", "Camera only"
min_rr_full = min(rr[c][full] for c in CONDS)

R = Report()
R.title_page("Feasibility Study: AI-Based Sensor Fusion for Autonomous Driving on Indian Roads",
             "Camera, radar, LiDAR and ultrasonic fusion for the Saarthi platform, with a Monte Carlo tracking experiment",
             ["Week 2 · Automotive AI Innovation Internship (YuvaIntern)", "Author: Him Sharma, B.Tech CSE, ABES Engineering College",
              "November 2026", f"Simulation code, results and figures: {GH}"])

# ------------------------------------------------------------------ 1
R.h1("1. Executive Summary")
R.p("An autonomous vehicle must build an accurate, real-time picture of its surroundings in every condition it is allowed to drive in, "
    "and no single sensor can do this alone. This study asks whether an AI-based fusion of cameras, radar, LiDAR and ultrasonic sensors "
    "is feasible for the Saarthi platform proposed in the Week 1 roadmap, first for Level 2+ driver assistance and later for autonomy on Indian roads.")
R.p(f"Besides the literature review, I built and ran a Monte Carlo tracking simulation ({E['total_runs']:,} runs) that compares five sensor "
    "configurations across four Indian-road scenarios (a car cutting in, a pedestrian crossing, a weaving motorbike and a stalled truck on a highway) "
    "and five conditions (clear day, night, heavy rain, dense fog and a blinded camera). An extended Kalman filter fuses the measurements as they arrive. "
    "The main results:")
R.bullets([
    ("Fusion is the only configuration that stays good everywhere:", f"camera + radar + LiDAR keeps a target within 1 m for {g(full, 'clear day'):.1f}% of the time "
     f"on a clear day and never less than {min(g(full, c) for c in CONDS):.1f}% in any condition. A camera alone falls to {g(cam, 'dense fog'):.1f}% in fog and {g(cam, 'camera blinded'):.1f}% when blinded."),
    ("Detection range decides safety at speed:", f"for a stalled truck at 80 km/h the stopping distance is {sd['dry (0.7 g), 1.0 s reaction']} m on a dry road and "
     f"{sd['wet (0.45 g), 1.0 s reaction']} m on a wet one. A camera alone confirms the truck at {truck['dense fog'][cam]:.0f} m in fog, too late to stop; "
     f"any configuration with radar confirms it at about {truck['dense fog'][cr]:.0f} m."),
    ("Worked example:", f"at 30 m a camera knows lateral position to ±{we['camera_sigma_lateral_m']} m but range only to ±{we['camera_sigma_range_m']} m; a radar is the "
     f"opposite (±{we['radar_sigma_range_m']} m range, ±{we['radar_sigma_lateral_m']} m lateral). Fused, the estimate is ±{we['fused_sigma_range_m']} m by ±{we['fused_sigma_lateral_m']} m: the best of both."),
    ("Calibration and timing matter:", "a 1° radar misalignment shifts a target 0.87 m sideways at 50 m. A 100 ms camera timestamp error doubles the camera-only error on the "
     "truck scenario, while camera + radar fusion is almost unaffected."),
    ("Feasibility scorecard:", "a weighted scorecard of nine criteria gives Level 2+ and geofenced low-speed autonomy a GO (3.85 out of 5) and open-road driverless "
     "operation in India a NO-GO for now (2.55, with data and regulation scored 1)."),
])
R.p("The recommendation is a staged plan: build Tier A (camera + radar + ultrasonic, about ₹1.06 lakh of sensors and compute per vehicle) for Level 2+, add a "
    "forward LiDAR for highway functions, and treat full autonomy on open Indian roads as a long-term goal that depends on Indian multi-sensor data and regulation.")

# ------------------------------------------------------------------ 2-3
R.emit(section(B, "2. Introduction"), drop={"2.4 Approach of This Study"})
R.h2("2.4 Approach of This Study")
R.p("The study combines a structured literature review with original quantitative work. The review covers sensor physics, levels of fusion, classical "
    "and deep-learning fusion methods, and the main integration challenges. The quantitative work has three parts: back-of-the-envelope engineering numbers "
    "for each challenge (Section 4.11), a worked fusion example (Section 6.5) and a Monte Carlo tracking simulation written in Python (Section 8). "
    "The simulation code and every number in this report are published in the GitHub repository so the results can be reproduced.")
R.emit(section(B, "3. Technical Background on Sensor Fusion"), images=IMG)

# ------------------------------------------------------------------ 4
R.emit(section(B, "4. Challenges of Integrating Multiple Sensor Types"))
R.h2("4.11 Putting Numbers on the Challenges")
R.p("Several challenges above are easier to judge with numbers. The values below use typical sensor specifications and simple geometry; the simulation "
    "in Section 8 measures their effect on a working tracker.")
R.table(["Challenge", "Worked number", "Why it matters"], [
    ["Calibration drift", "1° radar yaw error = 0.87 m lateral offset at 50 m and 1.75 m at 100 m", "1.75 m is half a lane: a car can be placed in the wrong lane"],
    ["Timestamp error", "50 ms at a closing speed of 22.2 m/s (80 km/h) = 1.11 m; 100 ms = 2.22 m", "Measured: camera-only error on the truck rises from 1.41 m to 2.86 m at 100 ms"],
    ["Different sampling rates", "Camera 30 Hz, radar 20 Hz, LiDAR 10 Hz: a target closing at 22.2 m/s moves 2.2 m between LiDAR sweeps", "Measurements must be motion-compensated to a common time"],
    ["Monocular range error", "About 3% of range + 0.2 m: ±1.1 m at 30 m, ±3.8 m at 120 m", "A camera alone cannot measure the gap to a distant vehicle precisely"],
    ["Radar lateral resolution", "1° azimuth noise: ±0.52 m at 30 m, ±2.6 m at 150 m", "Radar alone cannot tell which lane a distant object is in"],
    ["Adverse weather", f"Simulated fog cuts camera good-track time from {g(cam, 'clear day'):.0f}% to {g(cam, 'dense fog'):.0f}% and LiDAR from {g('LiDAR only', 'clear day'):.0f}% to {g('LiDAR only', 'dense fog'):.0f}%", "Radar is the only sensor that is nearly unaffected"],
    ["Stopping distance", f"80 km/h, 1.0 s reaction: {sd['dry (0.7 g), 1.0 s reaction']} m dry (0.7 g), {sd['wet (0.45 g), 1.0 s reaction']} m wet (0.45 g)", "Sets the minimum range at which a confirmed track is needed"],
], caption="Quantified integration challenges", widths=[3.3, 7.2, 6.0])

# ------------------------------------------------------------------ 5-6
R.emit(section(B, "5. AI Techniques to Mitigate the Challenges"))
R.emit(section(B, "6. Proposed Theoretical Framework"), images=IMG, drop={"6.5 Degraded Modes", "6.6 Real-Time Latency Budget"})
R.h2("6.5 Worked Example: Fusing One Camera and One Radar Measurement")
R.p("Take a car 30 m straight ahead. Using the sensor models in Section 8.2, the camera's range noise is 0.2 + 0.03 × 30 = 1.1 m and its bearing noise "
    "of 0.1° becomes 30 × tan(0.1°) = 0.052 m sideways. The radar measures range to 0.15 m, but its 1° bearing noise becomes 0.52 m sideways. Written as "
    "covariance matrices in (range, lateral) coordinates:")
R.code("P_cam   = diag(1.10², 0.052²) = diag(1.210, 0.0027) m²\n"
       "P_radar = diag(0.15², 0.524²) = diag(0.0225, 0.2746) m²\n"
       "P_fused = (P_cam⁻¹ + P_radar⁻¹)⁻¹\n"
       "        = diag( 1 / (1/1.210 + 1/0.0225),  1 / (1/0.0027 + 1/0.2746) )\n"
       "        = diag(0.0221, 0.0027) m²   ->   ±0.149 m range, ±0.052 m lateral")
R.p("The fused estimate is as good as the radar in range and as good as the camera sideways, which neither sensor achieves alone. The fused position "
    "is the same inverse-variance weighting applied to the measurements: x_fused = P_fused (P_cam⁻¹ z_cam + P_radar⁻¹ z_radar). If the camera reports 31.0 m and "
    "the radar 30.1 m, the fused range is 0.0221 × (31.0/1.210 + 30.1/0.0225) = 30.12 m, which is almost entirely the radar's value because the radar is "
    "54 times more precise in range. This is exactly what the Kalman filter does at every update, with the added step of predicting the state forward in time.")
R.figure("figures/fig10_worked_example_ellipses.png", "Two-sigma uncertainty ellipses of the camera, the radar and the fused estimate for a car at 30 m", width_cm=12.5)
R.emit(section(B, "6.5 Degraded Modes"), rename={"6.5 Degraded Modes": "6.6 Degraded Modes"})
R.emit(section(B, "6.6 Real-Time Latency Budget"), rename={"6.6 Real-Time Latency Budget": "6.7 Real-Time Latency Budget"})
tc = E["tracker_compute_us"]
R.p(f"The tracking stage is cheap. In the simulation, one extended Kalman filter update in plain Python and NumPy takes about {tc['single_update_python_numpy']:.0f} µs, "
    f"and updating 100 tracks with one radar scan takes about {tc['100_tracks_one_radar_scan_vectorised']:.0f} µs when vectorised, which is about "
    f"{tc['100_tracks_one_radar_scan_vectorised'] / 1000:.2f} ms or {tc['100_tracks_one_radar_scan_vectorised'] / 1000 / 10 * 100:.1f}% of the 10 ms tracking budget. "
    "Compiled C++ on the target processor would be faster still. The budget is dominated by the neural-network encoders, so model compression "
    "(Section 5.9) is where latency is won or lost.")

# ------------------------------------------------------------------ 7
R.emit(section(B, "7. Simulation and Evaluation Setup Using Public Data"), images=IMG)

# ------------------------------------------------------------------ 8 experiment
R.h1("8. Simulation Experiment and Results")
R.h2("8.1 Purpose and Design")
R.p("Before investing in a test vehicle or a large neural network, the key feasibility question can be tested cheaply: does combining sensors actually "
    "keep track of road users when one sensor degrades, and by how much? I wrote a two-dimensional tracking simulation (src/fusion_sim.py) that models "
    "each sensor's noise, detection probability, maximum range and update rate, simulates realistic Indian-road encounters and fuses the measurements with "
    "an extended Kalman filter. It isolates the fusion logic from the perception network, so it answers the fusion question without needing a trained detector.")
R.p(f"Each combination of scenario, condition and sensor configuration was run {E['runs_per_cell']} times with different random noise "
    f"({len(df)} combinations, {E['total_runs']:,} runs in total, fixed random seed {E['seed']}). The whole experiment runs in about {E['runtime_s']:.0f} seconds on a laptop.")
R.h2("8.2 Sensor Models")
R.table(["Sensor", "Rate", "Range noise (1σ)", "Bearing noise (1σ)", "Detection prob.", "Max range: car / pedestrian / truck"], [
    ["Camera (mono)", "30 Hz", "0.2 m + 3% of range", "0.10°", "0.95", "100 / 60 / 120 m"],
    ["Radar (77 GHz)", "20 Hz", "0.15 m; range-rate 0.1 m/s", "1.0°", "0.92 (0.75 pedestrian)", "160 / 50 / 200 m"],
    ["LiDAR", "10 Hz", "0.03 m", "0.10°", "0.97", "120 / 60 / 150 m"],
    ["Ultrasonic", "20 Hz", "0.02 m + 1%", "none (range only, ±35° beam)", "0.95", "5 m"],
], caption="Sensor models used in the simulation (typical datasheet-level values; assumptions, not a specific product)", widths=[2.6, 1.4, 3.3, 2.6, 2.6, 3.7], font=9)
R.p("Detection probability fades linearly to zero over the last 30% of each sensor's maximum range. Weather and lighting change detection probability, "
    "noise, maximum range and clutter as in the table below. Clutter is modelled as ghost or multipath returns close to the real object, which a tracker must reject.")
R.table(["Condition", "Camera", "Radar", "LiDAR"], [
    ["Clear day", "baseline", "baseline (4% clutter)", "baseline"],
    ["Night", "Pd × 0.84, noise × 1.6, range × 0.6 (headlights)", "unchanged", "unchanged"],
    ["Heavy rain", "Pd × 0.89, noise × 1.3, range × 0.8", "Pd × 0.95, 12% clutter", "Pd × 0.80, noise × 2, range × 0.7, rain returns"],
    ["Dense fog", "Pd × 0.63, noise × 2, range × 0.4", "6% clutter", "Pd × 0.50, noise × 3, range × 0.35"],
    ["Camera blinded (glare, dirt)", "Pd 0.05", "unchanged", "unchanged"],
], caption="Effect of each condition on the sensor models", widths=[3.4, 5.0, 3.8, 4.3], font=9)
R.h2("8.3 Scenarios")
R.table(["ID", "Scenario", "Geometry and motion", "What it tests"], [
    ["S1", "Car cut-in", "Car 25 m ahead in the next lane, 3 m/s slower, moves 3.5 m into the ego lane over 3 s", "Lateral accuracy and fast manoeuvre"],
    ["S2", "Pedestrian crossing", "Ego at 40 km/h; pedestrian 35 m ahead crosses at 1.4 m/s", "Small, low-reflectivity target; radar weak"],
    ["S3", "Motorbike weaving", "Motorbike 40 m ahead weaving ±2 m sideways, a common sight on Indian roads", "Manoeuvring target; filter tuning"],
    ["S4", "Stalled truck on a highway", "Ego at 80 km/h; truck stopped in lane 150 m ahead", "Detection range against stopping distance"],
], caption="Simulated scenarios", widths=[1.0, 3.0, 7.2, 5.3], font=9)
R.h2("8.4 Tracker")
R.p("The tracker is an extended Kalman filter with a constant-velocity state (position and velocity in x and y). Measurements are processed one at a time "
    "in time order, so sensors with different rates are fused naturally. Camera, radar and LiDAR measure range and bearing; radar also measures range-rate; "
    "ultrasonic measures range only. Each measurement is accepted only if it passes a 99.7% chi-square gate on the innovation; when a ghost return and the real "
    "return both arrive, the nearer one (by Mahalanobis distance) is used, so wrong associations are possible, as in a real system. A track is confirmed after "
    "three accepted updates and dropped after 1.5 s without one. The measurement Jacobians are listed in Appendix B.3.")
R.p("Four metrics are reported: the share of time a confirmed track is within 1 m of the truth (good-track time), position RMSE, median and 95th-percentile "
    "error, and the range at which the track is first confirmed. Filter consistency was checked with the normalised estimation error squared (NEES). For a "
    "four-dimensional state a consistent filter has a median NEES of about 3.4. Across scenarios and conditions the typical (median) value was "
    f"{df[df.config == full].nees_median.median():.1f} for the full suite and {df[df.config == cr].nees_median.median():.1f} for camera + radar, so the filter's "
    "uncertainty estimates are realistic. The highest values (up to "
    f"{df[df.config.str.contains('+', regex=False)].nees_median.max():.1f}) appear in the hardest cases, such as the weaving motorbike, where the constant-velocity "
    "model is stretched; an interacting-multiple-model filter would be the production fix.")
R.h2("8.5 Results: Tracking Quality")
rows = [[cfg] + [f"{g(cfg, c):.1f}%" for c in CONDS] for cfg in CFGS]
R.table(["Configuration"] + [c.title() for c in CONDS], rows, caption="Good-track time (error below 1 m), mean of the four scenarios", widths=[4.2, 2.3, 2.3, 2.3, 2.3, 2.6], font=9)
R.figure("figures/fig6_good_track_by_condition.png", "Share of time each configuration keeps the target within 1 m, by condition")
R.p(f"Three patterns stand out. First, each single sensor has a condition that breaks it: the camera at night, in fog and when blinded; LiDAR in fog "
    f"({g('LiDAR only', 'dense fog'):.1f}%) and heavy rain ({g('LiDAR only', 'heavy rain'):.1f}%). Radar is steady in all conditions ({g('Radar only', 'heavy rain'):.1f}–{g('Radar only', 'clear day'):.1f}%) "
    f"but is coarse sideways at long range, so it rarely gets under 1 m on the distant truck. Second, fusion beats the best single sensor in every condition, and "
    f"the full suite never drops below {min(g(full, c) for c in CONDS):.1f}%. Third, adding LiDAR to camera + radar helps most at night and when the camera is blinded "
    f"({g(cr, 'camera blinded'):.1f}% to {g(full, 'camera blinded'):.1f}%), and adds little in fog, where LiDAR itself degrades.")
rows = [[cfg] + [f"{rmse[c][cfg]:.2f}" for c in CONDS] for cfg in CFGS]
R.table(["Configuration"] + [c.title() for c in CONDS], rows, caption="Position RMSE of confirmed tracks in metres, mean of the four scenarios", widths=[4.2, 2.3, 2.3, 2.3, 2.3, 2.6], font=9)
R.figure("figures/fig7_rmse_heatmap.png", "Position RMSE heat map (green is better)", width_cm=14)
sel = df[(df.config.isin([cam, cr, full])) & (df.condition.isin(["clear day", "dense fog"]))]
rows = [[r.scenario, r.condition, r.config, f"{r.pos_err_median:.2f}", f"{r.pos_err_p95:.2f}", f"{r.good_track_pct:.1f}%"] for r in sel.itertuples()]
R.table(["Scenario", "Condition", "Configuration", "Median error (m)", "95th pct (m)", "Good-track"], rows,
        caption="Per-scenario detail for clear day and dense fog", widths=[3.8, 2.2, 4.0, 2.1, 1.9, 2.0], font=8.5)
R.p("RMSE is sensitive to rare large errors, so the median and 95th percentile are shown too. For example, in S2 at night the full suite has a median error of "
    "only 0.02 m but an RMSE of 0.56 m, because in a few runs a ghost return was briefly accepted. This is why the design keeps an independent radar and "
    "ultrasonic safety channel (Section 6.3) instead of trusting a single fused output.")
R.figure("figures/fig8_bike_trajectory.png", "One simulated run of the weaving motorbike at night: camera-only estimates scatter, fused estimates follow the true path")

R.h2("8.6 Results: Detection Range Against Stopping Distance")
R.p("The stalled-truck scenario turns accuracy into a safety question. At 80 km/h (22.2 m/s) the stopping distance with a 1.0 s reaction time is "
    f"22.2 × 1.0 + 22.2² / (2 × 0.7 × 9.81) = {sd['dry (0.7 g), 1.0 s reaction']} m on a dry road and {sd['wet (0.45 g), 1.0 s reaction']} m on a wet road (0.45 g). "
    f"An automatic emergency brake with a 0.3 s system delay and 0.8 g needs {sd['AEB only (0.8 g), 0.3 s system delay']} m. A confirmed track must exist before these distances.")
rows = [[cfg] + [f"{truck[c][cfg]:.0f} m" for c in CONDS] for cfg in CFGS]
R.table(["Configuration"] + [c.title() for c in CONDS], rows, caption="Median range at which the stalled truck is first confirmed", widths=[4.2, 2.3, 2.3, 2.3, 2.3, 2.6], font=9)
R.figure("figures/fig9_truck_detection_range.png", "Range at first confirmed track for the stalled truck, against dry and wet stopping distances")
R.p(f"A camera alone confirms the truck at {truck['clear day'][cam]:.0f} m on a clear day, enough to stop on a dry road, but only at {truck['night'][cam]:.0f} m at night, "
    f"{truck['dense fog'][cam]:.0f} m in fog and {truck['camera blinded'][cam]:.0f} m when blinded, all shorter than the {sd['dry (0.7 g), 1.0 s reaction']} m dry stopping distance. "
    f"LiDAR alone fails in fog ({truck['dense fog']['LiDAR only']:.0f} m). Every configuration that includes radar confirms the truck at about 145–148 m in every "
    "condition, nearly twice the wet stopping distance. For a camera-based product like Saarthi this is the single strongest argument for adding a radar.")

R.h2("8.7 Results: Robustness Ratios and Sensor Failure")
rows = [[cfg] + [f"{rr[c][cfg]:.2f}" for c in CONDS[1:]] for cfg in CFGS]
R.table(["Configuration"] + [c.title() for c in CONDS[1:]], rows, caption="Robustness ratio: good-track time in each condition divided by good-track time on a clear day", widths=[4.4, 3.0, 3.0, 3.0, 3.1], font=9)
R.p(f"The success criteria in Section 7.6 ask for a robustness ratio of at least 0.80 for a single-sensor failure and at least 0.75 for rain and night. "
    f"In this simulation the full suite reaches a minimum of {min_rr_full:.2f} and camera + radar a minimum of {min(rr[c][cr] for c in CONDS):.2f}, so both pass. "
    f"The camera alone fails the criterion at night ({rr['night'][cam]:.2f}), in fog ({rr['dense fog'][cam]:.2f}) and when blinded ({rr['camera blinded'][cam]:.2f}); "
    f"LiDAR alone fails in fog ({rr['dense fog']['LiDAR only']:.2f}). Sensor dropout during training (Section 5.4) is meant to give a neural fusion network the same "
    "graceful behaviour that the Kalman filter shows here.")

R.h2("8.8 Time Synchronisation and Calibration Experiments")
sync = pd.DataFrame(E["time_sync"]); cal = pd.DataFrame(E["calibration"])
R.p("Two further experiments inject the errors described in Sections 4.1 and 4.2. In the first, camera frames are stamped 0–150 ms late, as happens "
    "without hardware time synchronisation. In the second, the radar is rotated by 0–3° and the fused system is run at night, when it relies more on radar.")
rows = [[r.scenario[:2], f"{r.delay_ms} ms", f"{r.camera_only_rmse:.2f}", f"{r.fused_rmse:.2f}"] for r in sync.itertuples()]
R.table(["Scenario", "Camera timestamp error", "Camera only RMSE (m)", "Camera + radar RMSE (m)"], rows, caption="Effect of camera timestamp errors", widths=[2.2, 4.0, 4.0, 4.2], font=9)
rows = [[f"{r.bias_deg}°", f"{r.lateral_offset_at_50m_m:.2f} m", f"{r.radar_only_rmse:.2f}", f"{r.fused_night_rmse:.2f}", f"{r.fused_night_rmse_with_online_correction:.2f}"] for r in cal.itertuples()]
R.table(["Radar misalignment", "Offset at 50 m", "Radar only RMSE (m)", "Fused, night (m)", "Fused, night, online correction (m)"], rows,
        caption="Effect of radar azimuth misalignment (S1 car cut-in)", widths=[2.8, 2.6, 3.0, 2.8, 4.2], font=9)
R.figure("figures/fig11_sync_and_calibration.png", "Left: position error against camera timestamp error. Right: position error against radar misalignment")
r100 = sync[(sync.scenario.str.startswith("S4")) & (sync.delay_ms == 100)].iloc[0]
r3 = cal[cal.bias_deg == 3.0].iloc[0]; r1 = cal[cal.bias_deg == 1.0].iloc[0]
R.p(f"Timing errors hurt the camera-only tracker badly on the fast-closing truck (RMSE {sync.iloc[0].camera_only_rmse:.2f} m at 0 ms, {r100.camera_only_rmse:.2f} m at 100 ms) "
    f"but barely affect camera + radar fusion ({r100.fused_rmse:.2f} m at 100 ms), because the radar's precise range and range-rate dominate the range estimate. "
    "Hardware time synchronisation (PTP, IEEE 1588) is still required, because a neural network fusing features cannot reject late data the way a gated filter can.")
R.p(f"Misalignment shows a more worrying effect. Up to 1° the fused system absorbs it, and a simple online correction (the mean of the first 20 radar bearing "
    f"innovations, subtracted from later measurements) removes most of what is left ({r1.fused_night_rmse:.3f} m to {r1.fused_night_rmse_with_online_correction:.3f} m). "
    f"At 3° the fused night tracker ({r3.fused_night_rmse:.2f} m) becomes worse than the radar alone ({r3.radar_only_rmse:.2f} m), because the camera and the radar now "
    "disagree and the filter is pulled between them. The online correction brings it back to the radar-only level but not lower. The design rule that follows is "
    "the one in the degraded-modes table (Section 6.6): correct small drifts online, but flag anything above about 1° for workshop recalibration and reduce trust in that sensor until then.")

R.h2("8.9 What the Experiment Shows and What It Does Not")
R.bullets([
    ("Shows:", "that the fusion architecture is sound: combining complementary sensors keeps tracking quality high in every simulated condition and protects "
     "the stopping-distance margin that a camera alone loses at night, in fog and in glare."),
    ("Shows:", "that calibration and timing errors have measurable effects, and that a gated filter with online bias correction tolerates small ones."),
    ("Does not show:", "the accuracy of a neural detector. The simulation assumes each sensor detects the object with a given probability; real detection on "
     "Indian roads must be measured on nuScenes, RADIATE and IDD-3D (experiments E1–E5)."),
    ("Does not show:", "multi-object association in dense traffic, occlusion, or the full physics of radar multipath. These need CARLA and real recorded data."),
])

# ------------------------------------------------------------------ 9 feasibility
R.h1("9. Feasibility Assessment")
for old, new in [("8.1 Technical Feasibility", "9.1 Technical Feasibility"), ("8.2 Economic Feasibility", "9.2 Economic Feasibility")]:
    R.emit(section(B, old), rename={old: new})
cost = [("6 automotive cameras (2–8 MP)", 6, 3200, "A"), ("Front long-range 4D imaging radar", 1, 12000, "A"), ("Corner radars", 4, 5500, "A"),
        ("Ultrasonic sensors", 12, 600, "A"), ("L2+ AI processor (30–100 TOPS)", 1, 35000, "A"), ("Safety microcontroller", 1, 2500, "A"),
        ("Harness, mounts, cleaning jets", 1, 8000, "A"), ("Forward long-range LiDAR", 1, 42000, "B"), ("Compute upgrade for LiDAR fusion", 1, 25000, "B"),
        ("Two short-range corner LiDARs", 2, 25000, "C"), ("Redundant compute for driverless operation", 1, 60000, "C")]
tier = {"A": 0, "B": 0, "C": 0}
rows = []
for name, q, unit, t in cost:
    tier[t] += q * unit
    rows.append([name, str(q), f"₹{unit:,}", f"₹{q * unit:,}", t])
ta, tb, tcc = tier["A"], tier["A"] + tier["B"], tier["A"] + tier["B"] + tier["C"]
R.table(["Item", "Qty", "Unit cost", "Line cost", "Added in tier"], rows, caption="Indicative sensor and compute cost per vehicle at 10,000+ units a year (2026 estimates)", widths=[6.0, 1.2, 2.4, 2.6, 2.4], font=9)
R.p(f"Tier A costs about ₹{ta / 1e5:.2f} lakh per vehicle, Tier B about ₹{tb / 1e5:.2f} lakh and Tier C about ₹{tcc / 1e5:.2f} lakh. These are estimates built from "
    "public price ranges for automotive-grade parts (a long-range automotive LiDAR is taken at about US$500, ₹42,000) and will move with volume and exchange rates. "
    "For a ₹12–20 lakh car, Tier A adds 5–9% to the price, which is in line with today's Level 2 ADAS option packs in India; Tier C only makes sense for "
    "shuttles, ports and campuses where a vehicle replaces a paid driver.")
for old, new in [("8.3 Operational Feasibility", "9.3 Operational Feasibility"), ("8.4 Regulatory and Safety Feasibility", "9.4 Regulatory and Safety Feasibility")]:
    R.emit(section(B, old), rename={old: new})

R.h2("9.5 Feasibility Scorecard")
R.p("Each criterion is scored from 1 to 5 against fixed anchors, using evidence from this study, and weighted by its importance to the decision. The "
    "scorecard is filled in twice: for the near-term target (Level 2+ assistance and geofenced low-speed autonomy) and for open-road driverless operation in India.")
crit = [
    ("Tracking accuracy", 15, "5: good-track ≥ 95% and p95 error < 0.5 m on a clear day; 3: ≥ 85%; 1: < 70%", f"Full suite {g(full, 'clear day'):.1f}% good-track, median error 0.02–0.04 m (Section 8.5)", 5, 4),
    ("Robustness in rain, fog, night", 15, "5: robustness ratio ≥ 0.90 in all conditions; 3: ≥ 0.75; 1: < 0.6", f"Minimum ratio {min_rr_full:.2f} in simulation; real data still needed", 4, 3),
    ("Single-sensor failure tolerance", 10, "5: no criterion broken when any one sensor fails; 3: degraded but safe; 1: unsafe", f"Camera blinded: {g(full, 'camera blinded'):.1f}% good-track (Section 8.7)", 4, 3),
    ("Detection range vs stopping distance", 10, "5: confirmed range ≥ 1.5 × wet stopping distance in all conditions; 3: ≥ 1.0 ×; 1: below", f"About 146 m vs {sd['wet (0.45 g), 1.0 s reaction']} m wet (1.9 ×)", 5, 4),
    ("Real-time compute", 10, "5: ≥ 30% latency headroom; 3: 0–30%; 1: over budget", "95 ms against a 100 ms target (5% headroom); tracker negligible", 3, 2),
    ("Cost per vehicle", 10, "5: under 5% of vehicle price; 3: 5–10%; 1: over 20%", f"Tier A ₹{ta / 1e5:.2f} lakh (5–9% of a ₹12–20 lakh car); Tier C ₹{tcc / 1e5:.2f} lakh", 3, 2),
    ("Indian training and test data", 10, "5: large Indian multi-sensor dataset exists; 3: partial; 1: none", "IDD and IDD-3D only; no Indian radar dataset", 3, 1),
    ("Regulatory path in India", 10, "5: clear approval route; 3: route for assistance only; 1: no route", "AIS and Bharat NCAP cover assistance; no route for driverless vehicles", 4, 1),
    ("Service and operations", 10, "5: existing network can support it; 3: needs new tools and training; 1: not supportable", "Needs calibration rigs, cleaning systems and trained technicians", 3, 2),
]
rows, s1, s2 = [], 0, 0
for name, w, anchor, ev, a, b in crit:
    s1 += w * a; s2 += w * b
    rows.append([name, f"{w}%", anchor, ev, str(a), str(b)])
rows.append(["Weighted score (out of 5)", "100%", "", "", f"{s1 / 100:.2f}", f"{s2 / 100:.2f}"])
R.table(["Criterion", "Weight", "Scoring anchors", "Evidence", "L2+ / geofenced", "Open-road driverless"], rows,
        caption="Weighted feasibility scorecard", widths=[2.8, 1.3, 4.6, 4.4, 1.7, 1.7], font=8)
R.p(f"Decision rule: GO if the weighted score is at least 3.5 and no criterion scores 1; CONDITIONAL GO between 2.5 and 3.5; NO-GO below 2.5 or if any "
    f"criterion scores 1. Level 2+ and geofenced autonomy score {s1 / 100:.2f}, a GO. Open-road driverless operation scores {s2 / 100:.2f} but has two criteria at 1 "
    "(Indian data and regulation), so it is a NO-GO for now. Simulated evidence was capped at 4 for robustness and failure tolerance, even where the "
    "simulation alone would justify a 5, because it still has to be confirmed on real data.")
R.emit(section(B, "8.6 Recommendation and Next Steps"), rename={"8.6 Recommendation and Next Steps": "9.6 Recommendation and Next Steps"})

# ------------------------------------------------------------------ 10-11
R.h1("10. Risks and Limitations of This Study")
R.bullets([
    "The simulation is two-dimensional and tracks one object at a time. It tests fusion logic, not neural detection, multi-object association or occlusion.",
    "Sensor models are typical datasheet-level values, not measurements of chosen products. Real noise can be non-Gaussian, for example radar multipath near "
    "metal barriers, which is common on Indian flyovers.",
    "Weather effects are simple multipliers. RADIATE (radar in bad weather) and nuScenes rain and night scenes should be used to calibrate them.",
    "Public datasets are mostly foreign; results on them may overestimate performance on Indian roads, which is why IDD-3D and an Indian collection campaign are in the plan.",
    "Prices and regulations are moving quickly; the cost table and regulatory scores should be reviewed every six months.",
])
R.h1("11. Conclusion")
R.p("Sensor fusion is the foundation of safe automated driving because no single sensor sees reliably in every condition. The literature shows that "
    "bird's-eye-view, uncertainty-aware fusion is mature enough for production driver assistance, and this study's own experiment shows why it is needed: "
    f"across {E['total_runs']:,} simulated runs, every single sensor had a condition that broke it, while camera + radar + LiDAR fusion kept the target within "
    f"1 m for at least {min(g(full, c) for c in CONDS):.0f}% of the time in every condition and confirmed a stalled truck about 146 m away, nearly twice the wet stopping distance at 80 km/h.")
R.p("The feasibility scorecard gives a clear answer. A staged Saarthi platform is feasible now for Level 2+ assistance and geofenced low-speed autonomy "
    f"(weighted score {s1 / 100:.2f}, GO), starting with a Tier A camera + radar + ultrasonic suite at about ₹{ta / 1e5:.2f} lakh per vehicle. Open-road driverless "
    f"operation in India is not yet feasible ({s2 / 100:.2f}, NO-GO), held back by the lack of Indian multi-sensor data and of a regulatory route rather than by the "
    "fusion technology itself. The next step is to run experiments E1–E7 on public data and an Indian test vehicle, using the simulation in this report as the "
    "reference for what good fusion behaviour looks like.")

# ------------------------------------------------------------------ references + appendices
refs = section(B, "References")
extra = ["[25]  Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation. Wiley.",
         "[26]  Thrun, S., Burgard, W. and Fox, D. (2005). Probabilistic Robotics. MIT Press.",
         "[27]  Wong, J. Y. (2008). Theory of Ground Vehicles, 4th ed. Wiley (tyre-road friction values used for stopping distance).",
         f"[28]  Sharma, H. (2026). Sensor fusion feasibility study: simulation code and results. {GH}"]
R.emit(refs)
for e in extra:
    R.p(e)
R.h1("Appendix A: Supplementary Diagrams")
R.p("The first diagram shows the real-time processing cycle, including the two decision points where sensor health and agreement with the safety channel "
    "are checked. The second shows the coverage provided by the proposed sensor suite.")
R.figure(IMG[3], "Real-time processing cycle (flowchart)")
R.figure(IMG[4], "Coverage of the proposed sensor suite (schematic)")
R.emit(section(B, "Appendix B: Model Equations"))
R.h2("B.3 Measurement Model Used in the Simulation")
R.p("For a state x = [pₓ, p_y, vₓ, v_y] and r = √(pₓ² + p_y²), the measurement functions and the non-zero Jacobian terms are:")
R.code("range       h_r  = r                    ∂h_r/∂pₓ = pₓ/r,   ∂h_r/∂p_y = p_y/r\n"
       "bearing     h_θ  = atan2(p_y, pₓ)        ∂h_θ/∂pₓ = −p_y/r², ∂h_θ/∂p_y = pₓ/r²\n"
       "range-rate  h_ṙ  = (pₓvₓ + p_yv_y)/r     ∂h_ṙ/∂vₓ = pₓ/r,   ∂h_ṙ/∂v_y = p_y/r\n"
       "                                        ∂h_ṙ/∂pₓ = (vₓr² − pₓ(pₓvₓ + p_yv_y))/r³ (and similarly for p_y)\n"
       "gate: accept if yᵀ S⁻¹ y < 11.6 (2 dof) or 14.2 (3 dof), 99.7% chi-square")
R.emit(section(B, "Appendix C: Glossary"))
R.h1("Appendix D: How to Reproduce the Results")
R.code("pip install numpy pandas matplotlib python-docx\npython src/fusion_sim.py      # about 40 s, writes results/ and figures/\npython src/build_report.py    # rebuilds this document")
words = R.save(OUT)
print("words:", words, "figures:", R.fig_n, "tables:", R.tab_n)
