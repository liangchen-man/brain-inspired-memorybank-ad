"""
Case 2 Temporal State Baseline — minimal, no deep models.

Purpose: Demonstrate that Case 2 (conveyor workstation) requires temporal state
awareness and CANNOT rely on single-frame static memory bank (PatchCore, PaDiM, etc.).

Input: case2_frame_timeseries.json (240 frames, ~2s intervals)
Output: per-frame state labels, event flags, smoothed states, and figures.

All conclusions: 候选，需 Codex/用户确认
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# Use a CJK-capable font
_cjk_font = None
for _fname in ["Microsoft YaHei", "SimHei"]:
    for _f in fm.fontManager.ttflist:
        if _f.name == _fname:
            _cjk_font = _f.fname
            break
    if _cjk_font:
        break
if _cjk_font:
    plt.rcParams["font.family"] = fm.FontProperties(fname=_cjk_font).get_name()
    print(f"[OK] Using font: {_cjk_font}")
else:
    print("[WARN] No CJK font found, Chinese characters may not render")
plt.rcParams["axes.unicode_minus"] = False

# ——— Config ———
ROOT = Path("E:/leinaozuoye")
TS_JSON = ROOT / "results/week2/case2_frame_timeseries.json"
OUT_CSV = ROOT / "results/week2/case2_temporal_baseline.csv"
OUT_FIG_DIR = ROOT / "report/figures/week2_case2_temporal_baseline"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds (derived from prior frame-diff analysis)
DIFF_THRESHOLD = 30.87        # mean + 3*std — definite indexing motion
DIFF_ELEVATED = 16.0          # moderately elevated — possible transport
DIFF_LOW = 14.0               # low — stationary/dwell
BRIGHTNESS_TRANSITION_START = 360.0  # seconds
BRIGHTNESS_TRANSITION_END = 460.0    # seconds
CARTRIDGE_CHANGE_START = 316.0       # seconds (4 cartridges)
CARTRIDGE_CHANGE_END = 435.0         # seconds (3 cartridges)
# Refined by qwen3-vl-plus dense audit (2026-06-21):
# Human intervention window: t≈391-395s (blue cleanroom gown, hand holding 4th cartridge)
# Cartridge removal: t≈403-405s (rightmost position went from occupied → empty)
HUMAN_INTERACTION_START = 389.0      # seconds (human first visible behind guard)
HUMAN_INTERACTION_END = 397.0        # seconds (human departed)
CARTRIDGE_REMOVAL_WINDOW = (403.0, 405.0)  # seconds (automated or unseen removal)

# Sliding window for median smoothing
SMOOTH_WINDOW = 5  # frames (~10s)

# ——— Load data ———
with open(TS_JSON) as f:
    ts_data = json.load(f)

frames = ts_data["frames"]
n = len(frames)

# ——— Helper: sliding median ———
def sliding_median(values, window):
    """Return median-filtered series, same length as input."""
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(float(np.median(values[lo:hi])))
    return out

# ——— Step 1: assign raw state per frame ———
def assign_raw_state(idx, t_sec, mean_bright, frame_diff):
    """Rule-based state assignment (no ML)."""

    if t_sec == 0.0:
        return "INIT"

    in_brightness_zone = BRIGHTNESS_TRANSITION_START <= t_sec <= BRIGHTNESS_TRANSITION_END
    in_cartridge_change = CARTRIDGE_CHANGE_START <= t_sec <= CARTRIDGE_CHANGE_END
    in_human_window = HUMAN_INTERACTION_START <= t_sec <= HUMAN_INTERACTION_END

    # 1) Human interaction window
    if in_human_window:
        return "HUMAN_INTERACTION"

    # 2) Indexing spike
    if frame_diff is not None and frame_diff >= DIFF_THRESHOLD:
        return "INDEXING"

    # 3) Brightness transition zone + cartridge count change
    if in_brightness_zone and in_cartridge_change:
        return "TRANSITION"

    # 4) Elevated diff → transport
    if frame_diff is not None and frame_diff >= DIFF_ELEVATED:
        return "TRANSPORT"

    # 5) Late video, post cartridge change → END_STATE
    if t_sec >= CARTRIDGE_CHANGE_END:
        return "END_STATE"

    # 6) Brightness zone only (after cartridge change window)
    if in_brightness_zone:
        return "TRANSITION"

    # 7) Low diff, mid-video → DWELL
    if frame_diff is not None and frame_diff < DIFF_LOW:
        return "DWELL"

    # 8) Remaining → PROCESSING (mid-video, moderate diff, cartridges present)
    return "PROCESSING"

# ——— Step 2: compute raw states ———
raw_states = []
for i, f in enumerate(frames):
    s = assign_raw_state(i, f["t_sec"], f["mean_bright"], f.get("frame_diff"))
    raw_states.append(s)

# ——— Step 3: sliding median smoothing on frame_diff ———
diffs_raw = [f.get("frame_diff") or 0.0 for f in frames]
diffs_smooth = sliding_median(diffs_raw, SMOOTH_WINDOW)

# ——— Step 4: assign smoothed states (using smoothed diffs) ———
smoothed_states = []
for i, f in enumerate(frames):
    fd = diffs_smooth[i]
    s = assign_raw_state(i, f["t_sec"], f["mean_bright"], fd if fd > 0 else None)
    smoothed_states.append(s)

# ——— Step 5: event detection ———
def detect_events(i, t_sec, raw_s, smooth_s, diff_raw, diff_smooth, bright):
    """Flag significant events."""
    events = []

    # A) Human interaction window
    if smooth_s == "HUMAN_INTERACTION":
        if raw_s != "HUMAN_INTERACTION":
            events.append(("HUMAN_WINDOW_ENTER", f"t={t_sec:.1f}s, entering human interaction window (t={HUMAN_INTERACTION_START:.0f}-{HUMAN_INTERACTION_END:.0f}s)"))
        else:
            events.append(("HUMAN_PRESENT", f"t={t_sec:.1f}s, human in blue cleanroom gown near conveyor"))

    # B) INDEXING peak
    if raw_s == "INDEXING":
        events.append(("INDEXING_PEAK", f"frame_diff={diff_raw:.1f}, >threshold={DIFF_THRESHOLD:.1f}"))

    # B) State transition (smoothed state changed from previous frame)
    if i > 0 and smooth_s != smoothed_states[i - 1]:
        prev_s = smoothed_states[i - 1]
        events.append(("STATE_CHANGE", f"{prev_s} -> {smooth_s}"))

    # C) Entering brightness transition zone
    if i > 0 and BRIGHTNESS_TRANSITION_START <= t_sec and frames[i - 1]["t_sec"] < BRIGHTNESS_TRANSITION_START:
        events.append(("BRIGHTNESS_ZONE_ENTER", f"t={t_sec:.1f}s, brightness={bright:.1f}"))

    # D) Exiting brightness transition zone
    if i > 0 and t_sec > BRIGHTNESS_TRANSITION_END and frames[i - 1]["t_sec"] <= BRIGHTNESS_TRANSITION_END:
        events.append(("BRIGHTNESS_ZONE_EXIT", f"t={t_sec:.1f}s, brightness={bright:.1f}"))

    # E) Cartridge count change candidate window
    if CARTRIDGE_CHANGE_START <= t_sec <= CARTRIDGE_CHANGE_END:
        # Look for brightness dips within this window
        if bright < 115.0:  # below typical brightness
            events.append(("CARTRIDGE_CHANGE_CANDIDATE", f"t={t_sec:.1f}s, brightness={bright:.1f} (dip in change window)"))

    # F) Cartridge removal candidate (t≈403-405s)
    if CARTRIDGE_REMOVAL_WINDOW[0] <= t_sec <= CARTRIDGE_REMOVAL_WINDOW[1]:
        if "CARTRIDGE_CHANGE_CANDIDATE" not in [e[0] for e in events]:
            events.append(("CARTRIDGE_REMOVAL", f"t={t_sec:.1f}s, rightmost cartridge removed between t=403-405s; 4→3"))

    # G) Spike suppression: raw spike smoothed away
    if raw_s == "INDEXING" and smooth_s != "INDEXING":
        events.append(("SPIKE_SUPPRESSED", f"raw_diff={diff_raw:.1f}, smooth_diff={diff_smooth:.1f}"))

    return events

# ——— Step 6: assemble per-frame output ———
rows = []
for i, f in enumerate(frames):
    t_sec = f["t_sec"]
    bright = f["mean_bright"]
    diff_raw = diffs_raw[i]
    diff_sm = diffs_smooth[i]
    raw_s = raw_states[i]
    smooth_s = smoothed_states[i]

    events = detect_events(i, t_sec, raw_s, smooth_s, diff_raw, diff_sm, bright)
    event_flag = len(events) > 0
    event_types = "; ".join(e[0] for e in events) if events else ""
    explanations = " | ".join(f"{e[0]}: {e[1]}" for e in events) if events else ""

    # Human-readable explanation for this frame's state
    if smooth_s == "INIT":
        explain = "Start of video; initial state before any conveyor motion."
    elif smooth_s == "INDEXING":
        explain = "Conveyor indexing in progress — rapid frame-to-frame change from rigid-body translation."
    elif smooth_s == "TRANSPORT":
        explain = "Cartridges in moderate motion; motion blur visible; elevated frame diff but below indexing peak."
    elif smooth_s == "DWELL":
        explain = "Conveyor stationary; cartridges aligned at processing stations; low frame diff."
    elif smooth_s == "PROCESSING":
        explain = "Mid-cycle stationary phase; actuator possibly engaged; inspection or testing in progress."
    elif smooth_s == "HUMAN_INTERACTION":
        explain = "Human operator in blue cleanroom gown near conveyor (t≈391-395s); hand holding/repositioning cartridge. Manual intervention window ~4s."
    elif smooth_s == "TRANSITION":
        explain = "Within brightness transition zone (360-460s); lighting/camera adjustment detected; cartridge count change candidate."
    elif smooth_s == "END_STATE":
        explain = "Post-transition end-of-run state; 3 cartridges visible (was 4); system idle."
    else:
        explain = ""

    rows.append({
        "timestamp_sec": round(t_sec, 2),
        "mean_brightness": round(bright, 2),
        "frame_diff_raw": round(diff_raw, 2),
        "frame_diff_smooth": round(diff_sm, 2),
        "state_raw": raw_s,
        "state_smoothed": smooth_s,
        "event_flag": int(event_flag),
        "event_type": event_types,
        "explanation": explanations if explanations else explain,
    })

# ——— Write CSV ———
with open(OUT_CSV, "w", encoding="utf-8") as f:
    header = "timestamp_sec,mean_brightness,frame_diff_raw,frame_diff_smooth,state_raw,state_smoothed,event_flag,event_type,explanation\n"
    f.write(header)
    for r in rows:
        f.write(f"{r['timestamp_sec']},{r['mean_brightness']},{r['frame_diff_raw']},{r['frame_diff_smooth']},"
                f"{r['state_raw']},{r['state_smoothed']},{r['event_flag']},\"{r['event_type']}\",\"{r['explanation']}\"\n")

print(f"[OK] CSV written: {OUT_CSV} ({len(rows)} rows)")

# ——— Statistics ———
state_counts_raw = {}
state_counts_smooth = {}
for r in rows:
    state_counts_raw[r["state_raw"]] = state_counts_raw.get(r["state_raw"], 0) + 1
    state_counts_smooth[r["state_smoothed"]] = state_counts_smooth.get(r["state_smoothed"], 0) + 1

print("\n=== Raw state distribution ===")
for s, c in sorted(state_counts_raw.items()):
    pct = 100 * c / len(rows)
    print(f"  {s}: {c} frames ({pct:.1f}%)")

print("\n=== Smoothed state distribution ===")
for s, c in sorted(state_counts_smooth.items()):
    pct = 100 * c / len(rows)
    print(f"  {s}: {c} frames ({pct:.1f}%)")

# Count state changes
state_changes_raw = sum(1 for i in range(1, len(rows)) if rows[i]["state_raw"] != rows[i - 1]["state_raw"])
state_changes_smooth = sum(1 for i in range(1, len(rows)) if rows[i]["state_smoothed"] != rows[i - 1]["state_smoothed"])
print(f"\nState changes (raw): {state_changes_raw}")
print(f"State changes (smoothed): {state_changes_smooth}")
print(f"Reduction: {state_changes_raw - state_changes_smooth} transitions suppressed")

# Spike suppression count
spikes_suppressed = sum(1 for r in rows if "SPIKE_SUPPRESSED" in r["event_type"])
print(f"Spikes suppressed by smoothing: {spikes_suppressed}")

event_frames = [r for r in rows if r["event_flag"]]
print(f"\nTotal event frames: {len(event_frames)}")
for r in event_frames:
    print(f"  t={r['timestamp_sec']:.1f}s state={r['state_smoothed']} events=[{r['event_type']}]")

# ——— FIGURE 1: Frame diff curve + state zones ———
ts = [r["timestamp_sec"] for r in rows]
diffs_r = [r["frame_diff_raw"] for r in rows]
diffs_s = [r["frame_diff_smooth"] for r in rows]

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(22, 9), sharex=True)

# Top: raw diff with state color bands
state_colors = {
    "INIT": "#e8e8e8", "INDEXING": "#ff6b6b", "TRANSPORT": "#ffa502",
    "DWELL": "#7bed9f", "PROCESSING": "#70a1ff", "TRANSITION": "#eccc68",
    "END_STATE": "#a29bfe", "HUMAN_INTERACTION": "#ff6348",
}
# Draw state background bands
prev_s = rows[0]["state_raw"]
band_start = ts[0]
for i in range(1, len(rows)):
    s = rows[i]["state_raw"]
    if s != prev_s:
        ax1a.axvspan(band_start, ts[i], alpha=0.15, color=state_colors.get(prev_s, "#cccccc"))
        band_start = ts[i]
        prev_s = s
ax1a.axvspan(band_start, ts[-1], alpha=0.15, color=state_colors.get(prev_s, "#cccccc"))

ax1a.plot(ts, diffs_r, "r-", linewidth=0.7, alpha=0.85, label="Raw frame diff")
ax1a.axhline(y=DIFF_THRESHOLD, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label=f"Indexing threshold ({DIFF_THRESHOLD:.1f})")
ax1a.axhline(y=DIFF_ELEVATED, color="orange", linestyle=":", linewidth=0.8, alpha=0.5, label=f"Transport threshold ({DIFF_ELEVATED:.1f})")
ax1a.set_ylabel("Frame Diff (mean abs pixel)", fontsize=11)
ax1a.set_title("Case 2 — Raw Frame Difference with State Background Bands", fontsize=13, fontweight="bold")
ax1a.legend(fontsize=8, loc="upper right")
ax1a.grid(True, alpha=0.25)

# Bottom: smoothed diff
prev_s2 = rows[0]["state_smoothed"]
band_start2 = ts[0]
for i in range(1, len(rows)):
    s2 = rows[i]["state_smoothed"]
    if s2 != prev_s2:
        ax1b.axvspan(band_start2, ts[i], alpha=0.15, color=state_colors.get(prev_s2, "#cccccc"))
        band_start2 = ts[i]
        prev_s2 = s2
ax1b.axvspan(band_start2, ts[-1], alpha=0.15, color=state_colors.get(prev_s2, "#cccccc"))

ax1b.plot(ts, diffs_s, "b-", linewidth=1.0, alpha=0.9, label=f"Smoothed frame diff (median window={SMOOTH_WINDOW})")
ax1b.axhline(y=DIFF_THRESHOLD, color="red", linestyle="--", linewidth=1.0, alpha=0.6)
ax1b.axhline(y=DIFF_ELEVATED, color="orange", linestyle=":", linewidth=0.8, alpha=0.5)
ax1b.set_xlabel("Time (s)", fontsize=11)
ax1b.set_ylabel("Frame Diff (smoothed)", fontsize=11)
ax1b.set_title("Case 2 — Smoothed Frame Difference with Smoothed State Background Bands", fontsize=13, fontweight="bold")
ax1b.legend(fontsize=8, loc="upper right")
ax1b.grid(True, alpha=0.25)

# Legend for state colors
legend_patches = [mpatches.Patch(color=c, alpha=0.3, label=s) for s, c in state_colors.items()]
ax1b.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.0, label=f"Threshold ({DIFF_THRESHOLD:.1f})"),
], fontsize=7, loc="upper left", ncol=4)

plt.tight_layout()
fig1.savefig(OUT_FIG_DIR / "case2_frame_diff_states.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"[OK] Figure 1 saved: {OUT_FIG_DIR / 'case2_frame_diff_states.png'}")

# ——— FIGURE 2: Brightness curve + transition zone ———
bright_vals = [r["mean_brightness"] for r in rows]

fig2, ax2 = plt.subplots(figsize=(22, 6))
ax2.plot(ts, bright_vals, "b-", linewidth=0.8, alpha=0.8, label="Mean brightness")
ax2.axvspan(BRIGHTNESS_TRANSITION_START, BRIGHTNESS_TRANSITION_END, alpha=0.18, color="orange", label=f"Brightness transition zone\n(t={BRIGHTNESS_TRANSITION_START:.0f}-{BRIGHTNESS_TRANSITION_END:.0f}s)")
ax2.axvspan(CARTRIDGE_CHANGE_START, CARTRIDGE_CHANGE_END, alpha=0.10, color="purple", label=f"Cartridge count change candidate\n(t={CARTRIDGE_CHANGE_START:.0f}-{CARTRIDGE_CHANGE_END:.0f}s)")
ax2.axvline(x=25.7, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="t=25.7s indexing peak")
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.set_ylabel("Mean Brightness (0-255)", fontsize=11)
ax2.set_title("Case 2 — Brightness Timeseries with Transition Zone & Cartridge Change Window", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.25)
plt.tight_layout()
fig2.savefig(OUT_FIG_DIR / "case2_brightness_transition.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[OK] Figure 2 saved: {OUT_FIG_DIR / 'case2_brightness_transition.png'}")

# ——— FIGURE 3: State machine diagram (conceptual) ———
fig3, ax3 = plt.subplots(figsize=(18, 7))
ax3.set_xlim(0, 16)
ax3.set_ylim(0, 8)
ax3.axis("off")
ax3.set_title("Case 2 — Candidate State Machine (conceptual, 需 Codex/用户确认)", fontsize=14, fontweight="bold", pad=20)

# Node positions
nodes = {
    "INIT": (2, 6.5),
    "INDEXING": (6, 6.5),
    "TRANSPORT": (10, 6.5),
    "DWELL": (10, 4.5),
    "PROCESSING": (6, 4.5),
    "TRANSITION": (2, 4.5),
    "HUMAN_INTERACTION": (6, 2.5),
    "END_STATE": (10, 2.5),
}
node_colors = {
    "INIT": "#e8e8e8", "INDEXING": "#ff6b6b", "TRANSPORT": "#ffa502",
    "DWELL": "#7bed9f", "PROCESSING": "#70a1ff", "TRANSITION": "#eccc68",
    "END_STATE": "#a29bfe", "HUMAN_INTERACTION": "#ff6348",
}
node_descs = {
    "INIT": "t=0s\nVideo start",
    "INDEXING": "Conveyor step motion\n(frame_diff > 30.9)",
    "TRANSPORT": "Cartridges in motion\n(blur visible)",
    "DWELL": "Stationary, cartridges\naligned at stations",
    "PROCESSING": "Inspection / testing\n(actuator engaged)",
    "TRANSITION": "Brightness change zone\nt=360-460s",
    "END_STATE": "Post-unload\n3 cartridges (was 4)",
    "HUMAN_INTERACTION": "Human operator\nhandles cartridge\n(t≈391-395s)",
}

# Draw nodes
for name, (x, y) in nodes.items():
    circle = plt.Circle((x, y), 0.7, color=node_colors[name], ec="black", linewidth=1.5, zorder=3)
    ax3.add_patch(circle)
    ax3.text(x, y, name, ha="center", va="center", fontsize=7, fontweight="bold", zorder=4)
    ax3.text(x, y - 1.0, node_descs[name], ha="center", va="top", fontsize=6.5, color="#333333")

# Draw edges (arrows)
edges = [
    ("INIT", "INDEXING", "First conveyor\nindexing detected"),
    ("INDEXING", "TRANSPORT", "Diff drops below\nthreshold"),
    ("TRANSPORT", "DWELL", "Cartridges reach\nstation; motion stops"),
    ("DWELL", "PROCESSING", "Sustained low diff;\nactuator cycle"),
    ("PROCESSING", "INDEXING", "Next indexing\ncycle begins"),
    ("PROCESSING", "TRANSITION", "Enter brightness\nchange zone"),
    ("TRANSITION", "HUMAN_INTERACTION", "Human operator\napproaches conveyor"),
    ("HUMAN_INTERACTION", "TRANSITION", "Operator departs;\nautomation resumes"),
    ("TRANSITION", "END_STATE", "Brightness stabilizes;\ncartridge count 4→3"),
    ("DWELL", "INDEXING", "Next conveyor\nstep triggered"),
    ("TRANSPORT", "INDEXING", "Diff spikes\nagain"),
]

for src, dst, label in edges:
    sx, sy = nodes[src]
    dx, dy = nodes[dst]
    # Arrow
    ax3.annotate("", xy=(dx, dy), xytext=(sx, sy),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2, connectionstyle="arc3,rad=0.15"))
    # Label at midpoint
    mx, my = (sx + dx) / 2, (sy + dy) / 2
    ax3.text(mx + 0.15, my - 0.2, label, ha="center", va="top", fontsize=5.5, color="#666666",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

# Annotation: static memory bank failure point
ax3.text(8, 1.0,
          "STATIC MEMORY BANK FAILURE POINT:\n"
          "At 'DWELL' state, single-frame PatchCore sees 'empty slot' = normal (not yet arrived).\n"
          "At 'END_STATE', single-frame PatchCore sees 'empty slot' = anomaly (cartridge missing).\n"
          "Both look IDENTICAL to a static memory bank. Only temporal context distinguishes them.",
          ha="center", va="center", fontsize=7.5, color="#c0392b",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffeaa7", alpha=0.9, edgecolor="#c0392b", linewidth=1.2))

plt.tight_layout()
fig3.savefig(OUT_FIG_DIR / "case2_state_machine.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"[OK] Figure 3 saved: {OUT_FIG_DIR / 'case2_state_machine.png'}")

# ——— FIGURE 4: Spike suppression comparison ———
# Zoom into t=0-60s to show t=25.7s peak + smoothing effect
mask_zoom = [i for i, t in enumerate(ts) if t <= 60.0]
ts_z = [ts[i] for i in mask_zoom]
dr_z = [diffs_r[i] for i in mask_zoom]
ds_z = [diffs_s[i] for i in mask_zoom]

fig4, ax4 = plt.subplots(figsize=(18, 5))
ax4.plot(ts_z, dr_z, "r-", linewidth=1.2, alpha=0.7, label="Raw frame diff")
ax4.plot(ts_z, ds_z, "b-", linewidth=1.5, alpha=0.9, label=f"Smoothed (median, w={SMOOTH_WINDOW})")
ax4.axhline(y=DIFF_THRESHOLD, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label=f"Threshold ({DIFF_THRESHOLD:.1f})")
ax4.fill_between(ts_z, dr_z, ds_z, alpha=0.2, color="purple", label="Suppressed by smoothing")
ax4.set_xlabel("Time (s)", fontsize=11)
ax4.set_ylabel("Frame Diff", fontsize=11)
ax4.set_title("Case 2 — Spike Suppression: Raw vs Smoothed Frame Diff (t=0-60s zoom)", fontsize=13, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.25)
plt.tight_layout()
fig4.savefig(OUT_FIG_DIR / "case2_spike_suppression.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"[OK] Figure 4 saved: {OUT_FIG_DIR / 'case2_spike_suppression.png'}")

# ——— Summary stats for report ———
print("\n=== Summary for report ===")
print(f"Total frames: {n}")
print(f"Duration: {ts[-1]:.1f}s")
print(f"Human interaction frames (smoothed): {state_counts_smooth.get('HUMAN_INTERACTION', 0)}")
print(f"Indexing events (raw): {state_counts_raw.get('INDEXING', 0)}")
print(f"Indexing events (smoothed): {state_counts_smooth.get('INDEXING', 0)}")
print(f"State transitions (raw): {state_changes_raw}")
print(f"State transitions (smoothed): {state_changes_smooth}")
print(f"Spikes suppressed: {spikes_suppressed}")
print(f"Event frames: {len(event_frames)}")
print("\nDone.")
