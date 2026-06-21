"""Generate P1 figures for Case 2 final report.
P1-1: Human Intervention Contact Sheet
P1-2: Brain-Inspired 3-Layer Memory Framework
P1-3: Five-Level Argument Diagram
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import numpy as np
import os

ROOT = r"E:/leinaozuoye"
OUT = os.path.join(ROOT, "report", "final_assets")
os.makedirs(OUT, exist_ok=True)

# CJK font setup
for f in fm.fontManager.ttflist:
    if "Microsoft YaHei" in f.name:
        plt.rcParams["font.family"] = f.name
        print(f"Using font: {f.name}")
        break
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# P1-1: Case 2 Human Intervention Contact Sheet (6 frames)
# ============================================================
frame_dir = os.path.join(ROOT, "data", "workstation", "review_frames", "20241031_222226")
existing = sorted(os.listdir(frame_dir))

strip_frames = []
# Before context: t=197.6s
pre_fn = [f for f in existing if "197.65s" in f]
if pre_fn:
    strip_frames.append((pre_fn[0], "t=197.6s (before)", "Normal automated op"))
# 4 human window frames
for fn_tag, t_label, desc in [
    ("011624_389.40s", "t=389.4s", "Human behind guard\n(blue cleanroom gown)"),
    ("011680_391.30s", "t=391.3s", "Hand near conveyor\n(still observing)"),
    ("011740_393.30s", "t=393.3s", "HAND HOLDING\ncartridge into rail"),
    ("011800_395.30s", "t=395.3s", "Human GONE\n(4 cartridges seated)"),
]:
    fn = [f for f in existing if fn_tag in f and "human_intervention" in f]
    if fn:
        strip_frames.append((fn[0], t_label, desc))
# After context: t=434.8s
post_fn = [f for f in existing if "434.82s" in f]
if post_fn:
    strip_frames.append((post_fn[0], "t=434.8s (after)", "3 cartridges, end"))

n = len(strip_frames)
fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 5.5))
if n == 1:
    axes = [axes]
fig.patch.set_facecolor("#fafafa")

for i, (fn, time_label, desc) in enumerate(strip_frames):
    fp = os.path.join(frame_dir, fn)
    img = Image.open(fp).convert("RGB")
    img = img.resize((640, 360), Image.LANCZOS)
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f"{time_label}\n{desc}", fontsize=9, fontweight="bold", pad=6, color="#222")
    ax.set_xticks([])
    ax.set_yticks([])
    is_human = any(kw in desc.upper() for kw in ["HUMAN", "HAND", "HOLDING"])
    color = "#ff4444" if is_human else "#cccccc"
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5 if is_human else 1.0)

fig.suptitle("Case 2 Human Intervention Contact Sheet",
             fontsize=14, fontweight="bold", y=0.98)
fig.text(0.5, 0.01,
         "Red borders = HUMAN_INTERACTION window (t=389-397s). 4-second window captured by 2-3 frames at 2s sampling.",
         ha="center", fontsize=9, color="#666")
plt.tight_layout(rect=[0, 0.04, 1, 0.93])
out1 = os.path.join(OUT, "fig_case2_human_intervention_contact_sheet.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"P1-1: {out1} ({os.path.getsize(out1)} bytes)")

# ============================================================
# P1-2: Brain-Inspired 3-Layer Memory Framework
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.set_facecolor("#fafafa")
ax.axis("off")

# Colors
L1_COLOR = "#d1ecf1"
L2_COLOR = "#fff3cd"
L3_COLOR = "#f8d7da"

# Layer backgrounds
ax.add_patch(plt.Rectangle((1, 8.5), 14, 2.8, facecolor=L1_COLOR, edgecolor="#17a2b8", linewidth=2, alpha=0.3, zorder=0))
ax.text(8, 11.0, "Long-Term Memory  (stabilizing experience)", ha="center", va="center", fontsize=14, fontweight="bold", color="#0c5460")

lt_boxes = [
    (2, 9.0, "MVTec AD\nStatic Memory Bank\n(PaDiM/PatchCore/SPADE/DINO)"),
    (5.5, 9.0, "Cycle Statistics\nindexing period, dwell duration\nbrightness normal range"),
    (9.5, 9.0, "Pattern Library\nHUMAN_INTERVENTION freq\n4-to-3 reversibility"),
    (12.5, 9.0, "Versioned Bank\ntimestamped updates\nrollback capability"),
]
for x, y, text in lt_boxes:
    ax.text(x, y, text, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#17a2b8", linewidth=1.2))

ax.add_patch(plt.Rectangle((1, 4.5), 14, 2.8, facecolor=L2_COLOR, edgecolor="#ffc107", linewidth=2, alpha=0.3, zorder=0))
ax.text(8, 7.0, "Short-Term Memory  (working context)", ha="center", va="center", fontsize=14, fontweight="bold", color="#856404")

st_boxes = [
    (2.5, 5.3, "Sliding Window\nmedian filter (5fr=10s)\n94-to-32 transitions (-66%)"),
    (6.5, 5.3, "Frame-Pair Consistency\n'had cartridge -> should have'\ntemporal coherence check"),
    (10.5, 5.3, "Motion Blur Suppression\nINDEXING converted to TRANSPORT\nspike filtering"),
    (13.5, 5.3, "Empty-Slot Disambiguation\n'normal not-arrived-yet' vs\n'abnormal missing'"),
]
for x, y, text in st_boxes:
    ax.text(x, y, text, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ffc107", linewidth=1.2))

ax.add_patch(plt.Rectangle((1, 0.5), 14, 2.8, facecolor=L3_COLOR, edgecolor="#dc3545", linewidth=2, alpha=0.3, zorder=0))
ax.text(8, 3.0, "Feedback Gating  (anti-contamination consolidation)", ha="center", va="center", fontsize=14, fontweight="bold", color="#721c24")

fg_boxes = [
    (3, 1.3, "Human Confirmation\nHUMAN_INTERACTION\n-> normal replenishment?\n-> or anomaly?"),
    (8, 1.3, "Anti-Contamination Update\nconfirmed normal -> LT memory\nconfirmed anomaly -> anomaly case library"),
    (13, 1.3, "Acquisition Quality Gate\nbrightness/focus/vibration/occlusion\nunreliable frames -> skip inference"),
]
for x, y, text in fg_boxes:
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#dc3545", linewidth=1.2))

# Arrows
ax.annotate("", xy=(8, 7.5), xytext=(8, 8.3), arrowprops=dict(arrowstyle="<->", color="#17a2b8", lw=2.5))
ax.text(10.5, 7.95, "context query\n(current vs pattern)", fontsize=8, color="#555")

ax.annotate("", xy=(8, 3.5), xytext=(8, 4.3), arrowprops=dict(arrowstyle="<->", color="#ffc107", lw=2.5))
ax.text(10.5, 3.95, "uncertain events\n(new state / score > thresh)", fontsize=8, color="#555")

ax.annotate("", xy=(2, 8.2), xytext=(2.5, 3.5), arrowprops=dict(arrowstyle="->", color="#28a745", lw=2, connectionstyle="arc3,rad=-0.3"))
ax.text(0.1, 5.8, "normal\n-> consolidate", fontsize=7, color="#28a745", fontweight="bold")

ax.annotate("", xy=(14.5, 8.2), xytext=(13.5, 3.5), arrowprops=dict(arrowstyle="->", color="#dc3545", lw=2, connectionstyle="arc3,rad=0.3"))
ax.text(14.6, 5.8, "anomaly\n-> case lib", fontsize=7, color="#dc3545", fontweight="bold")

# Failure modes
ax.text(3, 9.8, "Case 4: silent failure\n(signal < noise)", fontsize=8, color="#dc3545", bbox=dict(facecolor="white", edgecolor="#dc3545", boxstyle="round", alpha=0.9))
ax.text(12, 5.8, "Case 2: noisy failure\n(legit changes = high score)", fontsize=8, color="#dc3545", bbox=dict(facecolor="white", edgecolor="#dc3545", boxstyle="round", alpha=0.9))

fig.suptitle("Brain-Inspired Three-Layer Normality Memory Framework", fontsize=15, fontweight="bold", y=0.99)

out2 = os.path.join(OUT, "fig_brain_inspired_three_layer_memory.png")
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"P1-2: {out2} ({os.path.getsize(out2)} bytes)")

# ============================================================
# P1-3: Five-Level Argument Diagram (Case 2)
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(20, 5.5))
fig.patch.set_facecolor("#fafafa")

levels = [
    ("1: Pixel-Object Decoupling", "#e74c3c",
     "Same pixel position\ndifferent times =\ndifferent objects.",
     "t=0s: empty slot\n(normal, not-yet-arrived)\nt=79s: moving cartridge\n(motion blur passing)\nt=158s: stationary cartridge\n(inspection position)"),
    ("2: Dual Semantics of Empty", "#e67e22",
     "An empty slot can be\nnormal (not arrived yet)\nOR\nabnormal (missing).",
     "Single-frame sees \"empty\"\nand cannot distinguish:\n\"Empty slot at t=0s\" (ok)\nvs \"Empty slot at t=435s\"\n(end-of-run or missing?)"),
    ("3: Human Hand = Extreme\nAnomaly but Legitimate", "#f39c12",
     "t=393s: human hand\nholding cartridge.",
     "PatchCore would assign\nMAXIMUM anomaly score.\nBut this is NORMAL\nreplenishment operation.\nOnly temporal context\n(\"hand appeared 2s ago,\ndisappeared 2s later\")\ncan suppress this."),
    ("4: Normality = a SET\nof Temporal States", "#27ae60",
     "Cartridge count:\n4 -> 3 -> 4 -> 3\nNOT monotonically 4->3.",
     "Reversibility (4<->3)\nis the strongest evidence\nthat normality is not\na single image distribution\nbut a state-transition graph."),
    ("5: State Machine = Free\nDetection Signal", "#2980b9",
     "Unexpected state\ntransition is itself\nan anomaly signal.",
     "If state machine says:\n\"Expected: DWELL or TRANSPORT\"\n\"Observed: 3 cartridges\nat mid-batch\"\n-> anomaly, zero training data\nneeded for this rule."),
]

for i, (title, color, claim, evidence) in enumerate(levels):
    ax = axes[i]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("#fcfcfc")

    # Title bar
    ax.add_patch(plt.Rectangle((0, 8.0), 10, 2.0, facecolor=color, edgecolor="#333", linewidth=1.5, alpha=0.85))
    ax.text(5, 9.0, title, ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    # Claim
    ax.text(5, 7.0, claim, ha="center", va="center", fontsize=9, color="#222", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9e6", edgecolor=color, linewidth=1))

    # Evidence
    ax.text(5, 3.5, evidence, ha="center", va="center", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", linewidth=1))

    # Arrow between levels
    if i < 4:
        ax.annotate("", xy=(10.5, 5), xytext=(10, 5),
                    arrowprops=dict(arrowstyle="->", color="#999", lw=1.5),
                    xycoords="axes fraction")

fig.suptitle("Case 2: Why Single-Frame Detection Fails — Five-Level Argument",
             fontsize=14, fontweight="bold", y=0.98)
fig.text(0.5, 0.01,
         "Each level builds on the previous, forming a cumulative argument for temporal-state-aware anomaly detection.",
         ha="center", fontsize=9, color="#666")
plt.tight_layout(rect=[0, 0.04, 1, 0.93])

out3 = os.path.join(OUT, "fig_case2_five_level_argument.png")
fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"P1-3: {out3} ({os.path.getsize(out3)} bytes)")

print("\n=== All P1 figures complete ===")
