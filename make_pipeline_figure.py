"""Draw the methodology pipeline figure used in the report/paper (results/figures/pipeline.png)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1] if len(sys.argv) > 1 else "results/figures/pipeline.png"
fig, ax = plt.subplots(figsize=(3.4, 3.7)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 11)
steps = ["Phishing URL dataset\n(579,920 records)", "Preprocessing\n(cleaning, encoding, split)", "Feature construction\n(lexical + DNS + WHOIS)",
         "Classifier training\n(multiple ML models)", "Evaluation\n(Acc, Prec, Rec, F1, CM)", "XAI analysis\n(global + local explanations)"]
h, gap, y = 1.35, 0.52, 11 - 1.35
for i, s in enumerate(steps):
    ax.add_patch(FancyBboxPatch((1, y), 8, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#f2f2f2", ec="black", lw=0.8))
    ax.text(5, y + h / 2, s, ha="center", va="center", fontsize=7)
    if i < len(steps) - 1:
        ax.annotate("", xy=(5, y - gap + 0.02), xytext=(5, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))
    y -= h + gap
fig.savefig(out, dpi=250, bbox_inches="tight", pad_inches=0.03)
