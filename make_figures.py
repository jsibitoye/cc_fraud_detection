import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

# Define pipeline steps with line-broken text for readability
steps = [
    "Data\nIngestion\n(CSV /\nStream)",
    "Pre-\nprocessing\n(OHE,\nScaling)",
    "Train /\nTest\nSplit\n(Stratified)",
    "Rebalancing\n(SMOTE /\nThreshold)",
    "Model\nTraining\n(RF /\nXGB /\nLGBM /\nCatBoost)",
    "Calibration\n(Threshold /\nTuning)",
    "Evaluation\n(ROC-AUC,\nPR-AUC,\nConfusion\nMatrix)",
    "Deploy &\nMonitor\n(Drift /\nAlerts)"
]

# Plot settings
fig, ax = plt.subplots(figsize=(16, 4))
ax.set_xlim(0, len(steps) * 3)
ax.set_ylim(0, 4)
ax.axis("off")

# Draw boxes and arrows
for i, step in enumerate(steps):
    # Draw rectangle
    rect = Rectangle((i * 3, 1), 2.5, 2, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(rect)

    # Add text inside rectangle (centered)
    ax.text(i * 3 + 1.25, 2, step, ha="center", va="center", fontsize=12, wrap=True)

    # Draw arrow to the next box
    if i < len(steps) - 1:
        arrow = FancyArrow(i * 3 + 2.5, 2, 0.5, 0, width=0.05, head_width=0.3, head_length=0.3, length_includes_head=True, color="black")
        ax.add_patch(arrow)

plt.tight_layout()
plt.show()
