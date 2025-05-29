# stamp/modeling/correlationplot.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


def plot_correlation(
    y_true,
    y_pred,
    output_dir: Path | str,
    *,
    diag_limit: float | None = None,  # CHANGED: Optional limit for 1:1 reference line; if None, computed automatically
) -> None:
    """
    Creates and saves a correlation scatter plot of predicted vs. actual values.

    The plot includes:
    - A scatter of predicted vs. actual points
    - A regression line with 95% confidence interval
    - A dashed 1:1 reference line (gray) from (0,0) to (diag_limit, diag_limit)

    Args:
        y_true: Ground truth values (list, array, or tensor-like)
        y_pred: Predicted values (same length as y_true)
        output_dir: Directory where the figure will be saved
        diag_limit: Upper limit for the 1:1 reference line. If None, it is set to
                    105% of the max(y_true, y_pred)

    Saves:
        correlation_plot.png in the output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})

    # Compute correlation statistics
    pearson_r, p_val = scipy.stats.pearsonr(y_true, y_pred)
    slope, intercept, r_val, _, _ = scipy.stats.linregress(y_true, y_pred)
    r2 = r_val ** 2

    # Create seaborn scatterplot with regression line
    g = sns.lmplot(
        x="Actual",
        y="Predicted",
        data=data,
        scatter_kws=dict(color="#1C1C1C", alpha=0.75, s=50, edgecolor="black"),
        line_kws=dict(color="#0E3CBD", linewidth=2),
        ci=95,
        height=6,
        aspect=1.2,
    )

    # Compute diagonal limit if not provided
    if diag_limit is None:
        max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
        diag_limit = max_val * 1.05  # CHANGED: Add 5% margin for better visual spacing

    # Plot 1:1 dashed reference line
    g.ax.plot([0, diag_limit], [0, diag_limit], ls="--", lw=1.5, color="gray")

    # Add labels and title with correlation stats
    g.set_axis_labels("Actual", "Predicted")
    g.ax.set_title(
        f"Correlation: R²={r2:.3f}, Pearson={pearson_r:.3f}, p={p_val:.1e}",
        pad=20,
    )
    g.fig.tight_layout()

    # Save plot to file
    save_path = output_dir / "correlation_plot.png"
    g.savefig(save_path, dpi=300)
    plt.close(g.fig)
    print(f"[plot_correlation] Saved → {save_path}")
