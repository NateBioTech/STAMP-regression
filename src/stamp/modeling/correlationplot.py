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
        diag_limit = max_val * 1.05

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



def plot_all_fold_correlations(output_dir: Path, n_splits: int) -> None:
    """
    Combine correlation plots from all splits into one single figure.

    Args:
        output_dir: The directory where split folders and plots are saved.
        n_splits: Number of CV splits.
    """
    n_cols = min(n_splits, 5)  # Maximum 5 plots in a row
    n_rows = (n_splits + n_cols - 1) // n_cols  # Compute rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of 1 row or 1 col

    for i in range(n_splits):
        plot_path = output_dir / f"split-{i}" / "correlation_plot.png"
        if plot_path.exists():
            img = plt.imread(plot_path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Fold {i}", fontsize=14)
        else:
            axes[i].text(0.5, 0.5, f"No plot for Fold {i}", ha='center', va='center')
            axes[i].axis("off")

    # Hide unused axes
    for j in range(n_splits, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    combined_path = output_dir / "all_folds_correlation.png"
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"[plot_all_fold_correlations] Saved → {combined_path}")
