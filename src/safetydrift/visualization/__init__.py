"""Publication-quality figures and tables for the paper."""

from safetydrift.visualization.figures import (
    plot_ablation_bars,
    plot_absorption_curves,
    plot_early_warning_distribution,
    plot_learning_curve,
    plot_monitor_roc,
    plot_per_category_absorption,
    plot_transition_heatmap,
)
from safetydrift.visualization.tables import (
    generate_comparison_table,
    generate_per_category_table,
    generate_transition_table,
)
