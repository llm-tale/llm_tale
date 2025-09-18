import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def set_plot_style():
    sns.set_theme(style="whitegrid", font_scale=1)
    plt.rcParams.update(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "font.size": 8,
            "grid.linewidth": 0.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.pad": -2.0,
            "ytick.major.pad": -2.0,
            "lines.linewidth": 1.3,
            "axes.xmargin": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": True,
            "font.family": "Helvetica",
        }
    )


def rolling_mean_numpy(data, window_size):
    half_window = window_size // 2
    moving_avg = np.empty(len(data))

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        moving_avg[i] = np.mean(data[start:end])

    return moving_avg


def extract_scalars(file, window_size, scalar_name="eval/success"):
    ea = event_accumulator.EventAccumulator(file, size_guidance={"scalars": 0})
    ea.Reload()
    scalar_events = ea.Scalars(scalar_name)
    steps = np.array([event.step for event in scalar_events])
    values = np.array([event.value for event in scalar_events])
    values = rolling_mean_numpy(values, window_size)
    return steps, values


def plot_scalars_tb(ax, tb_file, color, label, window_size=10, alpha=0.2, line_type=":", scalar_name="eval/success"):
    # Read the tensorboard file
    all_success_rates = []
    min_len = float("inf")
    for file in tb_file:
        steps, values = extract_scalars(file, window_size, scalar_name)
        min_len = min(min_len, len(values))
        all_success_rates.append(values)

    plot_scalars(ax, steps[:min_len], all_success_rates, color, label, window_size, alpha, line_type)


def plot_scalars(ax, steps, all_success_rates, color, label, window_size=5, alpha=0.2, line_type=":"):
    min_len = min(len(sr) for sr in all_success_rates)

    # Trim all to min length
    all_success_rates = np.array([sr[:min_len] for sr in all_success_rates])
    # rolling mean
    all_success_rates = np.array([rolling_mean_numpy(sr, window_size) for sr in all_success_rates])
    mean_success = np.mean(all_success_rates, axis=0)
    std_success = np.std(all_success_rates, axis=0)
    steps = np.array(steps[:min_len])

    steps_plot = steps
    mean_plot = mean_success
    std_plot = std_success
    ax.plot(steps_plot, mean_plot, linestyle=line_type, color=color, label=label)
    ax.fill_between(steps_plot, mean_plot - std_plot, mean_plot + std_plot, color=color, alpha=alpha)
