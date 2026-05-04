import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import OUTPUT_DIRS


DEFAULT_LOGDIR = OUTPUT_DIRS["tensorboard"]
DEFAULT_OUTPUT_DIR = OUTPUT_DIRS["training_reports"]
DEFAULT_SMOOTHING = 12

METRIC_INFO = {
    "reward/raw_queue": ("Queued Vehicles", "Lower is better"),
    "reward/raw_total_wait": ("Total Waiting Time", "Lower is better"),
    "reward/raw_worst_lane_wait": ("Worst-Lane Waiting Time", "Lower is better"),
    "reward/raw_avg_speed": ("Average Speed", "Higher is better"),
    "reward/raw_stopped_ratio": ("Stopped Vehicle Ratio", "Lower is better"),
    "reward/reward_total": ("Step Reward", "Higher is better"),
    "rollout/ep_rew_mean": ("Episode Reward Mean", "Higher is better"),
    "reward/gridlock": ("Gridlock Penalty", "Closer to 0 is better"),
    "reward/raw_gridlock_seconds": ("Gridlock Seconds", "Lower is better"),
    "reward/throughput": ("Throughput Reward", "Higher is better"),
    "reward/raw_collisions": ("Collisions", "Lower is better"),
    "reward/raw_emergency_stops": ("Emergency Stops", "Lower is better"),
    "chaos/incidents_started": ("Chaos Incidents Started", "Context only"),
    "train/explained_variance": ("Explained Variance", "Higher is better"),
    "train/approx_kl": ("Approximate KL", "Stable range: about 0.003-0.03"),
    "train/clip_fraction": ("Clip Fraction", "Stable range: about 0.05-0.25"),
    "train/entropy_loss": ("Entropy Loss", "Should rise gradually toward 0"),
    "train/value_loss": ("Value Loss", "Lower is usually better"),
    "train/policy_gradient_loss": ("Policy Gradient Loss", "Context only"),
}

PLOT_GROUPS = [
    (
        "Traffic Waiting",
        "Waiting-time metrics are large values, so they are plotted separately from queue/speed.",
        ["reward/raw_total_wait", "reward/raw_worst_lane_wait"],
    ),
    (
        "Traffic Queue And Speed",
        "Queued vehicles and stopped ratio should fall; average speed should rise.",
        ["reward/raw_queue", "reward/raw_avg_speed", "reward/raw_stopped_ratio"],
    ),
    (
        "Reward Signals",
        "Step reward should trend upward; penalties should move toward 0.",
        ["reward/reward_total", "reward/gridlock", "reward/throughput"],
    ),
    (
        "Episode Reward",
        "Higher episode reward means the full simulated episode is improving.",
        ["rollout/ep_rew_mean"],
    ),
    (
        "PPO Stability",
        "Healthy PPO has stable KL/clip fraction and useful explained variance.",
        ["train/explained_variance", "train/approx_kl", "train/clip_fraction", "train/entropy_loss"],
    ),
    (
        "Safety",
        "Collisions, emergency stops, and gridlock seconds should stay near zero.",
        ["reward/raw_collisions", "reward/raw_emergency_stops", "reward/raw_gridlock_seconds", "chaos/incidents_started"],
    ),
]


def find_latest_run(logdir=DEFAULT_LOGDIR):
    root = Path(logdir)
    if not root.exists():
        raise FileNotFoundError(f"TensorBoard logdir not found: {logdir}")

    run_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and list(path.glob("events.out.tfevents.*"))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard event runs found in: {logdir}")

    return max(run_dirs, key=lambda path: latest_event_mtime(path))


def latest_event_mtime(run_dir):
    return max(path.stat().st_mtime for path in run_dir.glob("events.out.tfevents.*"))


def load_scalar_table(run_dir):
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalar_tags = sorted(accumulator.Tags().get("scalars", []))
    if not scalar_tags:
        raise ValueError(f"No scalar tags found in TensorBoard run: {run_dir}")

    by_step = {}
    for tag in scalar_tags:
        for event in accumulator.Scalars(tag):
            by_step.setdefault(event.step, {})[tag] = event.value

    if not by_step:
        raise ValueError(f"No scalar values found in TensorBoard run: {run_dir}")

    rows = []
    for step in sorted(by_step):
        row = {"step": step}
        row.update(by_step[step])
        rows.append(row)

    return rows, scalar_tags


def write_metrics_csv(rows, columns, output_csv):
    fieldnames = ["step"] + columns
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_training(rows, columns):
    summary_rows = []
    for column in columns:
        values = [float(row[column]) for row in rows if column in row and row[column] != ""]
        if not values:
            continue
        display_name, interpretation = METRIC_INFO.get(column, (column, "Context only"))
        summary_rows.append(
            {
                "metric": column,
                "name": display_name,
                "interpretation": interpretation,
                "first": values[0],
                "last": values[-1],
                "delta": values[-1] - values[0],
                "best_min": min(values),
                "best_max": max(values),
                "mean": sum(values) / len(values),
                "steps_logged": len(values),
            }
        )
    return summary_rows


def write_summary_csv(summary_rows, output_csv):
    fieldnames = [
        "metric",
        "name",
        "interpretation",
        "first",
        "last",
        "delta",
        "best_min",
        "best_max",
        "mean",
        "steps_logged",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_summary_txt(summary_rows, output_txt, run_dir, last_step):
    key_metrics = [
        "rollout/ep_rew_mean",
        "reward/reward_total",
        "reward/raw_queue",
        "reward/raw_total_wait",
        "reward/raw_worst_lane_wait",
        "reward/raw_avg_speed",
        "reward/raw_collisions",
        "reward/raw_emergency_stops",
        "train/explained_variance",
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
    ]
    by_metric = {row["metric"]: row for row in summary_rows}

    with open(output_txt, "w", encoding="utf-8") as file:
        file.write("TrafficAI Training Report\n")
        file.write("=========================\n\n")
        file.write(f"Run: {run_dir}\n")
        file.write(f"Last logged step: {last_step}\n\n")
        file.write("How to read this report\n")
        file.write("-----------------------\n")
        file.write("Traffic metrics use raw SUMO values. Lower queues/waits are better; higher speed is better.\n")
        file.write("Reward metrics are shaped training signals. Use them for trend/debugging, not as final quality proof.\n")
        file.write("PPO metrics describe training stability. Final model quality should be judged with holdout evaluation.\n\n")
        file.write("Key Metrics\n")
        file.write("-----------\n")
        for metric in key_metrics:
            row = by_metric.get(metric)
            if not row:
                continue
            file.write(
                f"{row['name']} ({metric}) | {row['interpretation']} | "
                f"first={float(row['first']):.4g}, last={float(row['last']):.4g}, "
                f"delta={float(row['delta']):.4g}, min={float(row['best_min']):.4g}, "
                f"max={float(row['best_max']):.4g}\n"
            )


def rolling_mean(points, window):
    smoothed = []
    for index in range(len(points)):
        start = max(0, index - window + 1)
        segment = [value for _, value in points[start : index + 1]]
        smoothed.append((points[index][0], sum(segment) / len(segment)))
    return smoothed


def metric_label(tag):
    display_name, interpretation = METRIC_INFO.get(tag, (tag, "Context only"))
    return f"{display_name} ({interpretation})"


def plot_training_report(rows, output_png, smoothing=DEFAULT_SMOOTHING):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(PLOT_GROUPS), 1, figsize=(17, 22), sharex=True)

    for ax, (group_name, description, tags) in zip(axes, PLOT_GROUPS):
        plotted = False
        for tag in tags:
            points = [(row["step"], float(row[tag])) for row in rows if tag in row and row[tag] != ""]
            if not points:
                continue
            smoothed = rolling_mean(points, smoothing)
            ax.plot(
                [step for step, _ in smoothed],
                [value for _, value in smoothed],
                label=metric_label(tag),
                linewidth=2,
            )
            plotted = True

        ax.set_title(f"{group_name.upper()} - {description}", fontsize=12, fontweight="bold")
        ax.set_ylabel("value")
        ax.grid(True, linestyle="--", alpha=0.45)
        if plotted:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No matching metrics", transform=ax.transAxes, ha="center", va="center")

    axes[-1].set_xlabel("training timesteps")
    fig.suptitle("TrafficAI Training Metrics Report", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def generate_training_report(logdir=DEFAULT_LOGDIR, output_dir=DEFAULT_OUTPUT_DIR, run_name=None, smoothing=DEFAULT_SMOOTHING):
    run_dir = Path(logdir) / run_name if run_name else find_latest_run(logdir)
    if not run_dir.exists():
        raise FileNotFoundError(f"TensorBoard run not found: {run_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_run_name = run_dir.name.replace(" ", "_")

    rows, columns = load_scalar_table(run_dir)
    summary_rows = summarize_training(rows, columns)
    last_step = int(rows[-1]["step"])

    metrics_csv = output_path / f"{safe_run_name}_metrics.csv"
    summary_csv = output_path / f"{safe_run_name}_summary.csv"
    summary_txt = output_path / f"{safe_run_name}_summary.txt"
    report_png = output_path / f"{safe_run_name}_report.png"

    write_metrics_csv(rows, columns, metrics_csv)
    write_summary_csv(summary_rows, summary_csv)
    write_summary_txt(summary_rows, summary_txt, run_dir, last_step)

    try:
        plot_training_report(rows, report_png, smoothing=smoothing)
    except ImportError:
        report_png = None

    return {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "summary_csv": str(summary_csv),
        "summary_txt": str(summary_txt),
        "report_png": str(report_png) if report_png else None,
        "last_step": last_step,
    }


def main():
    parser = argparse.ArgumentParser(description="Build an English training report from TensorBoard scalars.")
    parser.add_argument("--logdir", default=DEFAULT_LOGDIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run", default=None, help="Run directory name inside logdir. Defaults to latest run.")
    parser.add_argument("--smoothing", type=int, default=DEFAULT_SMOOTHING)
    args = parser.parse_args()

    result = generate_training_report(args.logdir, args.output_dir, args.run, smoothing=args.smoothing)
    print(f"Run: {result['run_dir']}")
    print(f"Last step: {result['last_step']}")
    print(f"Metrics CSV: {result['metrics_csv']}")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Summary TXT: {result['summary_txt']}")
    if result["report_png"]:
        print(f"Report PNG: {result['report_png']}")
    else:
        print("Report PNG: skipped because matplotlib is not installed")


if __name__ == "__main__":
    main()
