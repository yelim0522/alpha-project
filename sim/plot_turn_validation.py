"""Render audited turn-validation CSVs, without running any simulations.

Run from any directory: python sim/plot_turn_validation.py
Install requirements-figures.txt in an isolated environment for rendering.
Validation/statistics helpers intentionally use only the standard library.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
POLICIES = ("ctho-approx", "pallas-approx", "coordinated", "turn-boundary")
LABELS = ("ctHO\napprox.", "Pallas\napprox.", "Coordinated", "Turn\nboundary")
COLORS = ("#7B8794", "#CC79A7", "#0072B2", "#D55E00")
SCENARIOS = {
    "fixed": "Fixed length / fixed gap",
    "variable": "Synthetic length / mixed gap",
    "azure-fixed": "Azure length / fixed gap",
    "azure-mixture": "Azure length / mixed gap",
    "azure-decode240": "+ decode cap: 240 token/s*",
    "azure-dt025": "+ time step: 0.25 s",
    "immediate": "Fixed length / zero gap",
}
METRICS = ("response_wait_mean_s", "response_wait_p99_s",
           "response_excess_p99_s", "responses_completed")


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_rows(rows, seeds):
    """Require a complete paired design and finite, nonnegative metrics."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid seed list")
    expected = {(s, p, int(seed)) for s in SCENARIOS for p in POLICIES for seed in seeds}
    indexed = {}
    for row in rows:
        key = (row["scenario"], row["policy"], int(row["seed"]))
        if key in indexed:
            raise ValueError("duplicate scenario/policy/seed: %r" % (key,))
        numeric = {k: float(v) for k, v in row.items()
                   if k not in ("scenario", "policy", "seed")}
        if not all(math.isfinite(v) and v >= 0 for v in numeric.values()):
            raise ValueError("nonfinite or negative metric: %r" % (key,))
        if not set(METRICS).issubset(numeric) or numeric["responses_completed"] <= 0:
            raise ValueError("missing metrics or no completed responses")
        if not math.isclose(numeric["observed_wait_s"],
                            numeric["completed_wait_s"] + numeric["censored_wait_s"],
                            rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("wait accounting mismatch: %r" % (key,))
        indexed[key] = numeric
    if set(indexed) != expected:
        raise ValueError("incomplete or unexpected scenario/policy/seed design")
    return indexed


def values(indexed, seeds, scenario, policy, metric):
    return [indexed[scenario, policy, seed][metric] for seed in seeds]


def mean_sd(series):
    return statistics.mean(series), statistics.stdev(series) if len(series) > 1 else 0.0


def validate_summary(indexed, summaries, seeds):
    seen = set()
    for row in summaries:
        pair = row["scenario"], row["policy"]
        if pair in seen or pair not in {(s, p) for s in SCENARIOS for p in POLICIES}:
            raise ValueError("duplicate or unknown summary row")
        seen.add(pair)
        if int(row["seeds"]) != len(seeds):
            raise ValueError("summary seed count mismatch")
        for metric in indexed[pair + (seeds[0],)]:
            for suffix, calculated in zip(("_mean", "_sd"), mean_sd(
                    values(indexed, seeds, *pair, metric))):
                if not math.isclose(float(row[metric + suffix]), calculated,
                                    rel_tol=1e-9, abs_tol=1e-8):
                    raise ValueError("summary mismatch: %r %s" % (pair, metric + suffix))
    if len(seen) != len(SCENARIOS) * len(POLICIES):
        raise ValueError("incomplete summary")


def paired_effects(indexed, seeds, scenario):
    c = [indexed[scenario, "coordinated", s] for s in seeds]
    t = [indexed[scenario, "turn-boundary", s] for s in seeds]
    return {
        "wait_p99_difference_s": [b["response_wait_p99_s"] - a["response_wait_p99_s"]
                                  for a, b in zip(c, t)],
        "completion_ratio": [b["responses_completed"] / a["responses_completed"]
                             for a, b in zip(c, t)],
    }


def save_figure(fig, out, stem):
    for extension in ("png", "pdf", "svg"):
        metadata = ({"CreationDate": None, "ModDate": None} if extension == "pdf"
                    else {"Date": None} if extension == "svg" else {})
        fig.savefig(out / (stem + "." + extension), dpi=220,
                    facecolor="white", metadata=metadata)


def render(indexed, seeds, configs, out):
    # Keep imports optional so the simulator's unittest discovery needs no deps.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import StrMethodFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42,
                         "svg.hashsalt": "alpha-turn-validation-v11"})
    cfg = configs["azure-mixture"]
    duration = cfg["steps"] * cfg["dt"]
    context = f"{cfg['users']} mobile users | {duration:g} s | {len(seeds)} paired seeds"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.85, bottom=0.16,
                        hspace=0.65, wspace=0.28)
    fig.suptitle("Turn-boundary trade-offs in the closed-loop model", y=0.98, fontsize=15)
    fig.text(0.5, 0.927, "Azure output lengths + synthetic mixed think times\n" + context,
             ha="center", fontsize=10)
    titles = ("(a) Mean generation wait", "(b) Tail generation wait",
              "(c) Tail excess response time", "(d) Completed responses")
    ylabels = ("Mean wait (s)", "Response-wait p99 (s)",
               "Excess-time p99 (s)", "Responses / observation window")
    for ax, metric, title, ylabel in zip(axes.flat, METRICS, titles, ylabels):
        series = [values(indexed, seeds, "azure-mixture", p, metric) for p in POLICIES]
        means = [statistics.mean(v) for v in series]
        upper = max(max(v) for v in series)
        ax.bar(range(4), means, width=0.60, color=COLORS, alpha=0.88, zorder=2)
        for i, sample in enumerate(series):
            for j, value in enumerate(sample):
                offset = 0.22 * (j / max(1, len(seeds) - 1) - 0.5)
                ax.scatter(i + offset, value, s=24, facecolor="white",
                           edgecolor="#222222", linewidth=0.8, zorder=4)
            label = f"{means[i]:,.0f}" if metric == "responses_completed" else f"{means[i]:.3f}"
            ax.text(i, max(sample) + upper * 0.055, label, ha="center", fontsize=10)
        ax.set(xticks=range(4), xticklabels=LABELS, ylabel=ylabel,
               title=title, ylim=(0, upper * 1.23))
        ax.grid(axis="y", alpha=0.22, zorder=0)
        if metric == "responses_completed":
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    fig.text(0.085, 0.035,
             "Bars: seed means; open dots: individual seeds. p99: mean of seed-specific percentiles.\n"
             "Latency metrics cover fully observed completed responses; completion/censoring differs by policy.\n"
             "Excess time subtracts output length / base decode rate; it is not first-token latency.",
             fontsize=9, color="#444444")
    save_figure(fig, out, "turn_boundary_main")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.1), sharey=True)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.79, bottom=0.22, wspace=0.20)
    fig.suptitle("Sensitivity across seven conversation conditions", y=0.98, fontsize=15)
    fig.text(0.5, 0.925, context, ha="center", fontsize=10)
    for y, scenario in enumerate(SCENARIOS):
        if scenario == "azure-mixture":
            for ax in axes:
                ax.axhspan(y - 0.48, y + 0.48, color="#0072B2", alpha=0.055)
        for policy, color, offset, marker in (("coordinated", COLORS[2], -0.14, "o"),
                                               ("turn-boundary", COLORS[3], 0.14, "D")):
            sample = values(indexed, seeds, scenario, policy, "response_wait_p99_s")
            mean, sd = mean_sd(sample)
            axes[0].scatter(sample, [y + offset] * len(seeds), color=color,
                            s=17, alpha=0.35, zorder=3)
            axes[0].errorbar(mean, y + offset, xerr=sd, fmt=marker, color=color,
                             markersize=5, capsize=3, zorder=4)
        ratios = paired_effects(indexed, seeds, scenario)["completion_ratio"]
        mean, sd = mean_sd(ratios)
        axes[1].scatter(ratios, [y] * len(seeds), color=COLORS[3], s=22, alpha=0.35)
        axes[1].errorbar(mean, y, xerr=sd, fmt="D", color=COLORS[3], markersize=5, capsize=3)
    axes[0].set(yticks=range(len(SCENARIOS)), yticklabels=list(SCENARIOS.values()),
                ylim=(len(SCENARIOS) - 0.5, -0.5), xlim=(0, None),
                xlabel="Response-wait p99 (s) — lower is better", title="(a) Tail generation wait")
    axes[1].set(xlim=(0.78, 1.02), xlabel="Completed responses\nTurn boundary / coordinated",
                title="(b) Completion ratio")
    axes[1].axvline(1, color="#666666", linestyle="--", linewidth=1)
    for ax in axes:
        ax.grid(axis="x", alpha=0.22)
    fig.legend(handles=[Line2D([], [], color=COLORS[2], marker="o", linestyle="", label="Coordinated"),
                        Line2D([], [], color=COLORS[3], marker="D", linestyle="", label="Turn boundary")],
               loc="upper center", bbox_to_anchor=(0.64, 0.9), ncol=2, frameon=False)
    fig.text(0.03, 0.055,
             "Solid markers / whiskers: seed mean ± sample SD (not confidence intervals); faint dots: individual seeds.\n"
             "Completion ratios are paired by seed; dashed line = equal completions. Shading = main condition.\n"
             "* Uncalibrated per-server decode budget, separate from prefill. Time-step change also changes mobility/prediction grids.",
             fontsize=9, color="#444444")
    save_figure(fig, out, "turn_boundary_robustness")
    plt.close(fig)
    return matplotlib.__version__


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prefix", type=Path, default=ROOT / "turn_validation_v09")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "paper/figures")
    args = parser.parse_args()
    files = {name: Path(str(args.input_prefix) + suffix) for name, suffix in (
        ("seeds", "_seeds.csv"), ("summary", "_summary.csv"), ("experiment", "_manifest.json"))}
    manifest = json.loads(files["experiment"].read_text())
    seeds = manifest["seeds"]
    indexed = validate_rows(read_rows(files["seeds"]), seeds)
    validate_summary(indexed, read_rows(files["summary"]), seeds)
    # Verify the implementation recorded at experiment time, not just its Git HEAD.
    for name, digest in manifest["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("experiment source has changed: " + name)
    workload = ROOT / "data/azure_output_histogram.json"
    if sha256(workload) != manifest["workload_sha256"]:
        raise ValueError("workload hash mismatch")
    if set(manifest["configs"]) != set(SCENARIOS):
        raise ValueError("scenario configuration mismatch")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    version = render(indexed, seeds, manifest["configs"], out)
    records = []
    for scenario in SCENARIOS:
        for metric, series in paired_effects(indexed, seeds, scenario).items():
            for seed, value in zip(seeds, series):
                records.append(dict(scenario=scenario, metric=metric, seed=seed, value=value))
    with (out / "turn_boundary_paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scenario", "metric", "seed", "value"))
        writer.writeheader()
        writer.writerows(records)
    audit = {
        "inputs_sha256": {path.name: sha256(path) for path in files.values()},
        "plot_source_sha256": sha256(Path(__file__)), "matplotlib": version,
        "seeds": seeds, "rows_checked": len(indexed),
        "statistics": "seed mean; sample SD, not CI; mean of seed-specific p99, not pooled p99",
        "ratios": "figure: mean of paired per-seed ratios; draft percentage: ratio of seed means",
        "source_and_workload_hashes_verified": True,
    }
    (out / "turn_boundary_plot_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Validated {len(indexed)} rows and all summary cells; wrote 2 figures (PNG/PDF/SVG) to {out}")


if __name__ == "__main__":
    main()
