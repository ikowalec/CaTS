"""Enumerate depth-three IFCT subgroup memberships and plot coverage versus enrichment."""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pysubgroup as ps

from run_au_subgroups import COMMON, build_selectors, load_cohort


BANDS = {"blue": "400–500 nm", "visible": "500–600 nm", "red_extended": "600–700 nm"}
FAMILIES = {"CO": "CO", "O": "Atomic O", "O2": "O₂", "O+O": "O+O", "combined": "Pooled"}
TARGETS = {"peak": "best_root", "sum": "weighted"}


def enumerate_memberships(data, descriptors, min_support=3):
    selectors = build_selectors(data, descriptors, 4, "thresholds")
    single = [ps.Rule((condition,)) for condition in selectors]
    covers = ps.subgroup_covers(single, data)
    # Bitsets make intersections cheap and serve as exact membership keys.
    viable = {}
    for selector, cover in zip(selectors, covers):
        mask = sum(1 << int(i) for i in np.flatnonzero(cover))
        if min_support <= mask.bit_count() < len(data):
            viable.setdefault((selector.feature, mask), selector)
    masks = [mask for _, mask in viable]
    selectors = list(viable.values())
    features = [selector.feature for selector in selectors]
    representatives = {}
    for depth in (1, 2, 3):
        for indices in combinations(range(len(selectors)), depth):
            if len({features[i] for i in indices}) != depth:
                continue
            mask = masks[indices[0]]
            for i in indices[1:]:
                mask &= masks[i]
            if not min_support <= mask.bit_count() < len(data):
                continue
            if mask not in representatives:
                representatives[mask] = ps.Rule(tuple(selectors[i] for i in indices))
    return [(np.array([i for i in range(len(data)) if mask & (1 << i)]), rule)
            for mask, rule in representatives.items()]


def evaluate(data, values, memberships, lower, upper, fallback):
    baseline = values.mean()
    geometry_ids = data.geometry_id.to_numpy()
    families = data.family.to_numpy()
    rows = []
    for positions, rule in memberships:
        mean = values[positions].mean()
        composition = Counter(families[positions])
        rows.append(dict(support=len(positions), cohort_size=len(data),
                         coverage=len(positions) / len(data), mean=mean, baseline=baseline,
                         lift=mean / baseline if baseline else np.nan,
                         delta=mean - baseline, descriptor_count=len(rule.conditions),
                         rule=str(rule), geometry_ids=";".join(map(str, sorted(geometry_ids[positions]))),
                         family_counts=";".join(f"{key}={value}" for key, value in
                                                sorted(composition.items()))))
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No subgroup memberships met minimum support")
    table = table.sort_values(["support", "mean", "descriptor_count", "rule"],
                              ascending=[False, False, True, True]).reset_index(drop=True)
    best_by_support = table.drop_duplicates("support")
    prior_best = best_by_support["mean"].cummax().shift(fill_value=-np.inf)
    front = best_by_support.index[best_by_support["mean"] > prior_best + 1e-12]
    table["pareto"] = table.index.isin(front)
    table["selected"] = False
    table["selection"] = ""
    pool = table[table.coverage.between(lower, upper)]
    policy = "in-band"
    if pool.empty and fallback:
        pool = table[table.support.eq(table.support.min())]
        policy = "minimum-support fallback"
    enriched = pool[pool.delta.gt(1e-12)]
    if not enriched.empty:
        winner = enriched.sort_values(["mean", "support", "descriptor_count", "rule"],
                                      ascending=[False, False, True, True]).index[0]
        table.loc[winner, ["selected", "selection"]] = [True, policy]
    return table


def plot(output, results, target):
    fig, axes = plt.subplots(5, 3, figsize=(12, 13), sharex=True)
    fig.subplots_adjust(left=.115, right=.97, top=.88, bottom=.075, hspace=.48, wspace=.26)
    objective = "Peak" if target == "peak" else "Summed"
    fig.text(.115, .967, f"{objective} IFCT product  |  Coverage and enrichment",
             fontsize=20, weight="bold")
    fig.text(.115, .941, "Au → adsorbate  •  Rules with up to three descriptors and at least three geometries",
             fontsize=10, color="#364c59")
    legend = [
        Line2D([], [], marker="o", color="#a8b3bb", linestyle="", markersize=5, label="Candidate membership"),
        Line2D([], [], marker="o", color="#145678", linewidth=1.3, markersize=4, label="Pareto front"),
        Line2D([], [], marker="*", color="#b85421", linestyle="", markersize=11, label="Selected subgroup"),
        plt.Rectangle((0, 0), 1, 1, color="#deece8", label="Coverage band"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.105, .929), frameon=False, ncol=4)
    for i, (family, label) in enumerate(FAMILIES.items()):
        for j, (band, wavelength) in enumerate(BANDS.items()):
            ax = axes[i, j]
            table = results[family, band, target]
            low, high = (10, 20) if family == "combined" else (25, 35)
            ax.axvspan(low, high, color="#deece8", zorder=0)
            ax.axhline(1, color="#647887", linestyle=":", linewidth=1)
            ax.scatter(100 * table.coverage, table.lift, s=8, c="#a8b3bb", alpha=.42, edgecolors="none")
            front = table[table.pareto].sort_values("coverage")
            ax.plot(100 * front.coverage, front.lift, "-o", color="#145678", linewidth=1.3, markersize=3)
            selected = table[table.selected]
            fallback = False
            if not selected.empty:
                rule = selected.iloc[0]
                ax.scatter(100 * rule.coverage, rule.lift, marker="*", s=120, color="#b85421",
                           edgecolors="white", linewidth=.5, zorder=5)
                note = f"{rule.support}/{rule.cohort_size}  ·  {rule.lift:.2f}×"
                if rule.selection != "in-band":
                    note += "  ·  fallback"
                    fallback = True
            else:
                note = "No enriched rule in coverage band"
            ax.text(.02 if fallback else .98, .96, note, transform=ax.transAxes,
                    ha="left" if fallback else "right", va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
            ax.set_title(f"{label}  ·  {wavelength}", loc="left", fontsize=10, weight="bold")
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=.15)
    fig.supxlabel("Coverage of eligible cohort (%)", y=.032, fontsize=12)
    fig.supylabel("Enrichment (subgroup mean / cohort mean)", x=.022, fontsize=12)
    fig.savefig(output / f"Au_IFCT_{target}_Pareto.png", dpi=190, facecolor="white")
    plt.close(fig)


def run(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for family in FAMILIES:
        data, specific = load_cohort(dataset, family)
        for band in BANDS:
            sources = {name: f"{band}_au_to_ads_{suffix}" for name, suffix in TARGETS.items()}
            eligible = data[list(sources.values())].notna().all(axis=1)
            eligible &= ~data.negative_energy_line_present.eq(True)
            cohort = data.loc[eligible].reset_index(drop=True)
            descriptors = [name for name in COMMON + specific if cohort[name].nunique() > 1]
            memberships = enumerate_memberships(cohort, descriptors)
            pooled = family == "combined"
            lower, upper = (.1, .2) if pooled else (.25, .35)
            fallback = not pooled and 3 / len(cohort) > upper
            for target, source in sources.items():
                table = evaluate(cohort, 100 * cohort[source].to_numpy(), memberships, lower, upper, fallback)
                results[family, band, target] = table.assign(family=family, band=band, target=target)
    all_rules = pd.concat(results.values(), ignore_index=True)
    all_rules.to_csv(output / "candidate_memberships.csv", index=False)
    all_rules[all_rules.pareto | all_rules.selected].to_csv(output / "pareto_and_selected.csv", index=False)
    all_rules[all_rules.selected].to_csv(output / "selected_subgroups.csv", index=False)
    for target in TARGETS:
        plot(output, results, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.dataset, args.output)
