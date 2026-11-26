"""Run comparable brightness and directional IFCT subgroup searches."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pysubgroup as ps

from prepare_au_subgroups import WINDOWS

COMMON = [
    "au_n", "system_charge", "spin_multiplicity", "au_rg_a", "au_span_a",
    "au_axis_short_long", "au_axis_middle_long", "au_asphericity",
    "au_gcn_mean", "au_gcn_std", "au_fraction_cn_le6", "ads_mulliken_e",
]
O_SITE = ["O_site", "O_au_contacts", "O_au_min_a", "O_au_cn_mean", "O_au_gcn_min",
          "O_au_gcn_mean", "O_au_gcn_spread"]
O_PAIR = ["oo_a", "o_proximal_au_a", "o_distal_au_a", "o_contact_asymmetry_a",
          "o_contact_mode", "o_shared_au", "o_mulliken_asymmetry_e",
          "o_proximal_au_contacts", "o_distal_au_contacts",
          "o_proximal_au_gcn_mean", "o_distal_au_gcn_mean"]
SPECIFIC = {
    "CO": ["C_site", "C_au_contacts", "C_au_min_a", "C_au_cn_mean",
           "C_au_gcn_min", "C_au_gcn_mean", "C_au_gcn_spread", "co_a"],
    "O": O_SITE,
    "O2": O_SITE + O_PAIR,
    "O+O": O_SITE + O_PAIR,
}
CONTACT_FEATURES = ("site", "au_contacts", "au_min_a", "au_cn_mean",
                    "au_gcn_min", "au_gcn_mean", "au_gcn_spread")


def load_cohort(dataset, family):
    if family != "combined":
        return pd.read_csv(dataset / f"{family.replace('+', '_')}.csv"), SPECIFIC[family]
    frames = []
    for species in SPECIFIC:
        frame = pd.read_csv(dataset / f"{species.replace('+', '_')}.csv")
        element = "C" if species == "CO" else "O"
        frames.append(frame.rename(columns={f"{element}_{name}": f"contact_{name}"
                                            for name in CONTACT_FEATURES}))
    data = pd.concat(frames, ignore_index=True)
    if not data.geometry_id.is_unique:
        raise ValueError("Combined cohort contains repeated geometry IDs")
    return data, [f"contact_{name}" for name in CONTACT_FEATURES]


def build_selectors(data, descriptors, bins, numeric_candidates):
    selectors = ps.generate_conditions(data[descriptors], n_bins=bins,
                                       numeric_candidates=numeric_candidates)
    if numeric_candidates == "thresholds":
        ordinal = {name for name in descriptors if pd.api.types.is_numeric_dtype(data[name])
                   and name not in {"system_charge", "spin_multiplicity"}
                   and data[name].nunique() <= bins}
        selectors = [c for c in selectors if c.operator != "is_missing" and c.feature not in ordinal]
        for name in sorted(ordinal):
            for value in sorted(data[name].dropna().unique())[:-1]:
                selectors.extend((ps.Condition(name, "le", value), ps.Condition(name, "gt", value)))
    return selectors


def discover(data, targets, descriptors, min_support, bins, depth, numeric_candidates):
    selectors = build_selectors(data, descriptors, bins, numeric_candidates)
    filters = [ps.MinSizeFilter(min_support)]
    if numeric_candidates == "thresholds":
        filters.append(ps.CustomFilter(
            lambda _coverage, _target, conditions: len({c.feature for c in conditions}) == len(conditions),
            anti_monotone=True,
        ))
    rules, covers = [], {}
    for target in targets:
        task = ps.SubgroupDiscoveryTask(
            data, ps.NumericTarget().fit(data, target), selectors, ps.StandardQFNumeric(a=0.5),
            depth=depth, result_set_size=10000 if numeric_candidates == "thresholds" else 100, min_quality=0,
            filters=filters,
        )
        found = ps.DFS().execute(task)
        unique_covers = set()
        ranked = sorted(found, key=lambda result: (-result[0], len(result[1].conditions), str(result[1])))
        for (score, subgroup), mask in zip(ranked, ps.subgroup_covers(ranked, data)):
            if score <= 0:
                continue
            members = frozenset(data.loc[mask, "geometry_id"].astype(int))
            if members in unique_covers:
                continue
            unique_covers.add(members)
            key = f"{target}:{len(unique_covers)}"
            covers[key] = members
            rules.append(dict(rule_id=key, target=target, rule=str(subgroup), score=score,
                              descriptor_count=len({c.feature for c in subgroup.conditions}),
                              support=len(members), mean=data.loc[mask, target].mean(),
                              baseline=data[target].mean(),
                              family_counts=";".join(f"{name}={count}" for name, count in
                                                     data.loc[mask, "family"].value_counts().sort_index().items()),
                              nuclearities=";".join(map(str, sorted(data.loc[mask, "au_n"].unique()))),
                              geometry_ids=";".join(map(str, sorted(members)))))
            if len(unique_covers) == 20:
                break
    return rules, covers


def main(args):
    data, specific = load_cohort(args.dataset, args.family)
    prefix = f"{args.window}_{args.direction}"
    comparison_targets = [f"{args.window}_f_sum", f"{prefix}_fraction", f"{prefix}_weighted"]
    if args.target == "compare":
        targets = comparison_targets
    else:
        source = f"{prefix}_best_root" if args.target == "peak-product" else f"{prefix}_weighted"
        targets = [f"{prefix}_{args.target.replace('-', '_')}"]
        data[targets[0]] = 100 * data[source]
    eligible = data[targets].notna().all(axis=1)
    if not args.include_negative_roots:
        eligible &= ~data.negative_energy_line_present.eq(True)
    if args.max_delta_e is not None:
        eligible &= data.delta_e_within_archive_family_ev.le(args.max_delta_e)
    if args.nuclearity is not None:
        eligible &= data.au_n.eq(args.nuclearity)
    data = data.loc[eligible].reset_index(drop=True)
    descriptors = [name for name in COMMON + specific
                   if name in data and data[name].nunique() > 1
                   and (not args.exclude_size or name not in {"au_n", "au_rg_a", "au_span_a"})]
    can_search = len(data) > args.min_support and bool(descriptors)
    rules, covers = discover(data, targets, descriptors, args.min_support, args.bins,
                             args.depth, args.numeric_candidates) if can_search else ([], {})
    overlaps = []
    for brightness in (r for r in rules if r["target"] == comparison_targets[0]):
        for fraction in (r for r in rules if r["target"] == comparison_targets[1]):
            left, right = covers[brightness["rule_id"]], covers[fraction["rule_id"]]
            intersection = left & right
            if intersection:
                values = data.loc[data.geometry_id.isin(intersection), comparison_targets[2]]
                overlaps.append(dict(brightness_rule=brightness["rule_id"], fraction_rule=fraction["rule_id"],
                                     overlap_n=len(intersection), jaccard=len(intersection)/len(left | right),
                                     mean_weighted_ifct=values.mean(),
                                     geometry_ids=";".join(map(str, sorted(intersection)))))
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(pysubgroup_version=version("pysubgroup"), eligible_count=len(data),
                    search_performed=can_search, targets=targets, descriptors=descriptors)
    (args.out / "search.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    pd.DataFrame(rules, columns=["rule_id","target","rule","descriptor_count","score","support","mean","baseline",
                                 "family_counts","nuclearities","geometry_ids"]).to_csv(args.out / "rules.csv", index=False)
    pd.DataFrame(overlaps, columns=["brightness_rule","fraction_rule","overlap_n","jaccard",
                                    "mean_weighted_ifct","geometry_ids"]).to_csv(args.out / "overlap.csv", index=False)
    data[["geometry_id", "family", "au_n", *targets]].to_csv(args.out / "eligible.csv", index=False)
    print(f"{args.family}: {len(data)} eligible geometries, {len(rules)} rules, {len(overlaps)} overlaps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--family", choices=(*SPECIFIC, "combined"), required=True)
    parser.add_argument("--window", choices=WINDOWS, default="visible")
    parser.add_argument("--direction", choices=["au_to_ads", "ads_to_au"], default="au_to_ads")
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--numeric-candidates", choices=("bins", "thresholds"), default="bins")
    parser.add_argument("--target", choices=("compare", "peak-product", "sum-product"), default="compare")
    parser.add_argument("--exclude-size", action="store_true",
                        help="Exclude Au nuclearity, radius of gyration and maximum span")
    parser.add_argument("--nuclearity", type=int)
    parser.add_argument("--max-delta-e", type=float)
    parser.add_argument("--include-negative-roots", action="store_true")
    args = parser.parse_args()
    if args.min_support < 2 or args.bins < 2:
        parser.error("minimum support and bin count must be at least two")
    main(args)

