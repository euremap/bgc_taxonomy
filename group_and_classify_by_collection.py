#! /usr/bin/env python3

"""Summarise GCF groups: size, deepest common rank with value, culture-collection
status, and member ibgc identifiers.

Ranking logic as per group_and_classify.py: a rank is common if, at all
shallower ranks, values present do not conflict, and the rank itself has
at least one value. Missing values never conflict.

Output
------
gcf_group  n_members  deepest_common_rank  deepest_common_rank_value  
in_culture_collection  members
Tab-separated; in_culture_collection blank if no member is True;
deepest_common_rank_value blank if rank is "none".
"""

import pandas as pd

INFILE = "data/ibgc_taxonomy_culture.tsv"
SUMMARY_FILE = "GCF_by_culture_collection.tsv"

RANKS = ["domain", "kingdom", "phylum", "class",
         "order", "family", "genus", "species", "strain"]

def load_gcf_groups(infile):
    df = pd.read_csv(
        infile,
        sep="\t",
        usecols=["ibgc", "gcf", "taxonomy_path", "in_culture_collection"],
        dtype="string",
        engine="c",
    )
    df = df.dropna(subset=["gcf", "taxonomy_path"])
    # normalise whitespace only; DO NOT collapse ".." — empty interior
    # ranks must stay as empty slots to preserve rank alignment
    df["tax_norm"] = (df["taxonomy_path"]
                      .str.strip()
                      .str.replace(r"\s*\.\s*", ".", regex=True)
                      .str.rstrip("."))

    # Classify only on the first element of the gcf dot notation
    # df["gcf_group"] = df["gcf"].str.split(".", n=1, expand=True)[0]
    # Classify on the whole gcf dot notation
    df["gcf_group"] = df["gcf"]

    ranks = (df["tax_norm"].str.split(".", expand=True)
             .reindex(columns=range(len(RANKS)))
             .fillna(""))
    ranks.columns = RANKS
    df = pd.concat([df, ranks], axis=1)
    df[RANKS] = df[RANKS].mask(df[RANKS] == "")
    return df

def compute_agreement(df):
    """Return (agree_df, n_distinct_df) for the rest of the pipeline."""
    n_distinct = df.groupby("gcf_group")[RANKS].nunique()
    
    # no contradiction at this rank or any shallower one
    no_conflict = (n_distinct <= 1).astype(int).cummin(axis=1).astype(bool)
    # at least one member actually supplies a value at this rank
    has_value = n_distinct == 1
    agree = no_conflict & has_value
    
    return agree, n_distinct

def deepest_common_ranks(df):
    """Per gcf_group: the deepest common rank ("none" if no such rank)."""
    agree, _ = compute_agreement(df)
    
    # deepest rank where agree is True ("none" if there is no such rank)
    last_true = agree.iloc[:, ::-1].idxmax(axis=1)
    return last_true.where(agree.any(axis=1), "none")

def deepest_rank_values(df):
    """Per gcf_group: the taxon value at the deepest common rank.
    Blank (NaN) where the deepest common rank is "none"."""
    agree, _ = compute_agreement(df)
    last_true = agree.iloc[:, ::-1].idxmax(axis=1).where(agree.any(axis=1))

    # first (any) non-NA value per group, per rank
    first_val = df.groupby("gcf_group")[RANKS].first()

    # pick the value at each group's deepest common rank
    has = last_true.notna()
    pairs = pd.MultiIndex.from_arrays(
        [last_true.index[has], last_true[has]])

    vals = first_val.stack().reindex(pairs).droplevel(-1)
    return vals.reindex(last_true.index)

def deepest_agreement_report(infile, outfile=None):
    df = load_gcf_groups(infile)

    sizes = df.groupby("gcf_group").size()
    multi_idx = sizes[sizes > 1].index

    deepest = deepest_common_ranks(df).loc[multi_idx]

    order = ["none"] + RANKS
    counts = deepest.value_counts().reindex(order).fillna(0).astype(int)

    print(f"multi-member groups analysed: {len(deepest)}")
    print("\nGroups by deepest common rank:")
    for rank, n in counts.items():
        print(f"  {rank:<10} {n:>6}  ({100 * n / len(deepest):.1f}%)")

    if outfile:
        deepest.rename_axis("gcf_group").rename("deepest_common_rank") \
               .to_csv(outfile, sep="\t")

    return counts
    
def build_summary_table(infile, outfile=SUMMARY_FILE):
    df = load_gcf_groups(infile)

    sizes = df.groupby("gcf_group").size()
    groups_index = df["gcf_group"].unique()

    # number of members flagged True in the culture collection
    culture_true = (df["in_culture_collection"]
                   .str.strip().str.lower().eq("true"))
    n_cultured = culture_true.groupby(df["gcf_group"]).sum()

    # ibgc identifiers concatenated in file order, but ONLY where
    # in_culture_collection is True
    cultured_ids = df[culture_true].groupby("gcf_group")["ibgc"].agg(
        lambda s: ";".join(s.fillna("")))

    # deepest common rank and its value
    deepest_rank = deepest_common_ranks(df)
    rank_value = deepest_rank_values(df)

    summary = pd.DataFrame({
        "gcf_group": groups_index,
        "n_members": sizes.reindex(groups_index).values,
        "deepest_common_rank": deepest_rank.reindex(groups_index).values,
        "deepest_common_rank_value": rank_value.reindex(groups_index).values,
        "in_culture_collection": n_cultured.reindex(groups_index)
                                         .where(n_cultured.reindex(groups_index) > 0)
                                         .values,
        "members": cultured_ids.reindex(groups_index).fillna("").values,
    }).reset_index(drop=True)

    # numeric order on the group prefix where possible
    summary["_k"] = pd.to_numeric(summary["gcf_group"], errors="coerce")
    summary = (summary.sort_values(["_k", "gcf_group"])
                     .drop(columns="_k")
                     .reset_index(drop=True))

    summary.to_csv(outfile, sep="\t", index=False, na_rep="")
    print(f"{len(summary)} groups written to {outfile}")
    return summary

if __name__ == "__main__":
    build_summary_table(INFILE)
    deepest_agreement_report(INFILE)   # uncomment for the rank histogram
