import numpy as np
import pandas as pd

from config import MOOC_RAW_DIR as RAW_DIR


def load_raw():
    actions = pd.read_csv(f"{RAW_DIR}/mooc_actions.tsv", sep="\t")
    feats = pd.read_csv(f"{RAW_DIR}/mooc_action_features.tsv", sep="\t")
    labels = pd.read_csv(f"{RAW_DIR}/mooc_action_labels.tsv", sep="\t")
    df = actions.merge(feats, on="ACTIONID").merge(labels, on="ACTIONID")
    return df


def build_feature_table(cutoff_day, k_min_users=50):
    df = load_raw()
    # TIMESTAMP is a *global*, platform-wide clock (seconds since the course/platform's own
    # start), NOT already relative to each user's enrollment -- verified empirically: per-user
    # minimum TIMESTAMP varies from 0 to >2.5M seconds across users. Relative day-of-enrollment
    # is therefore computed per user, matching the "Timestamp Normalization" step described in
    # the prior draft this case study extends.
    user_start = df.groupby("USERID")["TIMESTAMP"].transform("min")
    df["day"] = (df["TIMESTAMP"] - user_start) / (60 * 60 * 24.0)
    within = df[df["day"] <= cutoff_day].copy()

    feature_cols_raw = ["FEATURE0", "FEATURE1", "FEATURE2", "FEATURE3"]
    agg = within.groupby("USERID").agg(
        **{f"{c}_mean": (c, "mean") for c in feature_cols_raw},
        **{f"{c}_std": (c, "std") for c in feature_cols_raw},
        **{f"{c}_max": (c, "max") for c in feature_cols_raw},
        **{f"{c}_min": (c, "min") for c in feature_cols_raw},
        num_actions=("ACTIONID", "count"),
        day_mean=("day", "mean"), day_max=("day", "max"), day_min=("day", "min"), day_std=("day", "std"),
        max_label=("LABEL", "max"),
    ).reset_index()

    feature_cols = [c for c in agg.columns if c not in ("USERID", "max_label")]
    target_col = "max_label"
    return agg, feature_cols, target_col


if __name__ == "__main__":
    df = load_raw()
    print("raw actions", df.shape)
    user_start = df.groupby("USERID")["TIMESTAMP"].transform("min")
    df["day"] = (df["TIMESTAMP"] - user_start) / (60 * 60 * 24.0)
    mind = df.groupby("USERID")["day"].min()
    print("max of per-user min day (should be 0):", mind.max())
    print("relative day range:", df["day"].min(), df["day"].max())

    for c in [1, 3, 7, 14, 21]:
        counts = df[df["day"] <= c].groupby("USERID").size()
        print(f"cutoff={c}: users with >=1 action by this cutoff = {(counts>=1).sum()}")
