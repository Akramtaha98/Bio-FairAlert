import pandas as pd
import numpy as np

from config import DATA_DIR, OUT_DIR
CUTOFF_DAY = 75

def load_raw():
    si = pd.read_csv(f"{DATA_DIR}/studentInfo.csv")
    sr = pd.read_csv(f"{DATA_DIR}/studentRegistration.csv")
    sa = pd.read_csv(f"{DATA_DIR}/studentAssessment.csv")
    asm = pd.read_csv(f"{DATA_DIR}/assessments.csv")
    courses = pd.read_csv(f"{DATA_DIR}/courses.csv")
    return si, sr, sa, asm, courses

def build_cohort(si, sr, cutoff_day=CUTOFF_DAY):
    key = ["code_module", "code_presentation", "id_student"]
    df = si.merge(sr, on=key, how="left")
    # keep only students who had NOT already (formally) withdrawn by the cutoff day
    # i.e. this is the population for whom an early-warning intervention is still meaningful
    active_mask = df["date_unregistration"].isna() | (df["date_unregistration"] > cutoff_day)
    cohort = df[active_mask].copy()
    return cohort

def build_assessment_features(sa, asm, cutoff_day=CUTOFF_DAY):
    asm_early = asm[asm["date"].notna() & (asm["date"] <= cutoff_day)].copy()
    # expected number of assessments due by cutoff, per module/presentation
    n_due = asm_early.groupby(["code_module", "code_presentation"]).size().rename("n_assessments_due").reset_index()

    merged = sa.merge(asm_early, on="id_assessment", how="inner")  # only assessments due within cutoff
    merged = merged[merged["date_submitted"] <= cutoff_day]  # only count submissions actually made by cutoff
    merged["lateness"] = merged["date_submitted"] - merged["date"]
    merged["weighted_score"] = merged["score"].fillna(0) * merged["weight"]

    agg = merged.groupby(["code_module", "code_presentation", "id_student"]).agg(
        n_submitted=("id_assessment", "count"),
        mean_score=("score", "mean"),
        sum_weight=("weight", "sum"),
        sum_weighted_score=("weighted_score", "sum"),
        mean_lateness=("lateness", "mean"),
        n_banked=("is_banked", "sum"),
    ).reset_index()
    agg["weighted_mean_score"] = np.where(agg["sum_weight"] > 0, agg["sum_weighted_score"] / agg["sum_weight"], np.nan)
    agg = agg.drop(columns=["sum_weight", "sum_weighted_score"])
    return agg, n_due

def build_feature_table(cutoff_day=CUTOFF_DAY):
    si, sr, sa, asm, courses = load_raw()
    cohort = build_cohort(si, sr, cutoff_day)
    agg, n_due = build_assessment_features(sa, asm, cutoff_day)

    df = cohort.merge(agg, on=["code_module", "code_presentation", "id_student"], how="left")
    df = df.merge(n_due, on=["code_module", "code_presentation"], how="left")
    df = df.merge(courses, on=["code_module", "code_presentation"], how="left")

    for c in ["n_submitted", "n_banked", "n_assessments_due"]:
        df[c] = df[c].fillna(0)
    df["submission_rate"] = np.where(df["n_assessments_due"] > 0, df["n_submitted"] / df["n_assessments_due"], np.nan)
    # mean_score / weighted_mean_score / mean_lateness stay NaN when nothing submitted -> real signal, models handle NaN (trees) or impute (LR)

    feature_cols = [
        "code_module", "code_presentation", "gender", "region", "highest_education",
        "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability",
        "date_registration", "module_presentation_length",
        "n_assessments_due", "n_submitted", "submission_rate",
        "mean_score", "weighted_mean_score", "mean_lateness", "n_banked",
    ]
    target_col = "final_result"
    group_col = "id_student"

    out = df[feature_cols + [target_col, group_col]].copy()
    return out, feature_cols, target_col, group_col

if __name__ == "__main__":
    out, feature_cols, target_col, group_col = build_feature_table()
    print(out.shape)
    print(out[target_col].value_counts())
    print(out.isna().mean().sort_values(ascending=False).head(10))
    out.to_csv(f"{OUT_DIR}/feature_table.csv", index=False)
