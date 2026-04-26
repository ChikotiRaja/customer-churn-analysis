"""
Customer Churn Analysis - Enterprise-Grade Preprocessing Pipeline
=================================================================
Designed for large datasets (millions of rows).
Outputs an optimized, Tableau-ready aggregated dataset.

Usage:
    python churn_preprocessing.py --input data/raw_customers.csv --output data/tableau_ready.csv

Dependencies:
    pip install pandas numpy sqlalchemy pyarrow fastparquet tqdm
"""

import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("churn_pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SCHEMA CONSTANTS
# ─────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "customer_id",
    "churn_flag",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "payment_method",
    "internet_service",
    "phone_service",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tech_support",
    "online_security",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
    "num_products",
    "last_activity_days",
]

CONTRACT_TYPES = ["Month-to-Month", "One Year", "Two Year"]
INTERNET_TYPES = ["DSL", "Fiber Optic", "No Internet"]
PAYMENT_METHODS = ["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"]


# ─────────────────────────────────────────────
# STEP 1 — DATA INGESTION (chunk-based for scale)
# ─────────────────────────────────────────────
def load_data_chunked(filepath: str, chunksize: int = 100_000) -> pd.DataFrame:
    """
    Read large CSVs in chunks to avoid memory exhaustion.
    Alternatively, swap this with a SQLAlchemy read for DB sources.
    """
    log.info(f"Loading data from {filepath} (chunksize={chunksize:,})")
    chunks = []
    total_rows = 0

    for chunk in tqdm(
        pd.read_csv(filepath, chunksize=chunksize, low_memory=False),
        desc="Reading chunks",
    ):
        chunks.append(chunk)
        total_rows += len(chunk)

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"Loaded {total_rows:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# STEP 2 — DATA QUALITY & CLEANING
# ─────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log.info("Schema validation passed.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting data cleaning...")
    original_rows = len(df)

    # ── 2a. Deduplicate ──────────────────────────────────────────────────
    df = df.drop_duplicates(subset=["customer_id"], keep="last")
    log.info(f"Duplicates removed: {original_rows - len(df):,}")

    # ── 2b. Type coercion ────────────────────────────────────────────────
    df["churn_flag"] = (
        df["churn_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "1": 1, "true": 1, "no": 0, "0": 0, "false": 0})
        .fillna(0)
        .astype(np.int8)
    )

    df["tenure_months"] = pd.to_numeric(df["tenure_months"], errors="coerce")
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["num_products"] = pd.to_numeric(df["num_products"], errors="coerce")
    df["last_activity_days"] = pd.to_numeric(df["last_activity_days"], errors="coerce")
    df["senior_citizen"] = df["senior_citizen"].astype(np.int8)

    # ── 2c. Handle missing values ────────────────────────────────────────
    # Numeric: median imputation (robust to outliers)
    for col in ["tenure_months", "monthly_charges", "last_activity_days"]:
        median_val = df[col].median()
        filled = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        if filled > 0:
            log.info(f"  {col}: filled {filled:,} NaN → median ({median_val:.2f})")

    # total_charges: derive from tenure × monthly when missing
    mask = df["total_charges"].isna()
    df.loc[mask, "total_charges"] = (
        df.loc[mask, "tenure_months"] * df.loc[mask, "monthly_charges"]
    )
    log.info(f"  total_charges: derived {mask.sum():,} missing values")

    # Categorical: fill with mode or "Unknown"
    for col in ["contract_type", "payment_method", "internet_service"]:
        mode_val = df[col].mode(dropna=True)[0] if not df[col].mode(dropna=True).empty else "Unknown"
        df[col] = df[col].fillna(mode_val).astype(str).str.strip()

    for col in ["gender", "partner", "dependents", "tech_support",
                "online_security", "streaming_tv", "streaming_movies",
                "paperless_billing", "phone_service"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # ── 2d. Outlier clamping (IQR-based) ────────────────────────────────
    for col in ["monthly_charges", "tenure_months"]:
        q1, q3 = df[col].quantile([0.01, 0.99])
        before = ((df[col] < q1) | (df[col] > q3)).sum()
        df[col] = df[col].clip(lower=q1, upper=q3)
        log.info(f"  {col}: clamped {before:,} outliers to [{q1:.2f}, {q3:.2f}]")

    log.info(f"Cleaning complete. Rows remaining: {len(df):,}")
    return df


# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering features...")

    # ── 3a. Tenure groups (Tableau-friendly labels) ──────────────────────
    df["tenure_group"] = pd.cut(
        df["tenure_months"],
        bins=[0, 6, 12, 24, 36, 60, np.inf],
        labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-36 mo", "37-60 mo", "60+ mo"],
        right=True,
    ).astype(str)

    # ── 3b. Monthly charges bands ────────────────────────────────────────
    df["monthly_charges_group"] = pd.cut(
        df["monthly_charges"],
        bins=[0, 30, 55, 75, 90, np.inf],
        labels=["<$30", "$30-$55", "$55-$75", "$75-$90", "$90+"],
        right=True,
    ).astype(str)

    # ── 3c. Normalize contract type ──────────────────────────────────────
    contract_map = {
        "month-to-month": "Month-to-Month",
        "one year": "One Year",
        "two year": "Two Year",
    }
    df["contract_type"] = (
        df["contract_type"].str.lower().str.strip().map(contract_map).fillna("Month-to-Month")
    )

    # ── 3d. Churn risk score (pre-computed, no Tableau LOD needed) ───────
    # Simple weighted heuristic (replace with ML model score if available)
    df["churn_risk_score"] = (
        (df["contract_type"] == "Month-to-Month").astype(int) * 30
        + (df["tenure_months"] < 12).astype(int) * 20
        + (df["monthly_charges"] > 75).astype(int) * 15
        + (df["last_activity_days"] > 30).astype(int) * 20
        + (df["num_products"] <= 1).astype(int) * 10
        + (df["tech_support"] == "No").astype(int) * 5
    ).clip(0, 100)

    # ── 3e. Revenue at risk ──────────────────────────────────────────────
    df["monthly_revenue_at_risk"] = np.where(
        df["churn_flag"] == 1, df["monthly_charges"], 0
    )

    # ── 3f. Customer lifetime value (projected) ──────────────────────────
    df["projected_clv"] = df["monthly_charges"] * np.where(
        df["contract_type"] == "Two Year", 24,
        np.where(df["contract_type"] == "One Year", 12, 6),
    )

    # ── 3g. Engagement score ─────────────────────────────────────────────
    df["engagement_score"] = (
        df["streaming_tv"].eq("Yes").astype(int)
        + df["streaming_movies"].eq("Yes").astype(int)
        + df["online_security"].eq("Yes").astype(int)
        + df["tech_support"].eq("Yes").astype(int)
        + df["phone_service"].eq("Yes").astype(int)
    )

    log.info("Feature engineering complete.")
    return df


# ─────────────────────────────────────────────
# STEP 4 — AGGREGATION (Tableau extract optimization)
# ─────────────────────────────────────────────
def aggregate_for_tableau(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-aggregate by segment dimensions.
    This dramatically reduces Tableau extract size and query time.
    """
    log.info("Aggregating data for Tableau extract...")

    segment_keys = [
        "tenure_group",
        "monthly_charges_group",
        "contract_type",
        "internet_service",
        "senior_citizen",
        "payment_method",
    ]

    agg = (
        df.groupby(segment_keys, observed=True)
        .agg(
            total_customers=("customer_id", "count"),
            churned_customers=("churn_flag", "sum"),
            churn_rate=("churn_flag", "mean"),
            avg_tenure_months=("tenure_months", "mean"),
            avg_monthly_charges=("monthly_charges", "mean"),
            total_monthly_revenue=("monthly_charges", "sum"),
            revenue_at_risk=("monthly_revenue_at_risk", "sum"),
            avg_clv=("projected_clv", "mean"),
            avg_churn_risk_score=("churn_risk_score", "mean"),
            avg_engagement_score=("engagement_score", "mean"),
            high_risk_customers=(
                "churn_risk_score",
                lambda x: (x >= 60).sum(),
            ),
        )
        .reset_index()
    )

    # Round for Tableau readability
    agg["churn_rate"] = agg["churn_rate"].round(4)
    agg["avg_tenure_months"] = agg["avg_tenure_months"].round(1)
    agg["avg_monthly_charges"] = agg["avg_monthly_charges"].round(2)
    agg["avg_clv"] = agg["avg_clv"].round(2)
    agg["avg_churn_risk_score"] = agg["avg_churn_risk_score"].round(1)
    agg["avg_engagement_score"] = agg["avg_engagement_score"].round(2)

    log.info(f"Aggregation complete. Output rows: {len(agg):,}")
    return agg


# ─────────────────────────────────────────────
# STEP 5 — HIGH-RISK CUSTOMER DETAIL TABLE
# ─────────────────────────────────────────────
def extract_high_risk_detail(df: pd.DataFrame, top_n: int = 10_000) -> pd.DataFrame:
    """
    Extract a capped detail table of high-risk customers.
    Used for the 'High-Risk Customers' Tableau sheet (row-level but small).
    """
    log.info(f"Extracting top {top_n:,} high-risk customers...")

    detail_cols = [
        "customer_id",
        "churn_flag",
        "tenure_months",
        "tenure_group",
        "monthly_charges",
        "monthly_charges_group",
        "contract_type",
        "internet_service",
        "payment_method",
        "senior_citizen",
        "churn_risk_score",
        "monthly_revenue_at_risk",
        "projected_clv",
        "engagement_score",
        "last_activity_days",
    ]

    high_risk = (
        df[df["churn_risk_score"] >= 50]
        .sort_values("churn_risk_score", ascending=False)
        .head(top_n)[detail_cols]
        .reset_index(drop=True)
    )

    log.info(f"High-risk detail rows: {len(high_risk):,}")
    return high_risk


# ─────────────────────────────────────────────
# STEP 6 — EXPORT
# ─────────────────────────────────────────────
def export_datasets(
    agg_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Primary Tableau extract (aggregated)
    agg_path = os.path.join(output_dir, "churn_tableau_extract.csv")
    agg_df.to_csv(agg_path, index=False)
    log.info(f"Aggregated extract saved: {agg_path} ({os.path.getsize(agg_path)/1024:.1f} KB)")

    # High-risk detail table
    detail_path = os.path.join(output_dir, "churn_high_risk_detail.csv")
    detail_df.to_csv(detail_path, index=False)
    log.info(f"High-risk detail saved: {detail_path} ({os.path.getsize(detail_path)/1024:.1f} KB)")

    # Parquet versions (faster Tableau extracts)
    try:
        agg_df.to_parquet(os.path.join(output_dir, "churn_tableau_extract.parquet"), index=False)
        detail_df.to_parquet(os.path.join(output_dir, "churn_high_risk_detail.parquet"), index=False)
        log.info("Parquet files also saved (preferred for Tableau extracts).")
    except Exception as e:
        log.warning(f"Parquet export skipped: {e}")

    # Data dictionary
    _write_data_dictionary(agg_df, detail_df, output_dir)


def _write_data_dictionary(agg_df, detail_df, output_dir):
    dd_lines = [
        "# Churn Analysis — Data Dictionary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Aggregated Extract (churn_tableau_extract)",
        "| Column | Type | Description |",
        "|--------|------|-------------|",
        "| tenure_group | string | Binned tenure: 0-6 mo … 60+ mo |",
        "| monthly_charges_group | string | Charge band: <$30 … $90+ |",
        "| contract_type | string | Month-to-Month / One Year / Two Year |",
        "| internet_service | string | DSL / Fiber Optic / No Internet |",
        "| senior_citizen | int | 0=No, 1=Yes |",
        "| payment_method | string | Payment method label |",
        "| total_customers | int | Count of customers in segment |",
        "| churned_customers | int | Count of churned customers |",
        "| churn_rate | float | churned / total (0.0–1.0) |",
        "| avg_tenure_months | float | Mean tenure in months |",
        "| avg_monthly_charges | float | Mean monthly charge ($) |",
        "| total_monthly_revenue | float | Sum of monthly charges ($) |",
        "| revenue_at_risk | float | Revenue from churned customers ($) |",
        "| avg_clv | float | Projected customer lifetime value ($) |",
        "| avg_churn_risk_score | float | Heuristic risk score (0–100) |",
        "| avg_engagement_score | float | Mean # of active services |",
        "| high_risk_customers | int | Count with risk score ≥ 60 |",
        "",
        "## High-Risk Detail (churn_high_risk_detail)",
        "Row-level table capped at 10,000 records. Used for drill-down.",
        f"Rows: {len(detail_df):,}",
    ]
    with open(os.path.join(output_dir, "data_dictionary.md"), "w") as f:
        f.write("\n".join(dd_lines))
    log.info("Data dictionary written.")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(input_path: str, output_dir: str = "output") -> None:
    t0 = time.time()
    log.info("=" * 60)
    log.info("CUSTOMER CHURN PREPROCESSING PIPELINE — START")
    log.info("=" * 60)

    df = load_data_chunked(input_path)
    validate_schema(df)
    df = clean_data(df)
    df = engineer_features(df)

    agg_df = aggregate_for_tableau(df)
    detail_df = extract_high_risk_detail(df)

    export_datasets(agg_df, detail_df, output_dir)

    elapsed = time.time() - t0
    log.info(f"Pipeline complete in {elapsed:.1f}s")
    log.info(
        f"Summary: {len(df):,} customers → "
        f"{len(agg_df):,} segment rows + {len(detail_df):,} high-risk rows"
    )


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Analysis Preprocessing Pipeline")
    parser.add_argument("--input", required=True, help="Path to raw CSV input")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)
