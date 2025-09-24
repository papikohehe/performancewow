# app.py

import io
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="เครื่องมือประเมินประสิทธิภาพ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Constants & Helpers
# -----------------------------
# Count an hour ONLY if Processed Count > 100 (per instruction)
HOUR_COUNT_THRESHOLD = 100

# These names are now assigned based on column position
REQUIRED_COLS = ["Time (day)", "Hour", "Employee ID", "Name", "Processed Count"]
COLUMN_POSITION_MAPPING = {
    0: "Time (day)",
    1: "Hour",
    2: "Employee ID",
    3: "Name",
    4: "Processed Count",
}


@st.cache_data(show_spinner=False)
def read_csv_safely(file) -> pd.DataFrame:
    """Robust CSV loader that reads by position, skipping the header row."""
    for sep in [",", "\t", ";"]:
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                file.seek(0)
                # Use header=None to read by position, skiprows=1 to skip original header
                df = pd.read_csv(file, sep=sep, encoding=enc, header=None, skiprows=1)
                if df.shape[1] >= 3:
                    return df
            except Exception:
                continue
    # Fallback to pandas default
    file.seek(0)
    return pd.read_csv(file, header=None, skiprows=1)


def coerce_schema(df: pd.DataFrame, filename: str = "") -> Tuple[pd.DataFrame, List[str]]:
    """Ensure required columns exist and coerce dtypes, with improved date logic."""
    issues = []
    
    # Filter out records where Name is "-"
    if "Name" in df.columns:
        df = df[df["Name"] != "-"].copy()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns based on position: {missing}")

    # --- Date Handling Logic with pre-cleaning ---
    if "Time (day)" in df.columns:
        # First, drop rows that have no date value at all (e.g., from blank lines in CSV)
        df.dropna(subset=["Time (day)"], inplace=True)
        if df.empty:
            issues.append("No valid data rows remained after removing rows with empty dates.")
            return df, issues
        
        reconstructed_from_filename = False
        
        day_numbers = pd.to_numeric(df["Time (day)"], errors='coerce')
        is_likely_day_numbers = (
            day_numbers.notna().all() 
            and day_numbers.min() >= 1 
            and day_numbers.max() <= 31
        )

        if is_likely_day_numbers:
            match = re.search(r"(\d{4})-(\d{2})", filename)
            if match:
                year, month = match.groups()
                try:
                    df['Time (day)'] = pd.to_datetime(
                        {'year': int(year), 'month': int(month), 'day': day_numbers}
                    )
                    reconstructed_from_filename = True
                except Exception as e:
                    issues.append(f"Error building date from filename: {e}")
            else:
                issues.append("Data looks like day numbers, but no YYYY-MM pattern found in filename.")

        if not reconstructed_from_filename:
            try:
                processed_dates = pd.to_datetime(df["Time (day)"], errors="coerce")
                if processed_dates.isna().all():
                    issues.append("Failed to parse 'Time (day)' as dates.")
                    df['Time (day)'] = pd.NaT
                else:
                    df['Time (day)'] = processed_dates
            except Exception as e:
                issues.append(f"An error occurred while parsing dates: {e}")
                df['Time (day)'] = pd.NaT

        if pd.api.types.is_datetime64_any_dtype(df["Time (day)"]):
            df["Time (day)"] = df["Time (day)"].dt.date
        else:
            df["Time (day)"] = pd.NaT

    # Coerce Hour to int 0-23 where possible
    if "Hour" in df.columns:
        def _to_hour(x):
            try:
                s = str(x).strip()
                if ":" in s:
                    s = s.split(":")[0]
                h = int(float(s))
                if 0 <= h <= 23:
                    return h
            except Exception:
                return np.nan
            return np.nan
        df["Hour"] = df["Hour"].apply(_to_hour)

    # Employee ID as string
    if "Employee ID" in df.columns:
        df["Employee ID"] = df["Employee ID"].astype(str).str.replace(r'\.0$', '', regex=True)

    # Name as string
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str)

    # Processed Count as numeric >= 0
    if "Processed Count" in df.columns:
        df["Processed Count"] = pd.to_numeric(df["Processed Count"], errors="coerce")
        neg = (df["Processed Count"] < 0).sum()
        if neg > 0:
            issues.append(f"{neg} rows had negative 'Processed Count' and were set to 0.")
        df["Processed Count"] = df["Processed Count"].fillna(0)
        df.loc[df["Processed Count"] < 0, "Processed Count"] = 0

    return df, issues


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Combine duplicates by summing Processed Count within the same (day, hour, emp)."""
    key_cols = [c for c in ["Time (day)", "Hour", "Employee ID", "Name"] if c in df.columns]
    if not key_cols or "Processed Count" not in df.columns:
        return df
    g = df.groupby(key_cols, dropna=False, as_index=False)["Processed Count"].sum()
    return g


def compute_aggregates(
    df: pd.DataFrame,
    processed_threshold: int = HOUR_COUNT_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        - emp_hourly: granular per (day, hour, emp)
        - emp_daily: per (day, emp)
        - emp_summary: per employee

    Hour counting rule:
      Count a worked hour ONLY if 'Processed Count' > processed_threshold.
    """
    df = df.copy()

    if "Hour" not in df.columns or "Time (day)" not in df.columns:
        st.warning("Cannot compute aggregates without 'Time (day)' and 'Hour' columns.")
        return df, df, df

    # Count an hour only when Processed Count strictly greater than threshold
    worked_mask = df["Processed Count"] > processed_threshold
    df["WorkedHourFlag"] = worked_mask.astype(int)

    daily_keys = ["Time (day)", "Employee ID", "Name"]
    emp_daily = (
        df.groupby(daily_keys, dropna=False)
        .agg(
            Works=("Processed Count", "sum"),
            Hours=("WorkedHourFlag", "sum"),
            UniqueHours=("Hour", pd.Series.nunique),
        )
        .reset_index()
    )
    emp_daily["WPH"] = emp_daily["Works"] / emp_daily["Hours"].replace(0, np.nan)

    emp_summary = (
        emp_daily.groupby(["Employee ID", "Name"], dropna=False)
        .agg(
            Days=("Time (day)", pd.Series.nunique),
            Hours=("Hours", "sum"),
            UniqueHours=("UniqueHours", "sum"),
            Works=("Works", "sum"),
            AvgWPH=("WPH", "mean"),
            MedianWPH=("WPH", "median"),
            MaxDailyWorks=("Works", "max"),
        )
        .reset_index()
    )
    emp_summary["WPH"] = emp_summary["Works"] / emp_summary["Hours"].replace(0, np.nan)
    emp_summary["WorksPerDay"] = emp_summary["Works"] / emp_summary["Days"].replace(0, np.nan)

    emp_hourly = df.copy()

    return emp_hourly, emp_daily, emp_summary


def zscore(series: pd.Series) -> pd.Series:
    m = series.mean()
    s = series.std(ddof=0)
    if s == 0 or np.isnan(s):
        return pd.Series([0] * len(series), index=series.index)
    return (series - m) / s


def anomaly_report(emp_daily: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """Compute z-score on per-employee daily Works to find spikes or dips."""
    if emp_daily.empty:
        return emp_daily

    def _emp_ano
