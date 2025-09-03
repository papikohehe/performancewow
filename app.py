python
import io
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
    page_title="Performance Evaluator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLS = ["Time (day)", "Hour", "Employee ID", "Name", "Processed Count"]

NORMALIZE_MAP = {
    # Date/Day
    "time (day)": "Time (day)",
    "time": "Time (day)",
    "date": "Time (day)",
    "day": "Time (day)",
    "date (day)": "Time (day)",
    # Hour
    "hour": "Hour",
    "hours": "Hour",
    "time (hour)": "Hour",
    "start hour": "Hour",
    "hr": "Hour",
    # Employee ID
    "employee id": "Employee ID",
    "emp id": "Employee ID",
    "employeeid": "Employee ID",
    "id": "Employee ID",
    # Name
    "name": "Name",
    "employee": "Name",
    "employee name": "Name",
    # Processed Count
    "processed count": "Processed Count",
    "processed": "Processed Count",
    "count": "Processed Count",
    "work": "Processed Count",
    "works": "Processed Count",
    "items": "Processed Count",
}


@st.cache_data(show_spinner=False)
def read_csv_safely(file) -> pd.DataFrame:
    """Robust CSV loader that tries utf-8, then latin-1, with comma or tab separators."""
    # Try comma with utf-8
    for sep in [",", "\t", ";"]:
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                file.seek(0)
                df = pd.read_csv(file, sep=sep, encoding=enc)
                if df.shape[1] >= 3:
                    return df
            except Exception:
                continue
    # Fallback to pandas default
    file.seek(0)
    return pd.read_csv(file)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        key = str(c).strip().lower()
        new_cols.append(NORMALIZE_MAP.get(key, c))
    df.columns = new_cols
    return df


def coerce_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Ensure required columns exist and coerce dtypes."""
    issues = []
    df = normalize_columns(df.copy())

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Coerce Time (day)
    if "Time (day)" in df.columns:
        try:
            df["Time (day)"] = pd.to_datetime(df["Time (day)"], errors="coerce").dt.date
        except Exception as e:
            issues.append(f"Failed to parse 'Time (day)' as dates: {e}")
            df["Time (day)"] = pd.NaT

    # Coerce Hour to int 0-23 where possible
    if "Hour" in df.columns:
        def _to_hour(x):
            try:
                # Handle "09", "9", "9:00"
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
        # If hour strings like "24" appear, clamp to 23
        df.loc[df["Hour"].isna(), "Hour"] = np.nan

    # Employee ID as string
    if "Employee ID" in df.columns:
        df["Employee ID"] = df["Employee ID"].astype(str)

    # Name as string
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str)

    # Processed Count as numeric >= 0
    if "Processed Count" in df.columns:
        df["Processed Count"] = pd.to_numeric(df["Processed Count"], errors="coerce")
        neg = (df["Processed Count"] < 0).sum()
        if neg:
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
    count_hour_when_processed_gt_zero: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        - emp_hourly: granular per (day, hour, emp)
        - emp_daily: per (day, emp)
        - emp_summary: per employee
    """
    df = df.copy()

    # Fill missing hours if any—cannot compute hours without Hour column
    if "Hour" not in df.columns or "Time (day)" not in df.columns:
        st.warning("Cannot compute aggregates without 'Time (day)' and 'Hour'.")
        return df, df, df

    # Decide hours worked
    if count_hour_when_processed_gt_zero:
        worked_mask = df["Processed Count"] > 0
    else:
        worked_mask = ~df["Hour"].isna()

    df["WorkedHourFlag"] = worked_mask.astype(int)

    # Per-employee per-day
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

    # Per-employee summary
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

    # Back out the granular hourly for completeness
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

    def _emp_anom(df_emp):
        df_emp = df_emp.sort_values("Time (day)").copy()
        df_emp["WorksZ"] = zscore(df_emp["Works"])
        df_emp["IsAnomaly"] = df_emp["WorksZ"].abs() >= z_threshold
        return df_emp

    out = (
        emp_daily.groupby(["Employee ID", "Name"], group_keys=False)
        .apply(_emp_anom)
        .reset_index(drop=True)
    )
    return out


def heatmap_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return two pivot tables: throughput heatmap & staffing coverage heatmap."""
    if df.empty or "Time (day)" not in df.columns or "Hour" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    tmp = df.copy()
    tmp["Dow"] = pd.to_datetime(tmp["Time (day)"]).dt.day_name()

    # Throughput by hour x day-of-week
    thr = (
        tmp.groupby(["Dow", "Hour"])["Processed Count"]
        .sum()
        .reset_index()
        .pivot(index="Dow", columns="Hour", values="Processed Count")
        .fillna(0)
    )

    # Coverage: unique employees per hour x day-of-week
    cov = (
        tmp.groupby(["Dow", "Hour"])["Employee ID"]
        .nunique()
        .reset_index()
        .pivot(index="Dow", columns="Hour", values="Employee ID")
        .fillna(0)
    )
    # Sort DOW in conventional order
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    thr = thr.reindex(dow_order)
    cov = cov.reindex(dow_order)
    return thr, cov


def add_topn_bar(df: pd.DataFrame, metric: str, top_n: int, title: str):
    if df.empty or metric not in df.columns:
        st.info("No data to display.")
        return
    show = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top_n)
    fig = px.bar(show, x="Name", y=metric, hover_data=["Employee ID"], title=title)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(show.reset_index(drop=True))


def kpi_block(total_works, total_hours, avg_wph, employees, days):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Works", f"{int(total_works):,}")
    c2.metric("Total Hours", f"{int(total_hours):,}")
    c3.metric("Avg WPH", f"{avg_wph:,.2f}" if not np.isnan(avg_wph) else "—")
    c4.metric("# Employees", f"{employees:,}")
    c5.metric("# Days", f"{days:,}")


def download_csv_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# -----------------------------
# UI
# -----------------------------
st.title("📈 Performance Evaluator")

st.markdown(
    """
Upload 1 or more CSV files with the following columns:
- **Time (day)** (date)  
- **Hour** (0–23)  
- **Employee ID**  
- **Name**  
- **Processed Count**  

The app auto-normalizes similar column names (e.g., `Date`→`Time (day)`, `Processed`→`Processed Count`).
"""
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("How to count a worked hour?")
    hour_count_mode = st.radio(
        "Hours are counted when…",
        options=[
            "Processed Count > 0 (recommended)",
            "Any row exists (even if Processed Count = 0)",
        ],
        index=0,
    )
    count_when_gt_zero = hour_count_mode.startswith("Processed")

    min_hours_threshold = st.slider("Min hours to include in WPH leaderboards", 1, 40, 8, 1)
    top_n = st.slider("Top N to display in leaderboards", 3, 50, 10, 1)

    st.divider()
    st.caption("Optional filters")
    date_filter_on = st.checkbox("Filter by date range", value=False)
    employee_search = st.text_input("Search name contains (optional)", value="")

    st.divider()
    st.caption("Data options")
    use_sample = st.checkbox("Use sample data (if you don't have files yet)", value=False)


# -----------------------------
# Load data
# -----------------------------
uploaded_files = st.file_uploader(
    "Drop CSV files here",
    type=["csv"],
    accept_multiple_files=True,
)

dfs = []
issues_all = []

if use_sample and not uploaded_files:
    # Build a tiny synthetic sample for demo
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-08-01", periods=10, freq="D").date
    hours = list(range(9, 19))  # 9AM-6PM
    employees = [
        ("E001", "Alice"),
        ("E002", "Bob"),
        ("E003", "Chai"),
        ("E004", "Dao"),
    ]
    rows = []
    for d in dates:
        for h in hours:
            for eid, nm in employees:
                # Randomly sparse
                if rng.random() < 0.8:
                    cnt = int(rng.poisson(8))
                else:
                    cnt = 0
                rows.append([d, h, eid, nm, cnt])
    sample = pd.DataFrame(rows, columns=REQUIRED_COLS)
    df0, issues = coerce_schema(sample)
    df0 = deduplicate_rows(df0)
    dfs.append(df0)
    issues_all += issues

for f in uploaded_files or []:
    try:
        raw = read_csv_safely(f)
        df0, issues = coerce_schema(raw)
        df0 = deduplicate_rows(df0)
        dfs.append(df0)
        issues_all += issues
    except Exception as e:
        st.error(f"Failed to read {getattr(f, 'name', 'file')}: {e}")

if not dfs:
    st.info("👆 Upload CSVs or enable **Use sample data** in the sidebar to get started.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# Report issues (if any)
if issues_all:
    with st.expander("Data quality notes"):
        for msg in issues_all:
            st.warning(msg)

# Basic validation
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    st.error(f"Cannot proceed: missing required columns: {missing_cols}")
    st.stop()

# Optional filters
if date_filter_on and "Time (day)" in df.columns:
    min_d = pd.to_datetime(df["Time (day)"]).min()
    max_d = pd.to_datetime(df["Time (day)"]).max()
    d1, d2 = st.slider(
        "Date range",
        min_value=min_d.to_pydatetime().date(),
        max_value=max_d.to_pydatetime().date(),
        value=(min_d.to_pydatetime().date(), max_d.to_pydatetime().date()),
    )
    mask = (pd.to_datetime(df["Time (day)"]) >= pd.to_datetime(d1)) & (
        pd.to_datetime(df["Time (day)"]) <= pd.to_datetime(d2)
    )
    df = df.loc[mask].copy()

if employee_search.strip():
    mask = df["Name"].str.contains(employee_search.strip(), case=False, na=False)
    df = df.loc[mask].copy()

# Compute aggregates
emp_hourly, emp_daily, emp_summary = compute_aggregates(
    df, count_hour_when_processed_gt_zero=count_when_gt_zero
)

# KPIs
total_works = float(emp_daily["Works"].sum()) if not emp_daily.empty else 0.0
total_hours = float(emp_daily["Hours"].sum()) if not emp_daily.empty else 0.0
avg_wph = total_works / total_hours if total_hours > 0 else np.nan
employees = emp_summary.shape[0]
days = emp_daily["Time (day)"].nunique() if "Time (day)" in emp_daily.columns else 0

st.subheader("Overview")
kpi_block(total_works, total_hours, avg_wph, employees, days)

with st.expander("Raw data preview", expanded=False):
    st.dataframe(df.head(100))

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🏆 Leaderboards", "📅 Time Analysis", "👤 Employee Drilldown", "⚠️ Anomalies", "📊 Pivots", "📥 Export"]
)

with tab1:
    st.subheader("Leaderboards")
    st.caption(f"Using min-hours threshold = **{min_hours_threshold}** for WPH.")
    if not emp_summary.empty:
        # Apply min-hours for WPH
        eligible = emp_summary[emp_summary["Hours"] >= min_hours_threshold].copy()
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            add_topn_bar(emp_summary, "Works", top_n, "Top by Works (Total Processed)")
        with col2:
            if eligible.empty:
                st.info("No employees meet the min-hours threshold for WPH ranking.")
            else:
                add_topn_bar(eligible, "WPH", top_n, "Top by Works per Hour (WPH)")
        with col3:
            add_topn_bar(emp_summary, "Hours", top_n, "Top by Hours Worked")

        st.markdown("---")
        st.subheader("Bottom by WPH (with min-hours threshold)")
        if not eligible.empty:
            bottom = eligible.sort_values("WPH", ascending=True).head(top_n)
            fig = px.bar(bottom, x="Name", y="WPH", hover_data=["Employee ID"], title="Bottom by WPH")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bottom.reset_index(drop=True))
        else:
            st.info("No eligible employees for bottom WPH.")

with tab2:
    st.subheader("Time Analysis")
    if emp_hourly.empty:
        st.info("Not enough data for time analysis.")
    else:
        # Hour-of-day throughput
        hourly = emp_hourly.groupby("Hour", as_index=False)["Processed Count"].sum()
        fig1 = px.line(hourly, x="Hour", y="Processed Count", markers=True, title="Throughput by Hour of Day")
        st.plotly_chart(fig1, use_container_width=True)

        # Day-of-week & hour heatmaps
        thr, cov = heatmap_data(emp_hourly)
        if not thr.empty:
            st.markdown("**Throughput Heatmap (Day-of-Week × Hour)**")
            fig2 = px.imshow(thr, aspect="auto", title="Throughput Heatmap", labels=dict(color="Works"))
            st.plotly_chart(fig2, use_container_width=True)

        if not cov.empty:
            st.markdown("**Staffing Coverage Heatmap (Unique Employees Working)**")
            fig3 = px.imshow(cov, aspect="auto", title="Coverage Heatmap", labels=dict(color="# Employees"))
            st.plotly_chart(fig3, use_container_width=True)

        # Daily totals
        daily_total = emp_daily.groupby("Time (day)", as_index=False)["Works"].sum()
        fig4 = px.bar(daily_total, x="Time (day)", y="Works", title="Daily Total Works")
        st.plotly_chart(fig4, use_container_width=True)

        # Peak hours table
        st.markdown("**Top Hours by Throughput**")
        peak_hours = (
            emp_hourly.groupby("Hour", as_index=False)["Processed Count"].sum().sort_values("Processed Count", ascending=False)
        )
        st.dataframe(peak_hours.head(24))

with tab3:
    st.subheader("Employee Drilldown")
    if emp_summary.empty:
        st.info("No employee data.")
    else:
        names = emp_summary["Name"].tolist()
        picked = st.selectbox("Choose an employee", options=names)
        if picked:
            # Determine the employee ID (in case of duplicate names, show all matches)
            emp_ids = emp_summary.loc[emp_summary["Name"] == picked, "Employee ID"].unique().tolist()
            if len(emp_ids) > 1:
                emp_id = st.selectbox("Multiple IDs found for this name. Pick one:", options=emp_ids)
            else:
                emp_id = emp_ids[0]

            dfd = emp_daily[(emp_daily["Name"] == picked) & (emp_daily["Employee ID"] == emp_id)].copy()
            if dfd.empty:
                st.info("No daily records for selection.")
            else:
                c1, c2, c3 = st.columns(3)
                works_total = int(dfd["Works"].sum())
                hours_total = int(dfd["Hours"].sum())
                wph = works_total / hours_total if hours_total > 0 else np.nan
                c1.metric("Works (Total)", f"{works_total:,}")
                c2.metric("Hours (Total)", f"{hours_total:,}")
                c3.metric("WPH (Overall)", f"{wph:,.2f}" if not np.isnan(wph) else "—")

                fig = px.line(dfd, x="Time (day)", y="Works", markers=True, title=f"Daily Works — {picked}")
                st.plotly_chart(fig, use_container_width=True)

                fig_wph = px.line(dfd, x="Time (day)", y="WPH", markers=True, title=f"Daily WPH — {picked}")
                st.plotly_chart(fig_wph, use_container_width=True)

                # Hourly profile of this employee
                base = emp_hourly[(emp_hourly["Name"] == picked) & (emp_hourly["Employee ID"] == emp_id)].copy()
                prof = base.groupby("Hour", as_index=False)["Processed Count"].sum()
                fig_prof = px.bar(prof, x="Hour", y="Processed Count", title=f"Hourly Profile — {picked}")
                st.plotly_chart(fig_prof, use_container_width=True)

                st.markdown("**Records (daily)**")
                st.dataframe(dfd.sort_values("Time (day)"))

with tab4:
    st.subheader("Anomalies")
    if emp_daily.empty:
        st.info("No data for anomaly detection.")
    else:
        z_thr = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.5)
        anom = anomaly_report(emp_daily, z_threshold=z_thr)
        flagged = anom[anom["IsAnomaly"]].copy()
        st.caption("An anomaly is a day where an employee's daily 'Works' deviates from their own mean by >= threshold standard deviations.")
        st.dataframe(flagged.sort_values(["Name", "Time (day)"]))
        if not flagged.empty:
            by_emp = flagged.groupby("Name").size().reset_index(name="Anomaly Days")
            fig = px.bar(by_emp, x="Name", y="Anomaly Days", title="Anomaly Counts by Employee")
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Pivots")
    if emp_hourly.empty:
        st.info("No data for pivots.")
    else:
        # Daily x Employee pivot of Works
        piv = (
            emp_hourly.groupby(["Time (day)", "Employee ID", "Name"], as_index=False)["Processed Count"].sum()
            .pivot_table(index="Time (day)", columns="Name", values="Processed Count", aggfunc="sum")
            .fillna(0)
        )
        st.markdown("**Daily × Employee — Works**")
        st.dataframe(piv)

        # Hour x Employee pivot (sum)
        piv2 = (
            emp_hourly.groupby(["Hour", "Employee ID", "Name"], as_index=False)["Processed Count"].sum()
            .pivot_table(index="Hour", columns="Name", values="Processed Count", aggfunc="sum")
            .fillna(0)
        )
        st.markdown("**Hour × Employee — Works**")
        st.dataframe(piv2)

with tab6:
    st.subheader("Export")
    st.caption("Download aggregated tables as CSV.")

    c1, c2, c3 = st.columns(3)
    with c1:
        download_csv_button(emp_summary, "employee_summary.csv", "⬇️ Employee Summary (per employee)")
    with c2:
        download_csv_button(emp_daily, "employee_daily.csv", "⬇️ Daily Aggregates (per employee per day)")
    with c3:
        download_csv_button(emp_hourly, "employee_hourly.csv", "⬇️ Granular Hourly Records")

    st.markdown("---")
    st.markdown("**Tip:** Store these CSVs for audit and monthly reviews.")

st.caption("Built with ❤️ — drop in more CSVs anytime to refresh the dashboards.")
