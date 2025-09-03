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
# Helpers
# -----------------------------
# These names are assigned based on column position for Performance Report
REQUIRED_COLS = ["Time (day)", "Hour", "Employee ID", "Name", "Processed Count"]
COLUMN_POSITION_MAPPING = {
    0: "Time (day)",
    1: "Hour",
    2: "Employee ID",
    3: "Name",
    4: "Processed Count",
}


@st.cache_data(show_spinner=False)
def read_csv_safely(file, by_position=True) -> pd.DataFrame:
    """Robust CSV loader. Can read by position (skip header) or by header name."""
    read_params = {'header': None, 'skiprows': 1} if by_position else {'header': 0}
    
    for sep in [",", "\t", ";"]:
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                file.seek(0)
                df = pd.read_csv(file, sep=sep, encoding=enc, **read_params)
                if df.shape[1] >= 3:
                    return df
            except Exception:
                continue
    # Fallback to pandas default
    file.seek(0)
    return pd.read_csv(file, **read_params)


def coerce_schema(df: pd.DataFrame, filename: str = "") -> Tuple[pd.DataFrame, List[str]]:
    """Cleans and validates the main performance data."""
    issues = []
    
    if "Name" in df.columns:
        df = df[df["Name"] != "-"].copy()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns based on position: {missing}")

    if "Time (day)" in df.columns:
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
                df['Time (day)'] = processed_dates
            except Exception as e:
                issues.append(f"An error occurred while parsing dates: {e}")
                df['Time (day)'] = pd.NaT

        if pd.api.types.is_datetime64_any_dtype(df["Time (day)"]):
            df["Time (day)"] = df["Time (day)"].dt.date
        else:
            df["Time (day)"] = pd.NaT

    if "Hour" in df.columns:
        def _to_hour(x):
            try:
                s = str(x).strip()
                if ":" in s: s = s.split(":")[0]
                h = int(float(s))
                return h if 0 <= h <= 23 else np.nan
            except Exception: return np.nan
        df["Hour"] = df["Hour"].apply(_to_hour)

    if "Employee ID" in df.columns:
        df["Employee ID"] = df["Employee ID"].astype(str).str.replace(r'\.0$', '', regex=True)

    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str)

    if "Processed Count" in df.columns:
        df["Processed Count"] = pd.to_numeric(df["Processed Count"], errors="coerce").fillna(0)
        neg_mask = df["Processed Count"] < 0
        if neg_mask.any():
            issues.append(f"{neg_mask.sum()} rows had negative 'Processed Count' and were set to 0.")
            df.loc[neg_mask, "Processed Count"] = 0

    return df, issues

# --- NEW FUNCTIONS FOR QA ANALYSIS ---
def clean_percentage(series: pd.Series) -> pd.Series:
    """Converts a series of percentage strings (e.g., '99.5%') to floats (e.g., 0.995)."""
    series_str = series.astype(str).str.strip()
    return pd.to_numeric(series_str.str.replace('%', ''), errors='coerce') / 100

@st.cache_data(show_spinner=False)
def process_qa_data(file) -> pd.DataFrame:
    """Reads, cleans, and processes the QA accuracy report."""
    try:
        df = read_csv_safely(file, by_position=False)
        if df.empty: return pd.DataFrame()
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ QA ได้: {e}")
        return pd.DataFrame()

    # Forward-fill the 'Time (day)' column
    df['Time (day)'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df['Time (day)'].ffill(inplace=True)

    # Filter out summary rows or rows with missing identifiers
    df.dropna(subset=['Employee ID', 'Name', 'Time (day)'], inplace=True)
    df = df[df['Employee ID'] != '-'].copy()
    
    # Construct the full date from filename
    filename = getattr(file, 'name', '')
    match = re.search(r"(\d{4})-(\d{2})", filename)
    if match:
        year, month = match.groups()
        day_col = pd.to_numeric(df["Time (day)"], errors='coerce')
        df['Date'] = pd.to_datetime(
            {'year': int(year), 'month': int(month), 'day': day_col},
            errors='coerce'
        ).dt.date
    else:
        df['Date'] = pd.NaT

    df.dropna(subset=['Date'], inplace=True)
    if df.empty: return pd.DataFrame()

    # Clean the accuracy columns
    accuracy_cols = ['Total Accuracy', 'Approved Accuracy', 'Declined Accuracy', 'Kenta Accuracy']
    for col in accuracy_cols:
        if col in df.columns:
            df[col] = clean_percentage(df[col])
            
    # Select and return relevant columns
    final_cols = ['Date', 'Employee ID', 'Name'] + [c for c in accuracy_cols if c in df.columns]
    return df[final_cols]
# --- END OF NEW FUNCTIONS ---

def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [c for c in ["Time (day)", "Hour", "Employee ID", "Name"] if c in df.columns]
    if not key_cols or "Processed Count" not in df.columns: return df
    return df.groupby(key_cols, dropna=False, as_index=False)["Processed Count"].sum()


def compute_aggregates(df, count_hour_when_processed_gt_zero=True):
    df = df.copy()
    if "Hour" not in df.columns or "Time (day)" not in df.columns:
        st.warning("Cannot compute aggregates without 'Time (day)' and 'Hour' columns.")
        return df, df, df

    worked_mask = (df["Processed Count"] > 0) if count_hour_when_processed_gt_zero else ~df["Hour"].isna()
    df["WorkedHourFlag"] = worked_mask.astype(int)

    daily_keys = ["Time (day)", "Employee ID", "Name"]
    emp_daily = df.groupby(daily_keys, dropna=False).agg(
        Works=("Processed Count", "sum"),
        Hours=("WorkedHourFlag", "sum"),
    ).reset_index()
    emp_daily["WPH"] = emp_daily["Works"] / emp_daily["Hours"].replace(0, np.nan)

    emp_summary = emp_daily.groupby(["Employee ID", "Name"], dropna=False).agg(
        Days=("Time (day)", pd.Series.nunique),
        Hours=("Hours", "sum"),
        Works=("Works", "sum"),
        AvgWPH=("WPH", "mean"),
    ).reset_index()
    emp_summary["WPH"] = emp_summary["Works"] / emp_summary["Hours"].replace(0, np.nan)
    return df, emp_daily, emp_summary


def zscore(series: pd.Series) -> pd.Series:
    m, s = series.mean(), series.std(ddof=0)
    return ((series - m) / s) if s != 0 and not np.isnan(s) else pd.Series(0, index=series.index)


def anomaly_report(emp_daily: pd.DataFrame, z_threshold: float = 2.0):
    if emp_daily.empty: return emp_daily
    return emp_daily.groupby(["Employee ID", "Name"], group_keys=False).apply(
        lambda g: g.assign(WorksZ=zscore(g["Works"])).assign(IsAnomaly=lambda x: x["WorksZ"].abs() >= z_threshold)
    ).reset_index(drop=True)


def heatmap_data(df: pd.DataFrame):
    if df.empty or "Time (day)" not in df.columns or "Hour" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    tmp = df.copy()
    tmp['Time (day)'] = pd.to_datetime(tmp['Time (day)'])
    tmp["Dow"] = tmp["Time (day)"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    thr = tmp.pivot_table(index="Dow", columns="Hour", values="Processed Count", aggfunc="sum").reindex(dow_order).fillna(0)
    cov = tmp.pivot_table(index="Dow", columns="Hour", values="Employee ID", aggfunc="nunique").reindex(dow_order).fillna(0)
    return thr, cov


def add_topn_bar(df: pd.DataFrame, metric: str, top_n: int, title: str, ascending=False, format_spec=None):
    if df.empty or metric not in df.columns:
        st.info("ไม่มีข้อมูลที่จะแสดง")
        return
    show = df.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(top_n)
    fig = px.bar(show, x="Name", y=metric, hover_data=["Employee ID"], title=title, text_auto=format_spec)
    fig.update_traces(textangle=0, textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(show.reset_index(drop=True))


def kpi_block(total_works, total_hours, avg_wph, employees, days):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("งานทั้งหมด", f"{int(total_works):,}")
    c2.metric("ชั่วโมงทั้งหมด", f"{int(total_hours):,}")
    c3.metric("งานต่อชั่วโมง (เฉลี่ย)", f"{avg_wph:,.2f}" if not np.isnan(avg_wph) else "—")
    c4.metric("จำนวนพนักงาน", f"{employees:,}")
    c5.metric("จำนวนวัน", f"{days:,}")


def download_csv_button(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# -----------------------------
# UI
# -----------------------------
st.title("📈 เครื่องมือประเมินประสิทธิภาพ")

st.markdown("""
อัปโหลดไฟล์ CSV 1 ไฟล์หรือมากกว่า โดยข้อมูลต้องเรียงตามลำดับดังนี้:
- **คอลัมน์ที่ 1**: Time (day) (วันที่ของเดือน, เช่น 1, 2, ..., 31)
- **คอลัมน์ที่ 2**: Hour (ชั่วโมง, เช่น 9:00, 13:00)
- **คอลัมน์ที่ 3**: Employee ID (รหัสพนักงาน)
- **คอลัมน์ที่ 4**: Name (ชื่อพนักงาน)
- **คอลัมน์ที่ 5**: Processed Count (จำนวนงานที่ทำ)
**หมายเหตุ:** แอปจะข้ามแถวแรก (header) และอ่านข้อมูลตามลำดับคอลัมน์ที่กำหนดไว้เท่านั้น
""")

with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    st.caption("How to count a worked hour?")
    hour_count_mode = st.radio("นับชั่วโมงเมื่อ…", ["มีข้อมูล Processed Count > 0 (แนะนำ)", "มีแถวข้อมูลอยู่ (แม้ Processed Count = 0)"])
    count_when_gt_zero = hour_count_mode.startswith("มีข้อมูล")
    min_hours_threshold = st.slider("ชั่วโมงทำงานขั้นต่ำสำหรับ WPH", 1, 40, 8, 1)
    top_n = st.slider("จำนวนอันดับสูงสุดที่จะแสดง", 3, 50, 10, 1)
    st.divider()
    st.caption("ตัวกรอง (เลือกได้)")
    date_filter_on = st.checkbox("กรองตามช่วงวันที่", value=False)
    employee_search = st.text_input("ค้นหาชื่อพนักงาน (ไม่บังคับ)", value="")
    st.divider()
    st.caption("ตัวเลือกข้อมูล")
    use_sample = st.checkbox("ใช้ข้อมูลตัวอย่าง (หากยังไม่มีไฟล์)", value=False)

# -----------------------------
# Load Performance Data
# -----------------------------
st.header("1. Performance Report (ปริมาณงาน)")
uploaded_files = st.file_uploader("วางไฟล์ Performance Report ที่นี่", type=["csv"], accept_multiple_files=True)
dfs, issues_all = [], []

if use_sample and not uploaded_files:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-08-01", periods=10, freq="D").date
    rows = [[d, h, eid, nm, int(rng.poisson(8)) if rng.random() < 0.8 else 0]
            for d in dates for h in range(9, 19) for eid, nm in [("E001", "Alice"), ("E002", "Bob"), ("E003", "Chai"), ("E004", "Dao")]]
    sample = pd.DataFrame(rows).rename(columns=COLUMN_POSITION_MAPPING)
    df0, issues = coerce_schema(sample, "sample-2025-08.csv")
    dfs.append(deduplicate_rows(df0)); issues_all.extend(issues)

for f in uploaded_files:
    try:
        raw = read_csv_safely(f, by_position=True)
        raw = raw.rename(columns={k: v for k, v in COLUMN_POSITION_MAPPING.items() if k in raw.columns})
        df0, issues = coerce_schema(raw, getattr(f, 'name', ''))
        dfs.append(deduplicate_rows(df0)); issues_all.extend(issues)
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ {getattr(f, 'name', 'file')}: {e}")

if not dfs:
    st.info("👆 อัปโหลดไฟล์ Performance Report หรือเปิดใช้งาน **ใช้ข้อมูลตัวอย่าง** ในแถบด้านข้างเพื่อเริ่มต้น")
    st.stop()

df = pd.concat(dfs, ignore_index=True)
if issues_all:
    with st.expander("ข้อสังเกตเกี่ยวกับคุณภาพข้อมูล Performance"):
        st.warning("\n".join(set(issues_all)))

df.dropna(subset=['Time (day)'], inplace=True)
if df.empty:
    st.error("ไม่มีข้อมูล Performance ที่ถูกต้องเหลืออยู่หลังจากการประมวลผล โปรดตรวจสอบไฟล์ของคุณ"); st.stop()

# Main App Logic
if date_filter_on:
    min_d, max_d = df["Time (day)"].min(), df["Time (day)"].max()
    d1, d2 = st.slider("ช่วงวันที่", min_d, max_d, (min_d, max_d))
    df = df[df["Time (day)"].between(d1, d2)].copy()

if employee_search.strip():
    df = df[df["Name"].str.contains(employee_search.strip(), case=False, na=False)].copy()

emp_hourly, emp_daily, emp_summary = compute_aggregates(df, count_when_gt_zero)
total_works, total_hours = emp_daily["Works"].sum(), emp_daily["Hours"].sum()
avg_wph = total_works / total_hours if total_hours > 0 else np.nan
employees, days = emp_summary.shape[0], emp_daily["Time (day)"].nunique()

st.subheader("ภาพรวม Performance")
kpi_block(total_works, total_hours, avg_wph, employees, days)

# -----------------------------
# Tabs
# -----------------------------
tab_perf, tab_time, tab_drill, tab_anom, tab_pivot, tab_qa, tab_export = st.tabs(
    ["🏆 Leaderboards", "📅 Time Analysis", "👤 Employee Drilldown", "⚠️ Anomalies", "📊 Pivots", "🎯 วิเคราะห์ความแม่นยำ", "📥 Export"]
)

with tab_perf:
    st.subheader("Leaderboards")
    st.caption(f"ใช้เกณฑ์ชั่วโมงทำงานขั้นต่ำ **{min_hours_threshold}** ชั่วโมงสำหรับ WPH")
    if not emp_summary.empty:
        eligible = emp_summary[emp_summary["Hours"] >= min_hours_threshold].copy()
        c1, c2, c3 = st.columns(3, gap="large")
        with c1: add_topn_bar(emp_summary, "Works", top_n, "Top by Works (Total Processed)")
        with c2:
            if eligible.empty: st.info("ไม่มีพนักงานผ่านเกณฑ์สำหรับ WPH ranking")
            else: add_topn_bar(eligible, "WPH", top_n, "Top by Works per Hour (WPH)")
        with c3: add_topn_bar(emp_summary, "Hours", top_n, "Top by Hours Worked")

with tab_time:
    st.subheader("Time Analysis")
    if emp_hourly.empty: st.info("Not enough data")
    else:
        fig1 = px.line(emp_hourly.groupby("Hour", as_index=False)["Processed Count"].sum(), x="Hour", y="Processed Count", markers=True, title="Throughput by Hour of Day")
        st.plotly_chart(fig1, use_container_width=True)
        thr, cov = heatmap_data(emp_hourly)
        if not thr.empty:
            st.markdown("**Throughput Heatmap (Day × Hour)**")
            fig2 = px.imshow(thr, aspect="auto", title="Throughput", labels=dict(color="Works"))
            st.plotly_chart(fig2, use_container_width=True)

with tab_drill:
    st.subheader("Employee Drilldown")
    if emp_summary.empty: st.info("No employee data")
    else:
        names = sorted(emp_summary["Name"].unique().tolist())
        picked = st.selectbox("เลือกพนักงาน", options=names)
        if picked:
            emp_ids = emp_summary.loc[emp_summary["Name"] == picked, "Employee ID"].unique()
            emp_id = st.selectbox("Multiple IDs found, pick one:", options=emp_ids) if len(emp_ids) > 1 else emp_ids[0]
            dfd = emp_daily[(emp_daily["Name"] == picked) & (emp_daily["Employee ID"] == emp_id)].copy()
            if not dfd.empty:
                w, h = dfd["Works"].sum(), dfd["Hours"].sum()
                c1, c2, c3 = st.columns(3); c1.metric("Works", f"{w:,}"); c2.metric("Hours", f"{h:,}"); c3.metric("WPH", f"{(w/h if h>0 else 0):,.2f}")
                fig = px.line(dfd, x="Time (day)", y="Works", markers=True, title=f"Daily Works — {picked}")
                st.plotly_chart(fig, use_container_width=True)
                
with tab_anom:
    st.subheader("Anomalies")
    if emp_daily.empty or emp_daily.shape[0] < 2: st.info("No data for anomaly detection.")
    else:
        z_thr = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.5)
        anom = anomaly_report(emp_daily, z_thr)
        flagged = anom[anom["IsAnomaly"]].copy()
        st.caption("An anomaly is a day where an employee's daily 'Works' deviates from their own mean.")
        st.dataframe(flagged.sort_values(["Name", "Time (day)"]))

with tab_pivot:
    st.subheader("Pivots")
    if not emp_hourly.empty:
        piv = emp_hourly.pivot_table(index="Time (day)", columns="Name", values="Processed Count", aggfunc="sum").fillna(0)
        st.markdown("**Daily × Employee — Works**"); st.dataframe(piv)

with tab_qa:
    st.subheader("🎯 วิเคราะห์ความแม่นยำ (QA Accuracy)")
    qa_file = st.file_uploader("วางไฟล์ QA Report ที่นี่", type=["csv"])

    if qa_file:
        qa_df = process_qa_data(qa_file)
        if not qa_df.empty:
            st.metric("ค่าเฉลี่ย Total Accuracy โดยรวม", f"{qa_df['Total Accuracy'].mean():.2%}")
            
            # Leaderboard for Accuracy
            acc_summary = qa_df.groupby(["Employee ID", "Name"], as_index=False)["Total Accuracy"].mean()
            add_topn_bar(acc_summary, "Total Accuracy", top_n, "Top by Average Total Accuracy", format_spec='.2%')
            
            with st.expander("ดูข้อมูล QA ที่คลีนแล้ว"):
                st.dataframe(qa_df)
        else:
            st.warning("ไม่พบข้อมูลที่ถูกต้องในไฟล์ QA หรือไม่สามารถประมวลผลไฟล์ได้")
    else:
        st.info("อัปโหลดไฟล์ QA Report เพื่อเริ่มต้นการวิเคราะห์ความแม่นยำ")


with tab_export:
    st.subheader("Export Data")
    c1, c2, c3 = st.columns(3)
    with c1: download_csv_button(emp_summary, "employee_summary.csv", "⬇️ Employee Summary")
    with c2: download_csv_button(emp_daily, "employee_daily.csv", "⬇️ Daily Aggregates")
    with c3: download_csv_button(emp_hourly, "employee_hourly.csv", "⬇️ Granular Hourly Records")
    with st.expander("Raw data preview"): st.dataframe(df.head(100))

st.caption("สร้างด้วย ❤️ — สามารถวางไฟล์ CSV เพิ่มเติมได้ตลอดเวลาเพื่อรีเฟรชแดชบอร์ด")
