import streamlit as st
import pandas as pd
import sqlite3
import datetime
import os

st.set_page_config(page_title="Workplace Monitoring", layout="wide", page_icon="🕵️")

# Compute the absolute path to workplace.db relative to this file so the
# dashboard works correctly regardless of which directory it is launched from.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
DB_PATH = os.path.join(_PROJECT_ROOT, "workplace.db")

@st.cache_data(ttl=2) # Refresh data every 2 seconds
def load_data():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM events ORDER BY entry_time DESC", conn)
        return df
    except sqlite3.OperationalError:
        return pd.DataFrame()

st.title("🏢 Workplace Monitoring Dashboard")

df = load_data()

if df.empty:
    st.info("No data logged yet or database not found. Start monitor.py to generate logs.")
else:
    # Convert timestamps (keep entry_time as datetime for calculations, stringify exit_time for NaT display)
    df['entry_time_dt'] = pd.to_datetime(df['entry_time'], unit='s')
    df['exit_time_dt'] = pd.to_datetime(df['exit_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S').fillna("Active")
    
    # Calculate duration
    df['duration_seconds'] = df['exit_time'] - df['entry_time']
    df['duration_minutes'] = (df['duration_seconds'] / 60).round(2)
    
    # Layout top metrics
    col1, col2, col3 = st.columns(3)
    
    active_managers = len(df[df['exit_time'].isna()])
    total_entries_today = len(df[df['entry_time_dt'].dt.date == datetime.date.today()])
    avg_duration = df['duration_minutes'].mean()
    
    col1.metric("🟢 Currently Active Managers", active_managers)
    col2.metric("📅 Total Entries Today", total_entries_today)
    col3.metric("⏱️ Avg Session Duration (min)", f"{avg_duration:.2f}" if pd.notna(avg_duration) else "0.00")
    
    st.markdown("---")
    
    # Live Active Sessions
    st.subheader("🔴 Live Active Sessions")
    active_df = df[df['exit_time'].isna()]
    if active_df.empty:
        st.success("No active managers on the floor.")
    else:
        st.dataframe(active_df[['track_id', 'camera_id', 'role', 'entry_time_dt']], use_container_width=True)
    
    st.markdown("---")
    
    # Historical Logs with Filters
    st.subheader("📜 Historical Logs")
    
    # Filters
    f_col1, f_col2 = st.columns(2)
    
    if 'camera_id' in df.columns:
        cameras = ["All"] + list(df['camera_id'].dropna().unique())
        selected_camera = f_col1.selectbox("Filter by Camera", cameras)
        # Use .copy() to avoid pandas SettingWithCopyWarning when we mutate
        # duration_minutes below. Without this, the assignment may silently
        # modify the cached df or raise a warning in future pandas versions.
        display_df = df.copy() if selected_camera == "All" else df[df['camera_id'] == selected_camera].copy()
    else:
        display_df = df.copy()
        
    roles = ["All"] + list(display_df['role'].unique())
    selected_role = f_col2.selectbox("Filter by Role", roles)
    
    if selected_role != "All":
        display_df = display_df[display_df['role'] == selected_role]
        
    # Convert duration_minutes to string first to avoid PyArrow mixed type errors when filling NaNs
    display_df['duration_minutes'] = display_df['duration_minutes'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    
    # Display formatted table
    st.dataframe(
        display_df[['id', 'track_id', 'camera_id', 'role', 'entry_time_dt', 'exit_time_dt', 'duration_minutes']].fillna("-"), 
        use_container_width=True
    )
