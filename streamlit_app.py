# -*- coding: utf-8 -*-
# Beijing AirWatch — Task 4 CMP7005


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing AirWatch",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design system ─────────────────────────────────────────────────────────────
NAVY      = "#0B1628"
NAVY_MID  = "#112240"
NAVY_CARD = "#16294A"
GOLD      = "#C9A84C"
GOLD_LITE = "#E8C97A"
GOLD_DIM  = "#7A6030"
WHITE     = "#F0F4FF"
MUTED     = "#8899BB"
BORDER    = "#1E3560"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {NAVY};
    color: {WHITE};
}}

.stApp {{ background-color: {NAVY}; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {NAVY_MID} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
[data-testid="stSidebar"] .stRadio label {{
    padding: 8px 12px;
    border-radius: 6px;
    transition: background 0.2s;
}}

/* Metrics */
div[data-testid="metric-container"] {{
    background: {NAVY_CARD};
    border: 1px solid {BORDER};
    border-top: 2px solid {GOLD};
    border-radius: 8px;
    padding: 16px !important;
}}
div[data-testid="metric-container"] label {{
    color: {GOLD} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {WHITE} !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {NAVY_MID};
    border-bottom: 1px solid {BORDER};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {MUTED} !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom: 2px solid {GOLD} !important;
    background: transparent !important;
}}

/* Buttons */
.stButton button {{
    background: transparent;
    border: 1px solid {GOLD};
    color: {GOLD};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    border-radius: 4px;
    transition: all 0.2s;
}}
.stButton button:hover {{
    background: {GOLD};
    color: {NAVY};
}}

/* Selectbox / multiselect */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {{
    background: {NAVY_CARD} !important;
    border: 1px solid {BORDER} !important;
    color: {WHITE} !important;
    border-radius: 6px;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

/* Divider */
hr {{ border-color: {BORDER} !important; opacity: 0.5; }}

/* Info / warning boxes */
.stAlert {{
    background: {NAVY_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-left: 3px solid {GOLD} !important;
    color: {WHITE} !important;
    border-radius: 6px;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
POLLUTANTS = ["PM2.5","PM10","SO2","NO2","CO","O3"]
MET_VARS   = ["TEMP","PRES","DEWP","WSPM","RAIN"]
STATIONS   = ["Dongsi","Guanyuan","Shunyi","Huairou"]
SCOLORS    = {"Dongsi":GOLD,"Guanyuan":"#E07B54","Shunyi":"#5B9BD5","Huairou":"#4AADA8"}
SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}

GITHUB_USER   = "anulax1114-design"
GITHUB_REPO   = "PRAC1-st20347210-DA"
GITHUB_BRANCH = "main"
STATION_FILES = {
    "Dongsi"  : "PRSA_Data_Dongsi_20130301-20170228.csv",
    "Guanyuan": "PRSA_Data_Guanyuan_20130301-20170228.csv",
    "Shunyi"  : "PRSA_Data_Shunyi_20130301-20170228.csv",
    "Huairou" : "PRSA_Data_Huairou_20130301-20170228.csv",
}
DATA_PATH = "beijing_merged_data.csv"

# ── Matplotlib theme ──────────────────────────────────────────────────────────
def apply_chart_theme():
    mpl.rcParams.update({
        "figure.facecolor":  NAVY_CARD,
        "axes.facecolor":    NAVY_CARD,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   MUTED,
        "axes.titlecolor":   WHITE,
        "axes.titlesize":    11,
        "axes.labelsize":    9,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        WHITE,
        "grid.color":        BORDER,
        "grid.alpha":        0.4,
        "legend.facecolor":  NAVY_MID,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": WHITE,
        "legend.fontsize":   8,
        "font.family":       "monospace",
    })

apply_chart_theme()

def clean_ax(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"
        dfs = []
        for station, fname in STATION_FILES.items():
            try:
                d = pd.read_csv(BASE + fname)
                d["station"] = station
                dfs.append(d)
            except Exception as e:
                st.warning(f"Could not load {station}: {e}")
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(DATA_PATH, index=False)

    df.columns = [c.strip() for c in df.columns]
    for col in POLLUTANTS + MET_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"]         = pd.to_numeric(df["year"], errors="coerce")
    df["season"]       = df["month"].map(SEASON_MAP)
    df["station_type"] = df["station"].apply(
        lambda s: "Urban" if s in ["Dongsi","Guanyuan"] else "Suburban")
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:20px 8px 24px;'>
        <p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
                  letter-spacing:3px;color:{GOLD_DIM};margin:0 0 4px;'>CMP7005 · TASK 4</p>
        <h1 style='font-family:IBM Plex Mono,monospace;font-size:1.3rem;
                   color:{GOLD};margin:0;font-weight:600;'>Beijing<br>AirWatch</h1>
        <p style='font-size:0.72rem;color:{MUTED};margin:8px 0 0;line-height:1.5;'>
            Hourly air quality monitoring<br>4 stations · 2013–2017
        </p>
    </div>
    <hr style='border-color:{BORDER};margin:0 0 16px;'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Overview", "Dataset", "Visualisation", "Model Outputs"],
        label_visibility="collapsed"
    )

    st.markdown(f"""
    <div style='position:absolute;bottom:24px;left:0;right:0;padding:0 16px;'>
        <p style='font-size:0.65rem;color:{GOLD_DIM};font-family:IBM Plex Mono,monospace;
                  letter-spacing:1px;'>st20347210</p>
    </div>
    """, unsafe_allow_html=True)

df = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"""
    <div style='padding:40px 0 32px;'>
        <p style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
                  letter-spacing:3px;color:{GOLD_DIM};margin:0 0 8px;'>
            BEIJING AIRWATCH · OVERVIEW
        </p>
        <h1 style='font-size:2.4rem;font-weight:300;color:{WHITE};
                   margin:0 0 12px;line-height:1.2;'>
            Air Quality<br><span style='color:{GOLD};font-weight:600;'>Intelligence Platform</span>
        </h1>
        <p style='font-size:0.9rem;color:{MUTED};max-width:540px;line-height:1.7;margin:0;'>
            Analysing 140,256 hourly observations across 4 Beijing monitoring stations.
            March 2013 – February 2017.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("TOTAL RECORDS",   f"{len(df):,}")
        c2.metric("MEAN PM2.5",      f"{df['PM2.5'].mean():.1f} µg/m³")
        c3.metric("PEAK PM2.5",      f"{df['PM2.5'].max():.0f} µg/m³")
        c4.metric("STATIONS",        "4")
        c5.metric("YEARS COVERED",   f"{int(df['year'].min())}–{int(df['year'].max())}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    card_style = f"background:{NAVY_CARD};border:1px solid {BORDER};border-top:2px solid {GOLD};border-radius:8px;padding:24px;"

    with col1:
        st.markdown(f"""
        <div style='{card_style}'>
            <p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
                      letter-spacing:2px;color:{GOLD};margin:0 0 10px;'>01 · DATASET</p>
            <p style='font-size:0.85rem;color:{MUTED};line-height:1.6;margin:0;'>
                Browse, filter and export the merged air quality dataset.
                Descriptive statistics and missing value diagnostics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='{card_style}'>
            <p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
                      letter-spacing:2px;color:{GOLD};margin:0 0 10px;'>02 · VISUALISATION</p>
            <p style='font-size:0.85rem;color:{MUTED};line-height:1.6;margin:0;'>
                KDE distributions, temporal trends, correlation matrices
                and seasonal station comparisons.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='{card_style}'>
            <p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
                      letter-spacing:2px;color:{GOLD};margin:0 0 10px;'>03 · MODEL OUTPUTS</p>
            <p style='font-size:0.85rem;color:{MUTED};line-height:1.6;margin:0;'>
                Random Forest diagnostics, feature importance rankings
                and live PM2.5 prediction with AQI classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Station map-style summary
    st.markdown(f"""
    <p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
              letter-spacing:2px;color:{GOLD};margin:0 0 14px;'>MONITORING STATIONS</p>
    """, unsafe_allow_html=True)

    s1,s2,s3,s4 = st.columns(4)
    station_info = {
        "Dongsi":   ("Urban",    "Central Beijing"),
        "Guanyuan": ("Urban",    "West Beijing"),
        "Shunyi":   ("Suburban", "North-East suburbs"),
        "Huairou":  ("Suburban", "North suburbs"),
    }
    for col, (stn, (stype, loc)) in zip([s1,s2,s3,s4], station_info.items()):
        col.markdown(f"""
        <div style='background:{NAVY_CARD};border:1px solid {BORDER};
                    border-left:3px solid {SCOLORS[stn]};
                    border-radius:6px;padding:14px;'>
            <p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;
                      color:{SCOLORS[stn]};margin:0 0 4px;font-weight:600;'>{stn}</p>
            <p style='font-size:0.7rem;color:{MUTED};margin:0;'>{stype} · {loc}</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    if df is None:
        st.error("Dataset unavailable."); st.stop()

    st.markdown(f"""
    <p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
              letter-spacing:3px;color:{GOLD_DIM};margin:0 0 4px;'>01 · DATASET</p>
    <h2 style='font-size:1.8rem;font-weight:300;color:{WHITE};margin:0 0 24px;'>
        Data Explorer</h2>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;letter-spacing:2px;color:{GOLD};'>FILTERS</p>", unsafe_allow_html=True)
        sel_stations = st.multiselect("Stations", STATIONS, default=STATIONS)
        sel_years    = st.multiselect("Years", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
        sel_seasons  = st.multiselect("Seasons", ["Spring","Summer","Autumn","Winter"],
                                      default=["Spring","Summer","Autumn","Winter"])

    dff = df[df["station"].isin(sel_stations) & df["year"].isin(sel_years) & df["season"].isin(sel_seasons)]
    if dff.empty:
        st.warning("No data matches the current filters."); st.stop()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("RECORDS",    f"{len(dff):,}")
    c2.metric("MEAN PM2.5", f"{dff['PM2.5'].mean():.1f}")
    c3.metric("MAX PM2.5",  f"{dff['PM2.5'].max():.0f}")
    c4.metric("STATIONS",   str(dff["station"].nunique()))
    c5.metric("DATE RANGE", f"{int(dff['year'].min())}–{int(dff['year'].max())}")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["RAW DATA","STATISTICS","MISSING VALUES"])

    with tab1:
        ca,cb = st.columns([3,1])
        search = ca.text_input("Search records", "", placeholder="Type to filter…")
        n_rows = cb.select_slider("Rows", [50,100,250,500], value=100)
        show_cols = st.multiselect("Columns", dff.columns.tolist(),
            default=["year","month","day","hour","station","station_type",
                     "PM2.5","PM10","SO2","NO2","CO","O3","TEMP","WSPM"])
        disp = dff[show_cols]
        if search:
            disp = disp[disp.apply(lambda r: r.astype(str).str.contains(search,case=False).any(),axis=1)]
        st.dataframe(disp.head(n_rows), use_container_width=True, height=420)
        st.caption(f"{min(n_rows,len(disp)):,} of {len(dff):,} records shown")
        st.download_button("↓ Export CSV",
            dff[show_cols].to_csv(index=False).encode(),
            "beijing_airwatch_export.csv","text/csv")

    with tab2:
        grp = st.selectbox("Variable group", ["All pollutants","All meteorological","Custom"])
        if grp == "All pollutants":       num_cols = POLLUTANTS
        elif grp == "All meteorological": num_cols = MET_VARS
        else: num_cols = st.multiselect("Select variables", POLLUTANTS+MET_VARS, default=["PM2.5","NO2","TEMP"])
        if num_cols:
            st.dataframe(dff[num_cols].describe().T.round(3), use_container_width=True)
            pv = st.selectbox("Per-station breakdown", num_cols, key="pv")
            st.dataframe(
                dff.groupby("station")[pv].agg(["mean","median","std","min","max"]).round(2),
                use_container_width=True)

    with tab3:
        miss = dff[POLLUTANTS+MET_VARS].isna().sum().reset_index()
        miss.columns=["Feature","Missing"]
        miss["%"]=(miss["Missing"]/len(dff)*100).round(2)
        miss=miss.sort_values("%",ascending=False)
        cm1,cm2=st.columns([2,1])
        with cm1:
            fig,ax=plt.subplots(figsize=(8,4))
            colors_m=[GOLD if p>5 else GOLD_DIM if p>1 else BORDER for p in miss["%"]]
            bars=ax.barh(miss["Feature"],miss["%"],color=colors_m,edgecolor="none",height=0.6)
            for bar,val in zip(bars,miss["%"]):
                if val>0:
                    ax.text(bar.get_width()+0.1,bar.get_y()+bar.get_height()/2,
                            f"{val:.1f}%",va="center",fontsize=8,color=MUTED)
            ax.set_xlabel("% Missing"); ax.set_title("Missing Value Rate by Feature")
            clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with cm2:
            st.dataframe(miss,use_container_width=True,height=350)
        st.info("Gas pollutants (SO₂, NO₂, CO, O₃) carry the highest proportion of missing values.")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Visualisation":
    if df is None:
        st.error("Dataset unavailable."); st.stop()

    st.markdown(f"""
    <p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
              letter-spacing:3px;color:{GOLD_DIM};margin:0 0 4px;'>02 · VISUALISATION</p>
    <h2 style='font-size:1.8rem;font-weight:300;color:{WHITE};margin:0 0 24px;'>
        Analytical Charts</h2>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;letter-spacing:2px;color:{GOLD};'>FILTERS</p>", unsafe_allow_html=True)
        sel_stations = st.multiselect("Stations", STATIONS, default=STATIONS)
        sel_years    = st.multiselect("Years", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
        sel_seasons  = st.multiselect("Seasons", ["Spring","Summer","Autumn","Winter"],
                                      default=["Spring","Summer","Autumn","Winter"])

    dff = df[df["station"].isin(sel_stations) & df["year"].isin(sel_years) & df["season"].isin(sel_seasons)]
    if dff.empty:
        st.warning("No data matches the current filters."); st.stop()

    tab1,tab2,tab3,tab4 = st.tabs(["DISTRIBUTIONS","TEMPORAL TRENDS","CORRELATIONS","SEASONAL"])

    with tab1:
        rc1,rc2 = st.columns(2)
        poll_x = rc1.selectbox("Primary variable", POLLUTANTS+MET_VARS, index=0)
        poll_y = rc2.selectbox("Secondary variable (scatter)", POLLUTANTS+MET_VARS, index=1)
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,4))
            for stn in sel_stations:
                sub = dff[dff["station"]==stn][poll_x].dropna()
                if len(sub)>10:
                    sns.kdeplot(sub,ax=ax,label=stn,fill=True,alpha=0.15,
                                linewidth=2,color=SCOLORS.get(stn))
            ax.set_xlabel(poll_x); ax.set_ylabel("Density")
            ax.set_title(f"KDE — {poll_x}")
            ax.legend(); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            fig2,ax2 = plt.subplots(figsize=(6,4))
            data_bp=[dff[dff["station"]==s][poll_x].dropna().values for s in sel_stations]
            if any(len(d)>0 for d in data_bp):
                bp=ax2.boxplot(data_bp,labels=sel_stations,patch_artist=True,
                    medianprops=dict(color=GOLD,linewidth=2),
                    whiskerprops=dict(color=MUTED),capprops=dict(color=MUTED),
                    flierprops=dict(marker="o",markersize=2,alpha=0.3,color=MUTED))
                for patch,stn in zip(bp["boxes"],sel_stations):
                    patch.set_facecolor(SCOLORS.get(stn,MUTED)); patch.set_alpha(0.6)
            ax2.set_ylabel(poll_x); ax2.set_title(f"Box Plot — {poll_x}"); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()
        st.divider()
        sn = st.slider("Scatter sample size", 1000, 20000, 5000, 1000)
        samp = dff.sample(min(sn,len(dff)),random_state=42)
        fig3,ax3 = plt.subplots(figsize=(12,4))
        for stn in sel_stations:
            s = samp[samp["station"]==stn][[poll_x,poll_y]].dropna()
            ax3.scatter(s[poll_x],s[poll_y],alpha=0.35,s=7,
                        color=SCOLORS.get(stn),label=stn,rasterized=True)
        ax3.set_xlabel(poll_x); ax3.set_ylabel(poll_y)
        ax3.set_title(f"Scatter — {poll_x} vs {poll_y}")
        ax3.legend(); clean_ax(ax3)
        st.pyplot(fig3,use_container_width=True); plt.close()

    with tab2:
        t_var  = st.selectbox("Variable", POLLUTANTS+MET_VARS, key="tvar")
        t_mode = st.radio("Time resolution",
                          ["Hourly (diurnal)","Monthly trend","Annual trend"],
                          horizontal=True)
        fig,ax = plt.subplots(figsize=(12,4.5))
        if t_mode == "Hourly (diurnal)":
            grp = dff.groupby(["hour","station"])[t_var].mean().reset_index()
            for stn in sel_stations:
                s = grp[grp["station"]==stn].sort_values("hour")
                ax.plot(s["hour"],s[t_var],color=SCOLORS.get(stn),
                        linewidth=2.5,marker="o",markersize=4,label=stn)
            ax.set_xlabel("Hour of Day"); ax.set_xticks(range(0,24,2))
        elif t_mode == "Monthly trend":
            grp = dff.groupby(["year","month","station"])[t_var].mean().reset_index()
            grp["date"] = pd.to_datetime(grp[["year","month"]].assign(day=1))
            for stn in sel_stations:
                s = grp[grp["station"]==stn].sort_values("date")
                ax.plot(s["date"],s[t_var],color=SCOLORS.get(stn),linewidth=2,label=stn)
        else:
            grp = dff.groupby(["year","station"])[t_var].mean().reset_index()
            for stn in sel_stations:
                s = grp[grp["station"]==stn]
                ax.plot(s["year"],s[t_var],color=SCOLORS.get(stn),
                        linewidth=2.5,marker="s",markersize=8,label=stn)
        if t_var == "PM2.5":
            ax.axhline(15,color=GOLD,linestyle="--",linewidth=1,alpha=0.7,label="WHO guideline 15 µg/m³")
        ax.set_ylabel(t_var); ax.set_title(f"{t_mode} — {t_var}")
        ax.legend(); clean_ax(ax)
        st.pyplot(fig,use_container_width=True); plt.close()
        st.divider()
        st.markdown(f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:{GOLD};'>HOUR × MONTH HEATMAP</p>", unsafe_allow_html=True)
        hm_stn  = st.selectbox("Station", sel_stations, key="hmstn")
        hm_data = dff[dff["station"]==hm_stn].groupby(["month","hour"])["PM2.5"].mean().unstack(fill_value=0)
        fig_h,ax_h = plt.subplots(figsize=(14,5))
        sns.heatmap(hm_data,ax=ax_h,cmap="YlOrRd",linewidths=0.2,
                    linecolor=NAVY,cbar_kws={"shrink":0.7})
        ax_h.set_xlabel("Hour"); ax_h.set_ylabel("Month")
        ax_h.set_title(f"PM2.5 Hour × Month Heatmap — {hm_stn}")
        st.pyplot(fig_h,use_container_width=True); plt.close()

    with tab3:
        corr_vars = st.multiselect("Select variables",
                                   POLLUTANTS+MET_VARS,
                                   default=POLLUTANTS+["TEMP","WSPM","DEWP"])
        if len(corr_vars) >= 2:
            corr = dff[corr_vars].corr()
            mask = np.triu(np.ones_like(corr,dtype=bool))
            fig,ax = plt.subplots(figsize=(10,8))
            cmap = sns.diverging_palette(30,220,s=80,l=45,as_cmap=True)
            sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap=cmap,
                        linewidths=0.3,ax=ax,vmin=-1,vmax=1,
                        annot_kws={"size":8},cbar_kws={"shrink":0.7})
            ax.set_title("Pearson Correlation Matrix")
            st.pyplot(fig,use_container_width=True); plt.close()

    with tab4:
        season_order = ["Spring","Summer","Autumn","Winter"]
        s_var = st.selectbox("Variable", POLLUTANTS+MET_VARS, key="svar")
        col1,col2 = st.columns(2)
        with col1:
            sd  = dff.groupby(["season","station"])[s_var].mean().reset_index()
            fig,ax = plt.subplots(figsize=(7,4.5))
            x = np.arange(len(season_order)); w = 0.2
            for i,stn in enumerate(sel_stations):
                s = sd[sd["station"]==stn].set_index("season")
                vals=[s.loc[se,s_var] if se in s.index else 0 for se in season_order]
                ax.bar(x+i*w-w*len(sel_stations)/2,vals,w,label=stn,
                       color=SCOLORS.get(stn),alpha=0.85,edgecolor="none")
            ax.set_xticks(x); ax.set_xticklabels(season_order)
            ax.set_ylabel(f"Mean {s_var}"); ax.set_title("Seasonal Station Comparison")
            ax.legend(); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            td = dff.groupby(["station_type","season"])[s_var].mean().unstack().reindex(columns=season_order)
            fig2,ax2 = plt.subplots(figsize=(7,4.5))
            x2=np.arange(len(season_order)); w2=0.35
            ax2.bar(x2-w2/2,td.loc["Urban"]    if "Urban"    in td.index else [0]*4,
                    w2,label="Urban",   color=GOLD,    alpha=0.8)
            ax2.bar(x2+w2/2,td.loc["Suburban"] if "Suburban" in td.index else [0]*4,
                    w2,label="Suburban",color=SCOLORS["Shunyi"],alpha=0.8)
            ax2.set_xticks(x2); ax2.set_xticklabels(season_order)
            ax2.set_ylabel(f"Mean {s_var}"); ax2.set_title("Urban vs Suburban")
            ax2.legend(); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL OUTPUTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Outputs":
    import joblib

    MODEL_DIR = "model_artefacts"
    AQI_BREAKS = [
        (0,    12,   "Good",                  "#00C853","#000"),
        (12.1, 35.4, "Moderate",              "#FFD600","#000"),
        (35.5, 55.4, "Unhealthy (Sensitive)", "#FF6D00","#000"),
        (55.5, 150.4,"Unhealthy",             "#D50000","#fff"),
        (150.5,250.4,"Very Unhealthy",        "#6A1B9A","#fff"),
        (250.5,9999, "Hazardous",             "#4A148C","#fff"),
    ]

    def aqi_info(v):
        for lo,hi,label,bg,fg in AQI_BREAKS:
            if lo<=v<=hi: return label,bg,fg
        return "Hazardous","#4A148C","#fff"

    @st.cache_resource
    def load_model():
        try:
            gb    = joblib.load(os.path.join(MODEL_DIR,"gb_pm25_model.pkl"))
            le    = joblib.load(os.path.join(MODEL_DIR,"wind_label_encoder.pkl"))
            feats = joblib.load(os.path.join(MODEL_DIR,"feature_names.pkl"))
            preds = pd.read_csv(os.path.join(MODEL_DIR,"test_predictions.csv"))
            return gb,le,feats,preds
        except:
            return None,None,None,None

    gb_model,le_wind,feat_names,test_preds = load_model()

    st.markdown(f"""
    <p style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
              letter-spacing:3px;color:{GOLD_DIM};margin:0 0 4px;'>03 · MODEL OUTPUTS</p>
    <h2 style='font-size:1.8rem;font-weight:300;color:{WHITE};margin:0 0 24px;'>
        Random Forest Diagnostics</h2>
    """, unsafe_allow_html=True)

    if gb_model is None:
        st.markdown(f"""
        <div style='background:{NAVY_CARD};border:1px solid {BORDER};
                    border-left:3px solid {GOLD};border-radius:8px;padding:20px;'>
            <p style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                      color:{GOLD};margin:0 0 8px;'>MODEL ARTEFACTS NOT FOUND</p>
            <p style='font-size:0.85rem;color:{MUTED};margin:0;line-height:1.6;'>
                Upload the <code>model_artefacts/</code> folder to your GitHub repository.<br>
                Required files: gb_pm25_model.pkl · wind_label_encoder.pkl ·
                feature_names.pkl · test_predictions.csv
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    actual    = test_preds["actual"]
    predicted = test_preds["predicted"]
    rmse = np.sqrt(mean_squared_error(actual,predicted))
    mae  = mean_absolute_error(actual,predicted)
    r2   = r2_score(actual,predicted)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("TEST RMSE",    f"{rmse:.2f} µg/m³")
    c2.metric("TEST MAE",     f"{mae:.2f} µg/m³")
    c3.metric("R² SCORE",     f"{r2:.4f}")
    c4.metric("TEST SAMPLES", f"{len(actual):,}")

    st.markdown("<br>",unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["DIAGNOSTICS","FEATURE IMPORTANCE","LIVE PREDICTION"])

    with tab1:
        residuals = actual.values - predicted.values
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,5))
            sc = ax.scatter(actual,predicted,alpha=0.1,s=5,rasterized=True,
                            c=np.abs(residuals),cmap="YlOrRd",vmin=0,vmax=80)
            cbar = plt.colorbar(sc,ax=ax,shrink=0.8)
            cbar.set_label("Abs Residual",color=MUTED)
            cbar.ax.yaxis.set_tick_params(color=MUTED)
            mn,mx = actual.min(),actual.max()
            ax.plot([mn,mx],[mn,mx],"--",color=GOLD,linewidth=1.5,label="Perfect fit")
            ax.set_xlabel("Actual PM2.5 (µg/m³)")
            ax.set_ylabel("Predicted PM2.5 (µg/m³)")
            ax.set_title("Actual vs Predicted"); ax.legend(); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            fig2,ax2 = plt.subplots(figsize=(6,5))
            ax2.hist(residuals,bins=80,color=GOLD_DIM,edgecolor=NAVY,linewidth=0.3,alpha=0.85)
            ax2.axvline(0,color=GOLD,linestyle="--",linewidth=1.5,label="Zero")
            ax2.axvline(np.mean(residuals),color="#E07B54",linestyle="-",
                        linewidth=1.2,label=f"Mean {np.mean(residuals):.2f}")
            ax2.set_xlabel("Residual (µg/m³)"); ax2.set_ylabel("Count")
            ax2.set_title("Residual Distribution"); ax2.legend(); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()
        fig3,ax3 = plt.subplots(figsize=(12,3.5))
        ax3.scatter(predicted,residuals,alpha=0.08,s=4,color=SCOLORS["Shunyi"],rasterized=True)
        ax3.axhline(0,   color=GOLD,  linestyle="--",linewidth=1.2)
        ax3.axhline( mae,color=MUTED, linestyle=":", linewidth=1,label=f"+MAE {mae:.1f}")
        ax3.axhline(-mae,color=MUTED, linestyle=":", linewidth=1,label=f"−MAE {mae:.1f}")
        ax3.set_xlabel("Predicted PM2.5"); ax3.set_ylabel("Residual")
        ax3.set_title("Residuals vs Predicted"); ax3.legend(); clean_ax(ax3)
        st.pyplot(fig3,use_container_width=True); plt.close()
        if "station" in test_preds.columns:
            st.divider()
            st.markdown(f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:{GOLD};'>PER-STATION PERFORMANCE</p>", unsafe_allow_html=True)
            stn_m = test_preds.groupby("station").apply(lambda g: pd.Series({
                "RMSE": np.sqrt(mean_squared_error(g["actual"],g["predicted"])),
                "MAE":  mean_absolute_error(g["actual"],g["predicted"]),
                "R2":   r2_score(g["actual"],g["predicted"]),
                "n":    len(g)
            })).reset_index()
            st.dataframe(stn_m.set_index("station").round(4),use_container_width=True)

    with tab2:
        if hasattr(gb_model,"feature_importances_"):
            imp   = pd.Series(gb_model.feature_importances_,index=feat_names).sort_values()
            top_n = st.slider("Top N features",5,len(imp),min(15,len(imp)))
            top   = imp.tail(top_n)
            fig,ax = plt.subplots(figsize=(9,max(4,top_n*0.45)))
            colors_fi = [GOLD if v>0.05 else GOLD_DIM for v in top.values]
            top.plot(kind="barh",ax=ax,color=colors_fi,edgecolor="none")
            for i,(val,_) in enumerate(zip(top.values,top.index)):
                ax.text(val+0.001,i,f"{val:.4f}",va="center",fontsize=8,color=MUTED)
            ax.set_xlabel("Importance Score")
            ax.set_title(f"Top {top_n} Feature Importances — Random Forest")
            clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()

    with tab3:
        st.markdown(f"<p style='font-size:0.85rem;color:{MUTED};margin:0 0 16px;'>Enter current sensor readings to generate a PM2.5 prediction.</p>", unsafe_allow_html=True)
        WIND_OPT = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW","calm","CALM"]
        with st.form("predict_form"):
            st.markdown(f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:{GOLD};margin:0 0 8px;'>POLLUTANT READINGS</p>", unsafe_allow_html=True)
            rc1,rc2,rc3,rc4,rc5,rc6 = st.columns(6)
            pm10 = rc1.number_input("PM10",       0.0,2000.0, 80.0,step=1.0)
            so2  = rc2.number_input("SO2",        0.0, 500.0, 15.0,step=1.0)
            no2  = rc3.number_input("NO2",        0.0, 500.0, 50.0,step=1.0)
            co   = rc4.number_input("CO",         0.0,10000.0,600.0,step=10.0)
            o3   = rc5.number_input("O3",         0.0, 500.0, 40.0,step=1.0)
            lag1 = rc6.number_input("Prev PM2.5", 0.0,1000.0, 50.0,step=1.0)
            st.markdown(f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:{GOLD};margin:8px 0;'>METEOROLOGICAL CONDITIONS</p>", unsafe_allow_html=True)
            mc1,mc2,mc3,mc4 = st.columns(4)
            temp = mc1.number_input("Temp (°C)",  -30.0, 45.0,  15.0,step=0.5)
            pres = mc2.number_input("Pressure",   980.0,1060.0,1010.0,step=0.5)
            dewp = mc3.number_input("Dew Point",  -40.0,  30.0,   5.0,step=0.5)
            wspm = mc4.number_input("Wind Speed",   0.0,  20.0,   2.0,step=0.1)
            ec1,ec2,ec3,ec4 = st.columns(4)
            rain     = ec1.number_input("Rainfall",   0.0,100.0,0.0,step=0.1)
            wd       = ec2.selectbox("Wind Direction",WIND_OPT)
            stn_type = ec3.selectbox("Station Type",  ["Urban","Suburban"])
            hour_in  = ec4.slider("Hour",0,23,12)
            month_in = st.slider("Month",1,12,6)
            submitted = st.form_submit_button("→  RUN PREDICTION",use_container_width=True)

        if submitted:
            is_urban = 1 if stn_type=="Urban" else 0
            try:    wd_enc = le_wind.transform([wd])[0]
            except: wd_enc = 0
            feat_map = {
                "PM10":pm10,"SO2":so2,"NO2":no2,"CO":co,"O3":o3,
                "TEMP":temp,"PRES":pres,"DEWP":dewp,"WSPM":wspm,"RAIN":rain,
                "wd_encoded":wd_enc,"is_urban":is_urban,
                "hour_sin": np.sin(2*np.pi*hour_in/24),
                "hour_cos": np.cos(2*np.pi*hour_in/24),
                "month_sin":np.sin(2*np.pi*month_in/12),
                "month_cos":np.cos(2*np.pi*month_in/12),
                "PM2.5_lag1":lag1
            }
            X_pred     = np.array([[feat_map[f] for f in feat_names]])
            prediction = float(max(0.0,gb_model.predict(X_pred)[0]))
            label,bg_c,fg_c = aqi_info(prediction)

            st.divider()
            r1,r2,r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div style='background:{NAVY_CARD};border:1px solid {BORDER};
                            border-top:2px solid {GOLD};border-radius:8px;
                            padding:24px;text-align:center;'>
                    <p style='font-family:IBM Plex Mono,monospace;font-size:0.55rem;
                              letter-spacing:2px;color:{GOLD_DIM};margin:0 0 8px;'>
                        PREDICTED PM2.5</p>
                    <p style='font-size:3rem;font-weight:600;color:{GOLD};margin:0;
                              font-family:IBM Plex Mono,monospace;'>{prediction:.1f}</p>
                    <p style='font-size:0.75rem;color:{MUTED};margin:4px 0 0;'>µg/m³</p>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div style='background:{NAVY_CARD};border:1px solid {BORDER};
                            border-top:2px solid {bg_c};border-radius:8px;
                            padding:24px;text-align:center;'>
                    <p style='font-family:IBM Plex Mono,monospace;font-size:0.55rem;
                              letter-spacing:2px;color:{GOLD_DIM};margin:0 0 12px;'>
                        AQI CATEGORY</p>
                    <span style='background:{bg_c};color:{fg_c};padding:6px 18px;
                                 border-radius:20px;font-weight:600;font-size:0.85rem;
                                 font-family:IBM Plex Mono,monospace;'>{label}</span>
                </div>
                """, unsafe_allow_html=True)
            with r3:
                advice = {
                    "Good":                  ("All outdoor activities safe."),
                    "Moderate":              ("Sensitive individuals limit exertion."),
                    "Unhealthy (Sensitive)": ("Sensitive groups reduce outdoor activity."),
                    "Unhealthy":             ("Everyone limit prolonged outdoor exertion."),
                    "Very Unhealthy":        ("Avoid all outdoor physical activity."),
                    "Hazardous":             ("Stay indoors. Avoid all outdoor activity."),
                }
                msg = advice.get(label,"Monitor conditions.")
                st.markdown(f"""
                <div style='background:{NAVY_CARD};border:1px solid {BORDER};
                            border-radius:8px;padding:24px;height:100%;'>
                    <p style='font-family:IBM Plex Mono,monospace;font-size:0.55rem;
                              letter-spacing:2px;color:{GOLD_DIM};margin:0 0 10px;'>
                        RECOMMENDATION</p>
                    <p style='font-size:0.85rem;color:{WHITE};margin:0;line-height:1.6;'>
                        {msg}</p>
                </div>
                """, unsafe_allow_html=True)

            # AQI scale bar
            st.markdown("<br>",unsafe_allow_html=True)
            fig_bar,ax_bar = plt.subplots(figsize=(10,1.0))
            thresholds  = [0,12,35.4,55.4,150.4,250.4,350]
            aqi_colors  = ["#00C853","#FFD600","#FF6D00","#D50000","#6A1B9A","#4A148C"]
            aqi_labels  = ["Good","Moderate","USG","Unhealthy","V.Unhealthy","Hazardous"]
            txt_colors  = ["black","black","black","white","white","white"]
            for i,(lo,hi) in enumerate(zip(thresholds[:-1],thresholds[1:])):
                ax_bar.barh(0,hi-lo,left=lo,height=0.5,
                            color=aqi_colors[i],edgecolor=NAVY,linewidth=0.5)
                ax_bar.text((lo+hi)/2,0,aqi_labels[i],ha="center",va="center",
                            fontsize=6.5,color=txt_colors[i],fontweight="bold")
            ax_bar.axvline(min(prediction,340),color=WHITE,linewidth=3,ymin=0.05,ymax=0.95)
            ax_bar.set_xlim(0,350); ax_bar.set_ylim(-0.5,0.5); ax_bar.axis("off")
            ax_bar.set_facecolor(NAVY_CARD); fig_bar.patch.set_color(NAVY_CARD)
            st.pyplot(fig_bar,use_container_width=True); plt.close()
