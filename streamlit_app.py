# -*- coding: utf-8 -*-
# Beijing AirWatch — CMP7005 Task 4
# Warm Ivory & Rust · Card-Based Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Beijing AirWatch",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

IVORY      = "#FAF7F2"
IVORY_DARK = "#F0EBE1"
IVORY_MID  = "#E8E0D4"
RUST       = "#B5451B"
RUST_LIGHT = "#D4603A"
RUST_PALE  = "#F2D5C8"
CHARCOAL   = "#2C2416"
BROWN      = "#6B4F3A"
MUTED      = "#9C8270"
BORDER     = "#D9CFC4"
WHITE      = "#FFFFFF"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {IVORY}; color: {CHARCOAL}; }}
.stApp {{ background-color: {IVORY}; }}
[data-testid="stSidebar"] {{ background-color: {CHARCOAL} !important; border-right: none; }}
[data-testid="stSidebar"] * {{ color: {IVORY} !important; }}
div[data-testid="metric-container"] {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 20px !important; box-shadow: 0 1px 4px rgba(44,36,22,0.06); }}
div[data-testid="metric-container"] label {{ color: {MUTED} !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.58rem !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {CHARCOAL} !important; font-size: 1.9rem !important; font-weight: 600 !important; font-family: 'Playfair Display', serif !important; }}
.stTabs [data-baseweb="tab-list"] {{ background: transparent; border-bottom: 2px solid {BORDER}; gap: 0; }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED} !important; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 1px; padding: 10px 20px; border-bottom: 2px solid transparent; margin-bottom: -2px; background: transparent !important; }}
.stTabs [aria-selected="true"] {{ color: {RUST} !important; border-bottom: 2px solid {RUST} !important; }}
.stButton button {{ background: {RUST}; border: none; color: {WHITE}; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 1px; border-radius: 6px; }}
.stButton button:hover {{ background: {RUST_LIGHT}; color: {WHITE}; }}
hr {{ border-color: {BORDER} !important; opacity: 0.6; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

POLLUTANTS = ["PM2.5","PM10","SO2","NO2","CO","O3"]
MET_VARS   = ["TEMP","PRES","DEWP","WSPM","RAIN"]
STATIONS   = ["Dongsi","Guanyuan","Shunyi","Huairou"]
SCOLORS    = {"Dongsi":RUST,"Guanyuan":"#C47B2B","Shunyi":"#4A7FA5","Huairou":"#5A9E8F"}
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

def apply_chart_theme():
    mpl.rcParams.update({
        "figure.facecolor": WHITE, "axes.facecolor": WHITE,
        "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
        "axes.titlecolor": CHARCOAL, "axes.titlesize": 11,
        "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "text.color": CHARCOAL,
        "grid.color": BORDER, "grid.alpha": 0.5,
        "legend.facecolor": WHITE, "legend.edgecolor": BORDER, "legend.fontsize": 8,
    })
apply_chart_theme()

def clean_ax(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(BORDER)

@st.cache_data(show_spinner="Fetching dataset...")
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
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["season"] = df["month"].map(SEASON_MAP)
    df["station_type"] = df["station"].apply(lambda s: "Urban" if s in ["Dongsi","Guanyuan"] else "Suburban")
    return df

with st.sidebar:
    st.markdown(f"""
    <div style='padding:28px 16px 20px;'>
        <p style='font-family:JetBrains Mono,monospace;font-size:0.55rem;letter-spacing:3px;color:{BROWN};margin:0 0 6px;'>CMP7005 · TASK 4</p>
        <h1 style='font-family:Playfair Display,serif;font-size:1.5rem;color:{IVORY};margin:0;line-height:1.3;'>Beijing<br>AirWatch</h1>
        <div style='width:32px;height:2px;background:{RUST};margin:12px 0;'></div>
        <p style='font-size:0.72rem;color:{MUTED};margin:0;line-height:1.6;'>4 stations · 2013–2017<br>140,256 hourly records</p>
    </div>
    <hr style='border-color:{BROWN};margin:0 0 16px;'/>
    """, unsafe_allow_html=True)
    page = st.radio("Navigation", ["Overview","Dataset","Visualisation","Model Outputs"], label_visibility="collapsed")
    st.markdown(f"""
    <div style='padding:16px;margin-top:16px;background:rgba(255,255,255,0.04);border-radius:8px;border:1px solid {BROWN};'>
        <p style='font-family:JetBrains Mono,monospace;font-size:0.55rem;color:{MUTED};margin:0 0 8px;letter-spacing:1px;'>STATIONS</p>
        {"".join([f"<p style='font-size:0.72rem;color:{SCOLORS[s]};margin:4px 0;'>● {s}</p>" for s in STATIONS])}
    </div>
    <p style='font-size:0.6rem;color:{BROWN};font-family:JetBrains Mono,monospace;padding:16px;margin:0;letter-spacing:1px;'>st20347210</p>
    """, unsafe_allow_html=True)

df = load_data()

def section_header(tag, title, subtitle=""):
    sub_html = f"<p style='font-size:0.88rem;color:{MUTED};margin:8px 0 0;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='padding:32px 0 24px;'>
        <p style='font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:3px;color:{RUST};margin:0 0 6px;text-transform:uppercase;'>{tag}</p>
        <h2 style='font-family:Playfair Display,serif;font-size:2rem;font-weight:400;color:{CHARCOAL};margin:0;line-height:1.2;'>{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

# ══ OVERVIEW ══════════════════════════════════════════════════════════════════
if page == "Overview":
    section_header("Beijing AirWatch · Overview", "Air Quality Intelligence",
                   "Monitoring PM2.5 and co-pollutants across Beijing's urban and suburban stations.")
    if df is not None:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Records",  f"{len(df):,}")
        c2.metric("Mean PM2.5",     f"{df['PM2.5'].mean():.1f} µg/m³")
        c3.metric("Peak PM2.5",     f"{df['PM2.5'].max():.0f} µg/m³")
        c4.metric("Stations",       "4")
        c5.metric("Years Covered",  f"{int(df['year'].min())}–{int(df['year'].max())}")
    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    features = [
        ("01 · Dataset",       RUST,              "Browse and filter 140,256 records. Export subsets, view descriptive statistics and explore missing value patterns."),
        ("02 · Visualisation", "#C47B2B",          "KDE distributions, temporal trend lines, Pearson correlation matrices and seasonal station comparisons."),
        ("03 · Model Outputs", SCOLORS["Shunyi"],  "Random Forest diagnostics, feature importance rankings and a live PM2.5 predictor with AQI classification."),
    ]
    for col,(title,accent,desc) in zip([col1,col2,col3], features):
        with col:
            st.markdown(f"""
            <div style='background:{WHITE};border:1px solid {BORDER};border-radius:12px;padding:24px;
                        box-shadow:0 2px 8px rgba(44,36,22,0.05);min-height:160px;'>
                <div style='width:28px;height:3px;background:{accent};margin-bottom:14px;'></div>
                <p style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:{accent};margin:0 0 10px;letter-spacing:1px;'>{title}</p>
                <p style='font-size:0.83rem;color:{MUTED};line-height:1.7;margin:0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:2px;color:{RUST};text-transform:uppercase;margin:0 0 14px;'>Monitoring Network</p>", unsafe_allow_html=True)
    s_cols = st.columns(4)
    station_meta = {
        "Dongsi":   ("Urban",    "Central Beijing"),
        "Guanyuan": ("Urban",    "West Beijing"),
        "Shunyi":   ("Suburban", "North-East Suburbs"),
        "Huairou":  ("Suburban", "Northern Suburbs"),
    }
    for col,(stn,(stype,loc)) in zip(s_cols, station_meta.items()):
        pm_mean = f"{df[df['station']==stn]['PM2.5'].mean():.1f}" if df is not None else "—"
        with col:
            st.markdown(f"""
            <div style='background:{WHITE};border:1px solid {BORDER};border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(44,36,22,0.05);'>
                <div style='display:flex;align-items:center;margin-bottom:12px;'>
                    <div style='width:10px;height:10px;border-radius:50%;background:{SCOLORS[stn]};margin-right:10px;'></div>
                    <p style='font-family:Playfair Display,serif;font-size:1rem;color:{CHARCOAL};margin:0;font-weight:600;'>{stn}</p>
                </div>
                <p style='font-size:0.72rem;color:{MUTED};margin:0 0 4px;'>{stype} · {loc}</p>
                <p style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{RUST};margin:8px 0 0;'>avg PM2.5 {pm_mean} µg/m³</p>
            </div>
            """, unsafe_allow_html=True)

# ══ DATASET ═══════════════════════════════════════════════════════════════════
elif page == "Dataset":
    if df is None: st.error("Dataset unavailable."); st.stop()
    section_header("01 · Dataset", "Data Explorer", "Filter, inspect and export the merged Beijing air quality dataset.")
    with st.sidebar:
        st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:2px;color:{RUST};text-transform:uppercase;'>Filters</p>", unsafe_allow_html=True)
        sel_stations = st.multiselect("Stations", STATIONS, default=STATIONS)
        sel_years    = st.multiselect("Years", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
        sel_seasons  = st.multiselect("Seasons", ["Spring","Summer","Autumn","Winter"], default=["Spring","Summer","Autumn","Winter"])
    dff = df[df["station"].isin(sel_stations) & df["year"].isin(sel_years) & df["season"].isin(sel_seasons)]
    if dff.empty: st.warning("No data matches the current filters."); st.stop()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Records",    f"{len(dff):,}")
    c2.metric("Mean PM2.5", f"{dff['PM2.5'].mean():.1f}")
    c3.metric("Max PM2.5",  f"{dff['PM2.5'].max():.0f}")
    c4.metric("Stations",   str(dff["station"].nunique()))
    c5.metric("Date Range", f"{int(dff['year'].min())}–{int(dff['year'].max())}")
    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["RAW DATA","STATISTICS","MISSING VALUES"])
    with tab1:
        ca,cb = st.columns([3,1])
        search = ca.text_input("Search","",placeholder="Filter records...")
        n_rows = cb.select_slider("Rows",[50,100,250,500],value=100)
        show_cols = st.multiselect("Columns", dff.columns.tolist(),
            default=["year","month","day","hour","station","station_type","PM2.5","PM10","SO2","NO2","CO","O3","TEMP","WSPM"])
        disp = dff[show_cols]
        if search:
            disp = disp[disp.apply(lambda r: r.astype(str).str.contains(search,case=False).any(),axis=1)]
        st.dataframe(disp.head(n_rows), use_container_width=True, height=420)
        st.caption(f"{min(n_rows,len(disp)):,} of {len(dff):,} records")
        st.download_button("Export CSV", dff[show_cols].to_csv(index=False).encode(), "beijing_airwatch.csv","text/csv")
    with tab2:
        grp = st.selectbox("Variable group",["All pollutants","All meteorological","Custom"])
        if grp=="All pollutants": num_cols=POLLUTANTS
        elif grp=="All meteorological": num_cols=MET_VARS
        else: num_cols=st.multiselect("Select",POLLUTANTS+MET_VARS,default=["PM2.5","NO2","TEMP"])
        if num_cols:
            st.dataframe(dff[num_cols].describe().T.round(3),use_container_width=True)
            pv=st.selectbox("Per-station breakdown",num_cols,key="pv")
            st.dataframe(dff.groupby("station")[pv].agg(["mean","median","std","min","max"]).round(2),use_container_width=True)
    with tab3:
        miss=dff[POLLUTANTS+MET_VARS].isna().sum().reset_index()
        miss.columns=["Feature","Missing"]; miss["%"]=(miss["Missing"]/len(dff)*100).round(2)
        miss=miss.sort_values("%",ascending=False)
        cm1,cm2=st.columns([2,1])
        with cm1:
            fig,ax=plt.subplots(figsize=(8,4))
            colors_m=[RUST if p>5 else "#C47B2B" if p>1 else IVORY_MID for p in miss["%"]]
            bars=ax.barh(miss["Feature"],miss["%"],color=colors_m,edgecolor="none",height=0.55)
            for bar,val in zip(bars,miss["%"]):
                if val>0: ax.text(bar.get_width()+0.1,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va="center",fontsize=8,color=MUTED)
            ax.set_xlabel("% Missing"); ax.set_title("Missing Value Rate by Feature"); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with cm2:
            st.dataframe(miss,use_container_width=True,height=350)
        st.info("Gas pollutants (SO2, NO2, CO, O3) carry the highest proportion of missing values.")

# ══ VISUALISATION ═════════════════════════════════════════════════════════════
elif page == "Visualisation":
    if df is None: st.error("Dataset unavailable."); st.stop()
    section_header("02 · Visualisation","Analytical Charts","Explore distributions, trends, correlations and seasonal patterns.")
    with st.sidebar:
        st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:0.58rem;letter-spacing:2px;color:{RUST};text-transform:uppercase;'>Filters</p>", unsafe_allow_html=True)
        sel_stations = st.multiselect("Stations",STATIONS,default=STATIONS)
        sel_years    = st.multiselect("Years",sorted(df["year"].unique()),default=sorted(df["year"].unique()))
        sel_seasons  = st.multiselect("Seasons",["Spring","Summer","Autumn","Winter"],default=["Spring","Summer","Autumn","Winter"])
    dff = df[df["station"].isin(sel_stations)&df["year"].isin(sel_years)&df["season"].isin(sel_seasons)]
    if dff.empty: st.warning("No data matches the current filters."); st.stop()
    tab1,tab2,tab3,tab4 = st.tabs(["DISTRIBUTIONS","TEMPORAL TRENDS","CORRELATIONS","SEASONAL"])
    with tab1:
        rc1,rc2=st.columns(2)
        poll_x=rc1.selectbox("Primary variable",POLLUTANTS+MET_VARS,index=0)
        poll_y=rc2.selectbox("Secondary variable",POLLUTANTS+MET_VARS,index=1)
        col1,col2=st.columns(2)
        with col1:
            fig,ax=plt.subplots(figsize=(6,4))
            for stn in sel_stations:
                sub=dff[dff["station"]==stn][poll_x].dropna()
                if len(sub)>10: sns.kdeplot(sub,ax=ax,label=stn,fill=True,alpha=0.12,linewidth=2,color=SCOLORS.get(stn))
            ax.set_xlabel(poll_x); ax.set_ylabel("Density"); ax.set_title(f"KDE — {poll_x}"); ax.legend(); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            fig2,ax2=plt.subplots(figsize=(6,4))
            data_bp=[dff[dff["station"]==s][poll_x].dropna().values for s in sel_stations]
            if any(len(d)>0 for d in data_bp):
                bp=ax2.boxplot(data_bp,labels=sel_stations,patch_artist=True,
                    medianprops=dict(color=RUST,linewidth=2),
                    whiskerprops=dict(color=MUTED),capprops=dict(color=MUTED),
                    flierprops=dict(marker="o",markersize=2,alpha=0.3,color=MUTED))
                for patch,stn in zip(bp["boxes"],sel_stations):
                    patch.set_facecolor(SCOLORS.get(stn,MUTED)); patch.set_alpha(0.55)
            ax2.set_ylabel(poll_x); ax2.set_title(f"Box Plot — {poll_x}"); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()
        st.divider()
        sn=st.slider("Sample size",1000,20000,5000,1000)
        samp=dff.sample(min(sn,len(dff)),random_state=42)
        fig3,ax3=plt.subplots(figsize=(12,4))
        for stn in sel_stations:
            s=samp[samp["station"]==stn][[poll_x,poll_y]].dropna()
            ax3.scatter(s[poll_x],s[poll_y],alpha=0.3,s=8,color=SCOLORS.get(stn),label=stn,rasterized=True)
        ax3.set_xlabel(poll_x); ax3.set_ylabel(poll_y); ax3.set_title(f"Scatter — {poll_x} vs {poll_y}"); ax3.legend(); clean_ax(ax3)
        st.pyplot(fig3,use_container_width=True); plt.close()
    with tab2:
        t_var=st.selectbox("Variable",POLLUTANTS+MET_VARS,key="tvar")
        t_mode=st.radio("Time resolution",["Hourly (diurnal)","Monthly trend","Annual trend"],horizontal=True)
        fig,ax=plt.subplots(figsize=(12,4.5))
        if t_mode=="Hourly (diurnal)":
            grp=dff.groupby(["hour","station"])[t_var].mean().reset_index()
            for stn in sel_stations:
                s=grp[grp["station"]==stn].sort_values("hour")
                ax.plot(s["hour"],s[t_var],color=SCOLORS.get(stn),linewidth=2.5,marker="o",markersize=4,label=stn)
            ax.set_xlabel("Hour of Day"); ax.set_xticks(range(0,24,2))
        elif t_mode=="Monthly trend":
            grp=dff.groupby(["year","month","station"])[t_var].mean().reset_index()
            grp["date"]=pd.to_datetime(grp[["year","month"]].assign(day=1))
            for stn in sel_stations:
                s=grp[grp["station"]==stn].sort_values("date")
                ax.plot(s["date"],s[t_var],color=SCOLORS.get(stn),linewidth=2,label=stn)
        else:
            grp=dff.groupby(["year","station"])[t_var].mean().reset_index()
            for stn in sel_stations:
                s=grp[grp["station"]==stn]
                ax.plot(s["year"],s[t_var],color=SCOLORS.get(stn),linewidth=2.5,marker="s",markersize=8,label=stn)
        if t_var=="PM2.5": ax.axhline(15,color=RUST,linestyle="--",linewidth=1,alpha=0.6,label="WHO 15 µg/m3")
        ax.set_ylabel(t_var); ax.set_title(f"{t_mode} — {t_var}"); ax.legend(); clean_ax(ax)
        st.pyplot(fig,use_container_width=True); plt.close()
        st.divider()
        st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:0.62rem;letter-spacing:1px;color:{RUST};'>HOUR x MONTH HEATMAP</p>",unsafe_allow_html=True)
        hm_stn=st.selectbox("Station",sel_stations,key="hmstn")
        hm_data=dff[dff["station"]==hm_stn].groupby(["month","hour"])["PM2.5"].mean().unstack(fill_value=0)
        fig_h,ax_h=plt.subplots(figsize=(14,5))
        sns.heatmap(hm_data,ax=ax_h,cmap="YlOrRd",linewidths=0.2,linecolor=IVORY,cbar_kws={"shrink":0.7})
        ax_h.set_xlabel("Hour"); ax_h.set_ylabel("Month"); ax_h.set_title(f"PM2.5 Hour x Month — {hm_stn}")
        st.pyplot(fig_h,use_container_width=True); plt.close()
    with tab3:
        corr_vars=st.multiselect("Variables",POLLUTANTS+MET_VARS,default=POLLUTANTS+["TEMP","WSPM","DEWP"])
        if len(corr_vars)>=2:
            corr=dff[corr_vars].corr(); mask=np.triu(np.ones_like(corr,dtype=bool))
            fig,ax=plt.subplots(figsize=(10,8))
            cmap=sns.diverging_palette(15,220,s=70,l=50,as_cmap=True)
            sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap=cmap,linewidths=0.3,ax=ax,vmin=-1,vmax=1,annot_kws={"size":8},cbar_kws={"shrink":0.7})
            ax.set_title("Pearson Correlation Matrix"); st.pyplot(fig,use_container_width=True); plt.close()
    with tab4:
        season_order=["Spring","Summer","Autumn","Winter"]
        s_var=st.selectbox("Variable",POLLUTANTS+MET_VARS,key="svar")
        col1,col2=st.columns(2)
        with col1:
            sd=dff.groupby(["season","station"])[s_var].mean().reset_index()
            fig,ax=plt.subplots(figsize=(7,4.5)); x=np.arange(len(season_order)); w=0.2
            for i,stn in enumerate(sel_stations):
                s=sd[sd["station"]==stn].set_index("season")
                vals=[s.loc[se,s_var] if se in s.index else 0 for se in season_order]
                ax.bar(x+i*w-w*len(sel_stations)/2,vals,w,label=stn,color=SCOLORS.get(stn),alpha=0.8,edgecolor="none")
            ax.set_xticks(x); ax.set_xticklabels(season_order); ax.set_ylabel(f"Mean {s_var}"); ax.set_title("Seasonal Station Comparison"); ax.legend(); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            td=dff.groupby(["station_type","season"])[s_var].mean().unstack().reindex(columns=season_order)
            fig2,ax2=plt.subplots(figsize=(7,4.5)); x2=np.arange(len(season_order)); w2=0.35
            ax2.bar(x2-w2/2,td.loc["Urban"] if "Urban" in td.index else [0]*4,w2,label="Urban",color=RUST,alpha=0.8)
            ax2.bar(x2+w2/2,td.loc["Suburban"] if "Suburban" in td.index else [0]*4,w2,label="Suburban",color=SCOLORS["Shunyi"],alpha=0.8)
            ax2.set_xticks(x2); ax2.set_xticklabels(season_order); ax2.set_ylabel(f"Mean {s_var}"); ax2.set_title("Urban vs Suburban"); ax2.legend(); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()

# ══ MODEL OUTPUTS ═════════════════════════════════════════════════════════════
elif page == "Model Outputs":
    import joblib
    MODEL_DIR="model_artefacts"
    AQI_BREAKS=[(0,12,"Good","#00C853","#000"),(12.1,35.4,"Moderate","#FFD600","#000"),
                (35.5,55.4,"Unhealthy (Sensitive)","#FF6D00","#000"),(55.5,150.4,"Unhealthy","#D50000","#fff"),
                (150.5,250.4,"Very Unhealthy","#6A1B9A","#fff"),(250.5,9999,"Hazardous","#4A148C","#fff")]
    def aqi_info(v):
        for lo,hi,label,bg,fg in AQI_BREAKS:
            if lo<=v<=hi: return label,bg,fg
        return "Hazardous","#4A148C","#fff"
    @st.cache_resource
    def load_model():
        try:
            gb=joblib.load(os.path.join(MODEL_DIR,"gb_pm25_model.pkl"))
            le=joblib.load(os.path.join(MODEL_DIR,"wind_label_encoder.pkl"))
            feats=joblib.load(os.path.join(MODEL_DIR,"feature_names.pkl"))
            preds=pd.read_csv(os.path.join(MODEL_DIR,"test_predictions.csv"))
            return gb,le,feats,preds
        except: return None,None,None,None
    gb_model,le_wind,feat_names,test_preds=load_model()
    section_header("03 · Model Outputs","Random Forest Diagnostics","Model performance, feature importance and live PM2.5 prediction.")
    if gb_model is None:
        st.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-left:4px solid {RUST};border-radius:10px;padding:24px;'>
        <p style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:{RUST};margin:0 0 8px;letter-spacing:1px;'>MODEL ARTEFACTS NOT FOUND</p>
        <p style='font-size:0.85rem;color:{MUTED};margin:0;line-height:1.7;'>Upload the <code>model_artefacts/</code> folder to your GitHub repository.<br>
        Required: gb_pm25_model.pkl · wind_label_encoder.pkl · feature_names.pkl · test_predictions.csv</p></div>""", unsafe_allow_html=True)
        st.stop()
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    actual=test_preds["actual"]; predicted=test_preds["predicted"]
    rmse=np.sqrt(mean_squared_error(actual,predicted)); mae=mean_absolute_error(actual,predicted); r2=r2_score(actual,predicted)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Test RMSE",f"{rmse:.2f} µg/m³"); c2.metric("Test MAE",f"{mae:.2f} µg/m³")
    c3.metric("R2 Score",f"{r2:.4f}"); c4.metric("Test Samples",f"{len(actual):,}")
    st.markdown("<br>",unsafe_allow_html=True)
    tab1,tab2,tab3=st.tabs(["DIAGNOSTICS","FEATURE IMPORTANCE","LIVE PREDICTION"])
    with tab1:
        residuals=actual.values-predicted.values
        col1,col2=st.columns(2)
        with col1:
            fig,ax=plt.subplots(figsize=(6,5))
            sc=ax.scatter(actual,predicted,alpha=0.1,s=5,rasterized=True,c=np.abs(residuals),cmap="YlOrRd",vmin=0,vmax=80)
            plt.colorbar(sc,ax=ax,shrink=0.8,label="Abs Residual")
            mn,mx=actual.min(),actual.max()
            ax.plot([mn,mx],[mn,mx],"--",color=RUST,linewidth=1.5,label="Perfect fit")
            ax.set_xlabel("Actual PM2.5 (µg/m³)"); ax.set_ylabel("Predicted PM2.5 (µg/m³)"); ax.set_title("Actual vs Predicted"); ax.legend(); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            fig2,ax2=plt.subplots(figsize=(6,5))
            ax2.hist(residuals,bins=80,color=RUST_PALE,edgecolor=BORDER,linewidth=0.5,alpha=0.9)
            ax2.axvline(0,color=RUST,linestyle="--",linewidth=1.5,label="Zero")
            ax2.axvline(np.mean(residuals),color=BROWN,linestyle="-",linewidth=1.2,label=f"Mean {np.mean(residuals):.2f}")
            ax2.set_xlabel("Residual (µg/m³)"); ax2.set_ylabel("Count"); ax2.set_title("Residual Distribution"); ax2.legend(); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()
        fig3,ax3=plt.subplots(figsize=(12,3.5))
        ax3.scatter(predicted,residuals,alpha=0.08,s=4,color=SCOLORS["Shunyi"],rasterized=True)
        ax3.axhline(0,color=RUST,linestyle="--",linewidth=1.2)
        ax3.axhline(mae,color=MUTED,linestyle=":",linewidth=1,label=f"+MAE {mae:.1f}")
        ax3.axhline(-mae,color=MUTED,linestyle=":",linewidth=1,label=f"-MAE {mae:.1f}")
        ax3.set_xlabel("Predicted PM2.5"); ax3.set_ylabel("Residual"); ax3.set_title("Residuals vs Predicted"); ax3.legend(); clean_ax(ax3)
        st.pyplot(fig3,use_container_width=True); plt.close()
        if "station" in test_preds.columns:
            st.divider()
            st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:{RUST};letter-spacing:1px;'>PER-STATION PERFORMANCE</p>",unsafe_allow_html=True)
            stn_m=test_preds.groupby("station").apply(lambda g: pd.Series({
                "RMSE":np.sqrt(mean_squared_error(g["actual"],g["predicted"])),
                "MAE":mean_absolute_error(g["actual"],g["predicted"]),
                "R2":r2_score(g["actual"],g["predicted"]),"n":len(g)
            })).reset_index()
            st.dataframe(stn_m.set_index("station").round(4),use_container_width=True)
    with tab2:
        if hasattr(gb_model,"feature_importances_"):
            imp=pd.Series(gb_model.feature_importances_,index=feat_names).sort_values()
            top_n=st.slider("Top N features",5,len(imp),min(15,len(imp))); top=imp.tail(top_n)
            fig,ax=plt.subplots(figsize=(9,max(4,top_n*0.45)))
            colors_fi=[RUST if v>0.05 else "#C47B2B" if v>0.01 else IVORY_MID for v in top.values]
            top.plot(kind="barh",ax=ax,color=colors_fi,edgecolor="none")
            for i,(val,_) in enumerate(zip(top.values,top.index)):
                ax.text(val+0.001,i,f"{val:.4f}",va="center",fontsize=8,color=MUTED)
            ax.set_xlabel("Importance Score"); ax.set_title(f"Top {top_n} Feature Importances"); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
    with tab3:
        st.markdown(f"<p style='font-size:0.85rem;color:{MUTED};margin:0 0 16px;'>Enter current sensor readings to generate a PM2.5 prediction.</p>",unsafe_allow_html=True)
        WIND_OPT=["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW","calm","CALM"]
        with st.form("predict_form"):
            st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:{RUST};letter-spacing:1px;margin:0 0 8px;'>POLLUTANT READINGS</p>",unsafe_allow_html=True)
            rc1,rc2,rc3,rc4,rc5,rc6=st.columns(6)
            pm10=rc1.number_input("PM10",0.0,2000.0,80.0,step=1.0); so2=rc2.number_input("SO2",0.0,500.0,15.0,step=1.0)
            no2=rc3.number_input("NO2",0.0,500.0,50.0,step=1.0); co=rc4.number_input("CO",0.0,10000.0,600.0,step=10.0)
            o3=rc5.number_input("O3",0.0,500.0,40.0,step=1.0); lag1=rc6.number_input("Prev PM2.5",0.0,1000.0,50.0,step=1.0)
            st.markdown(f"<p style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:{RUST};letter-spacing:1px;margin:8px 0;'>METEOROLOGICAL CONDITIONS</p>",unsafe_allow_html=True)
            mc1,mc2,mc3,mc4=st.columns(4)
            temp=mc1.number_input("Temp (C)",-30.0,45.0,15.0,step=0.5); pres=mc2.number_input("Pressure",980.0,1060.0,1010.0,step=0.5)
            dewp=mc3.number_input("Dew Point",-40.0,30.0,5.0,step=0.5); wspm=mc4.number_input("Wind Speed",0.0,20.0,2.0,step=0.1)
            ec1,ec2,ec3,ec4=st.columns(4)
            rain=ec1.number_input("Rainfall",0.0,100.0,0.0,step=0.1); wd=ec2.selectbox("Wind Dir",WIND_OPT)
            stn_type=ec3.selectbox("Station Type",["Urban","Suburban"]); hour_in=ec4.slider("Hour",0,23,12)
            month_in=st.slider("Month",1,12,6)
            submitted=st.form_submit_button("Run Prediction",use_container_width=True)
        if submitted:
            is_urban=1 if stn_type=="Urban" else 0
            try: wd_enc=le_wind.transform([wd])[0]
            except: wd_enc=0
            feat_map={"PM10":pm10,"SO2":so2,"NO2":no2,"CO":co,"O3":o3,"TEMP":temp,"PRES":pres,"DEWP":dewp,"WSPM":wspm,"RAIN":rain,
                      "wd_encoded":wd_enc,"is_urban":is_urban,"hour_sin":np.sin(2*np.pi*hour_in/24),"hour_cos":np.cos(2*np.pi*hour_in/24),
                      "month_sin":np.sin(2*np.pi*month_in/12),"month_cos":np.cos(2*np.pi*month_in/12),"PM2.5_lag1":lag1}
            X_pred=np.array([[feat_map[f] for f in feat_names]])
            prediction=float(max(0.0,gb_model.predict(X_pred)[0]))
            label,bg_c,fg_c=aqi_info(prediction)
            st.divider()
            r1,r2,r3=st.columns(3)
            with r1:
                st.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-top:3px solid {RUST};border-radius:12px;padding:28px;text-align:center;box-shadow:0 2px 8px rgba(44,36,22,0.08);'>
                <p style='font-family:JetBrains Mono,monospace;font-size:0.55rem;letter-spacing:2px;color:{MUTED};margin:0 0 10px;'>PREDICTED PM2.5</p>
                <p style='font-family:Playfair Display,serif;font-size:3.2rem;font-weight:600;color:{RUST};margin:0;line-height:1;'>{prediction:.1f}</p>
                <p style='font-size:0.78rem;color:{MUTED};margin:6px 0 0;'>µg/m³</p></div>""",unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-top:3px solid {bg_c};border-radius:12px;padding:28px;text-align:center;box-shadow:0 2px 8px rgba(44,36,22,0.08);'>
                <p style='font-family:JetBrains Mono,monospace;font-size:0.55rem;letter-spacing:2px;color:{MUTED};margin:0 0 14px;'>AQI CATEGORY</p>
                <span style='background:{bg_c};color:{fg_c};padding:7px 20px;border-radius:20px;font-size:0.85rem;font-weight:600;font-family:JetBrains Mono,monospace;'>{label}</span>
                </div>""",unsafe_allow_html=True)
            with r3:
                advice={"Good":"All outdoor activities are safe.","Moderate":"Sensitive individuals should limit exertion.",
                        "Unhealthy (Sensitive)":"Sensitive groups should reduce outdoor activity.","Unhealthy":"Everyone should limit prolonged outdoor exertion.",
                        "Very Unhealthy":"Avoid all outdoor physical activity.","Hazardous":"Stay indoors and avoid all outdoor activity."}
                msg=advice.get(label,"Monitor conditions closely.")
                st.markdown(f"""<div style='background:{WHITE};border:1px solid {BORDER};border-radius:12px;padding:28px;box-shadow:0 2px 8px rgba(44,36,22,0.08);'>
                <p style='font-family:JetBrains Mono,monospace;font-size:0.55rem;letter-spacing:2px;color:{MUTED};margin:0 0 10px;'>RECOMMENDATION</p>
                <p style='font-size:0.88rem;color:{CHARCOAL};margin:0;line-height:1.7;'>{msg}</p></div>""",unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            fig_bar,ax_bar=plt.subplots(figsize=(10,0.9))
            thresholds=[0,12,35.4,55.4,150.4,250.4,350]
            aqi_colors=["#00C853","#FFD600","#FF6D00","#D50000","#6A1B9A","#4A148C"]
            aqi_labels=["Good","Moderate","USG","Unhealthy","V.Unhealthy","Hazardous"]
            txt_colors=["black","black","black","white","white","white"]
            for i,(lo,hi) in enumerate(zip(thresholds[:-1],thresholds[1:])):
                ax_bar.barh(0,hi-lo,left=lo,height=0.5,color=aqi_colors[i],edgecolor=WHITE,linewidth=0.8)
                ax_bar.text((lo+hi)/2,0,aqi_labels[i],ha="center",va="center",fontsize=6.5,color=txt_colors[i],fontweight="bold")
            ax_bar.axvline(min(prediction,340),color=CHARCOAL,linewidth=3,ymin=0.05,ymax=0.95)
            ax_bar.set_xlim(0,350); ax_bar.set_ylim(-0.5,0.5); ax_bar.axis("off")
            fig_bar.patch.set_color(IVORY)
            st.pyplot(fig_bar,use_container_width=True); plt.close()
