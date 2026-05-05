# -*- coding: utf-8 -*-
# Task 4: Application Development — Streamlit Multi-Page App
# ─────────────────────────────────────────────────────────────
# HOW TO USE:
#   Streamlit Cloud: deploy this file directly — it runs as app.py
#   Google Colab:    run setup_colab.py first, then this file
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, urllib.request
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Beijing Air Quality Explorer", page_icon="🌫️", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
POLLUTANTS = ["PM2.5","PM10","SO2","NO2","CO","O3"]
MET_VARS   = ["TEMP","PRES","DEWP","WSPM","RAIN"]
STATIONS   = ["Dongsi","Guanyuan","Shunyi","Huairou"]
SCOLORS    = {"Dongsi":"#e53e3e","Guanyuan":"#dd6b20","Shunyi":"#3182ce","Huairou":"#2c7a7b"}
SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
DATA_PATH  = "beijing_merged_data.csv"

GITHUB_USER   = "anulax1114-design"
GITHUB_REPO   = "PRAC1-st20347210-DA"
GITHUB_BRANCH = "main"
STATION_FILES = {
    "Dongsi"  : "PRSA_Data_Dongsi_20130301-20170228.csv",
    "Guanyuan": "PRSA_Data_Guanyuan_20130301-20170228.csv",
    "Shunyi"  : "PRSA_Data_Shunyi_20130301-20170228.csv",
    "Huairou" : "PRSA_Data_Huairou_20130301-20170228.csv",
}

# ── Load or download data ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
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
                st.warning(f"Could not download {station}: {e}")
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
    df["station_type"] = df["station"].apply(lambda s: "Urban" if s in ["Dongsi","Guanyuan"] else "Suburban")
    return df

def clean_ax(ax):
    ax.spines[["top","right"]].set_visible(False)

# ── Sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigate", ["🏠 Home", "📋 Dataset", "📊 Visualisation", "🤖 Model Outputs"])

df = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🌫️ Beijing Air Quality Explorer")
    st.caption("CMP7005 · TASK 4 — Hourly monitoring data from 4 stations · March 2013 – February 2017")
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📋 DATASET**\n\nBrowse, filter, search and download the merged Beijing air quality dataset.")
    with col2:
        st.info("**📊 VISUALISATION**\n\nKDE distributions, temporal trends, correlation heatmaps, seasonal comparisons.")
    with col3:
        st.info("**🤖 MODEL OUTPUTS**\n\nRandom Forest diagnostics, feature importance and live PM2.5 prediction.")
    st.divider()
    st.info("👈 Use the sidebar to navigate between sections")
    if df is not None:
        st.subheader("Dataset Overview")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Mean PM2.5",    f"{df['PM2.5'].mean():.1f} µg/m³")
        c3.metric("Max PM2.5",     f"{df['PM2.5'].max():.0f} µg/m³")
        c4.metric("Stations",      str(df["station"].nunique()))
        c5.metric("Years",         f"{int(df['year'].min())}–{int(df['year'].max())}")
    else:
        st.error("Could not load dataset. Check your GitHub repo and file names.")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset":
    if df is None:
        st.error("Dataset not available."); st.stop()

    with st.sidebar:
        st.markdown("### Filters")
        sel_stations = st.multiselect("Stations", STATIONS, default=STATIONS)
        sel_years    = st.multiselect("Years", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
        sel_seasons  = st.multiselect("Seasons", ["Spring","Summer","Autumn","Winter"], default=["Spring","Summer","Autumn","Winter"])

    dff = df[df["station"].isin(sel_stations) & df["year"].isin(sel_years) & df["season"].isin(sel_seasons)]
    if dff.empty:
        st.warning("No data matches filters."); st.stop()

    st.title("📋 Dataset Explorer")
    st.caption("Browse, filter and summarise the merged Beijing air quality dataset")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Records",    f"{len(dff):,}")
    c2.metric("Mean PM2.5", f"{dff['PM2.5'].mean():.1f} µg/m³")
    c3.metric("Max PM2.5",  f"{dff['PM2.5'].max():.0f} µg/m³")
    c4.metric("Stations",   str(dff["station"].nunique()))
    c5.metric("Date Range", f"{int(dff['year'].min())}–{int(dff['year'].max())}")

    st.divider()
    tab1,tab2,tab3 = st.tabs(["🗂️ Raw Data","📈 Statistics","❓ Missing Values"])

    with tab1:
        ca,cb = st.columns([3,1])
        search = ca.text_input("Search","")
        n_rows = cb.select_slider("Rows",[50,100,250,500],value=100)
        show_cols = st.multiselect("Columns", dff.columns.tolist(),
            default=["year","month","day","hour","station","station_type","PM2.5","PM10","SO2","NO2","CO","O3","TEMP","WSPM"])
        disp = dff[show_cols]
        if search:
            disp = disp[disp.apply(lambda r: r.astype(str).str.contains(search,case=False).any(),axis=1)]
        st.dataframe(disp.head(n_rows), use_container_width=True, height=400)
        st.caption(f"Showing {min(n_rows,len(disp)):,} of {len(dff):,} rows")
        st.download_button("⬇️ Download CSV", dff[show_cols].to_csv(index=False).encode(), "beijing_filtered.csv","text/csv")

    with tab2:
        grp = st.selectbox("Variable group",["All pollutants","All meteorological","Custom"])
        if grp=="All pollutants":       num_cols=POLLUTANTS
        elif grp=="All meteorological": num_cols=MET_VARS
        else: num_cols=st.multiselect("Select",POLLUTANTS+MET_VARS,default=["PM2.5","NO2","TEMP"])
        if num_cols:
            st.dataframe(dff[num_cols].describe().T.round(3),use_container_width=True)
            pv=st.selectbox("Station comparison",num_cols,key="pv")
            st.dataframe(dff.groupby("station")[pv].agg(["mean","median","std","min","max"]).round(2),use_container_width=True)

    with tab3:
        miss=dff[POLLUTANTS+MET_VARS].isna().sum().reset_index()
        miss.columns=["Feature","Missing"]; miss["%"]=(miss["Missing"]/len(dff)*100).round(2)
        miss=miss.sort_values("%",ascending=False)
        cm1,cm2=st.columns([2,1])
        with cm1:
            fig,ax=plt.subplots(figsize=(8,4))
            colors_m=["#e53e3e" if p>5 else "#dd6b20" if p>1 else "#38a169" for p in miss["%"]]
            bars=ax.barh(miss["Feature"],miss["%"],color=colors_m,edgecolor="none")
            for bar,val in zip(bars,miss["%"]):
                ax.text(bar.get_width()+0.1,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va="center",fontsize=8)
            ax.set_xlabel("% Missing"); ax.set_title("Missing Value Rate"); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with cm2:
            st.dataframe(miss,use_container_width=True,height=350)
        st.info("Missing values are most prevalent in gas pollutants (SO2, NO2, CO, O3).")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualisation":
    if df is None:
        st.error("Dataset not available."); st.stop()

    with st.sidebar:
        st.markdown("### Filters")
        sel_stations = st.multiselect("Stations", STATIONS, default=STATIONS)
        sel_years    = st.multiselect("Years", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
        sel_seasons  = st.multiselect("Seasons", ["Spring","Summer","Autumn","Winter"], default=["Spring","Summer","Autumn","Winter"])

    dff = df[df["station"].isin(sel_stations) & df["year"].isin(sel_years) & df["season"].isin(sel_seasons)]
    if dff.empty:
        st.warning("No data matches filters."); st.stop()

    st.title("📊 Visualisation Suite")
    st.caption("Distributions, temporal trends, correlations, seasonal comparisons")
    st.divider()

    tab1,tab2,tab3,tab4 = st.tabs(["🌡️ Distributions","📅 Temporal Trends","🔗 Correlations","🌿 Seasonal"])

    with tab1:
        rc1,rc2=st.columns(2)
        poll_x=rc1.selectbox("X variable",POLLUTANTS+MET_VARS,index=0)
        poll_y=rc2.selectbox("Y variable (scatter)",POLLUTANTS+MET_VARS,index=1)
        col1,col2=st.columns(2)
        with col1:
            fig,ax=plt.subplots(figsize=(6,4))
            for stn in sel_stations:
                sub=dff[dff["station"]==stn][poll_x].dropna()
                if len(sub)>10:
                    sns.kdeplot(sub,ax=ax,label=stn,fill=True,alpha=0.2,linewidth=2,color=SCOLORS.get(stn))
            ax.set_xlabel(poll_x); ax.set_ylabel("Density"); ax.legend(fontsize=8); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            fig2,ax2=plt.subplots(figsize=(6,4))
            data_bp=[dff[dff["station"]==s][poll_x].dropna().values for s in sel_stations]
            if any(len(d)>0 for d in data_bp):
                bp=ax2.boxplot(data_bp,labels=sel_stations,patch_artist=True,
                    medianprops=dict(color="black",linewidth=2),
                    whiskerprops=dict(color="gray"),capprops=dict(color="gray"),
                    flierprops=dict(marker="o",markersize=2,alpha=0.3))
                for patch,stn in zip(bp["boxes"],sel_stations):
                    patch.set_facecolor(SCOLORS.get(stn,"#718096")); patch.set_alpha(0.7)
            ax2.set_ylabel(poll_x); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()
        st.divider()
        sn=st.slider("Sample size",1000,20000,5000,1000)
        samp=dff.sample(min(sn,len(dff)),random_state=42)
        fig3,ax3=plt.subplots(figsize=(12,4))
        for stn in sel_stations:
            s=samp[samp["station"]==stn][[poll_x,poll_y]].dropna()
            ax3.scatter(s[poll_x],s[poll_y],alpha=0.4,s=8,color=SCOLORS.get(stn),label=stn,rasterized=True)
        ax3.set_xlabel(poll_x); ax3.set_ylabel(poll_y); ax3.legend(fontsize=9); clean_ax(ax3)
        st.pyplot(fig3,use_container_width=True); plt.close()

    with tab2:
        t_var=st.selectbox("Variable",POLLUTANTS+MET_VARS,key="tvar")
        t_mode=st.radio("View",["Hourly (diurnal)","Monthly trend","Annual trend"],horizontal=True)
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
                ax.plot(s["date"],s[t_var],color=SCOLORS.get(stn),linewidth=1.8,label=stn)
        else:
            grp=dff.groupby(["year","station"])[t_var].mean().reset_index()
            for stn in sel_stations:
                s=grp[grp["station"]==stn]
                ax.plot(s["year"],s[t_var],color=SCOLORS.get(stn),linewidth=2.5,marker="s",markersize=8,label=stn)
        if t_var=="PM2.5":
            ax.axhline(15,color="red",linestyle="--",linewidth=1.2,alpha=0.7,label="WHO 15 µg/m³")
        ax.set_ylabel(t_var); ax.set_title(f"{t_mode} — {t_var}"); ax.legend(fontsize=9); clean_ax(ax)
        st.pyplot(fig,use_container_width=True); plt.close()
        st.divider()
        st.markdown("**Hour x Month PM2.5 Heatmap**")
        hm_stn=st.selectbox("Station",sel_stations,key="hmstn")
        hm_data=dff[dff["station"]==hm_stn].groupby(["month","hour"])["PM2.5"].mean().unstack(fill_value=0)
        fig_h,ax_h=plt.subplots(figsize=(14,5))
        sns.heatmap(hm_data,ax=ax_h,cmap="YlOrRd",linewidths=0.3,cbar_kws={"shrink":0.7})
        ax_h.set_xlabel("Hour"); ax_h.set_ylabel("Month"); ax_h.set_title(f"Hour x Month PM2.5 — {hm_stn}")
        st.pyplot(fig_h,use_container_width=True); plt.close()

    with tab3:
        corr_vars=st.multiselect("Variables",POLLUTANTS+MET_VARS,default=POLLUTANTS+["TEMP","WSPM","DEWP"])
        if len(corr_vars)>=2:
            corr=dff[corr_vars].corr()
            mask=np.triu(np.ones_like(corr,dtype=bool))
            fig,ax=plt.subplots(figsize=(10,8))
            sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="coolwarm",
                linewidths=0.3,ax=ax,vmin=-1,vmax=1,annot_kws={"size":8},cbar_kws={"shrink":0.7})
            ax.set_title("Pearson Correlation Matrix")
            st.pyplot(fig,use_container_width=True); plt.close()

    with tab4:
        season_order=["Spring","Summer","Autumn","Winter"]
        s_var=st.selectbox("Variable",POLLUTANTS+MET_VARS,key="svar")
        col1,col2=st.columns(2)
        with col1:
            sd=dff.groupby(["season","station"])[s_var].mean().reset_index()
            fig,ax=plt.subplots(figsize=(7,4.5))
            x=np.arange(len(season_order)); w=0.2
            for i,stn in enumerate(sel_stations):
                s=sd[sd["station"]==stn].set_index("season")
                vals=[s.loc[se,s_var] if se in s.index else 0 for se in season_order]
                ax.bar(x+i*w-w*len(sel_stations)/2,vals,w,label=stn,color=SCOLORS.get(stn),alpha=0.85,edgecolor="none")
            ax.set_xticks(x); ax.set_xticklabels(season_order); ax.set_ylabel(f"Mean {s_var}")
            ax.legend(fontsize=8); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            td=dff.groupby(["station_type","season"])[s_var].mean().unstack().reindex(columns=season_order)
            fig2,ax2=plt.subplots(figsize=(7,4.5))
            x2=np.arange(len(season_order)); w2=0.35
            ax2.bar(x2-w2/2,td.loc["Urban"]    if "Urban"    in td.index else [0]*4,w2,label="Urban",   color="#3182ce",alpha=0.85)
            ax2.bar(x2+w2/2,td.loc["Suburban"] if "Suburban" in td.index else [0]*4,w2,label="Suburban",color="#2c7a7b",alpha=0.85)
            ax2.set_xticks(x2); ax2.set_xticklabels(season_order); ax2.set_ylabel(f"Mean {s_var}")
            ax2.legend(fontsize=9); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL OUTPUTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Outputs":
    import joblib

    MODEL_DIR = "model_artefacts"
    AQI_BREAKS = [
        (0,    12,   "Good",                 "#00c853","#000"),
        (12.1, 35.4, "Moderate",             "#ffd600","#000"),
        (35.5, 55.4, "Unhealthy (Sensitive)","#ff6d00","#000"),
        (55.5, 150.4,"Unhealthy",            "#d50000","#fff"),
        (150.5,250.4,"Very Unhealthy",       "#6a1b9a","#fff"),
        (250.5,9999, "Hazardous",            "#4a148c","#fff"),
    ]

    def aqi_info(v):
        for lo,hi,label,bg,fg in AQI_BREAKS:
            if lo<=v<=hi: return label,bg,fg
        return "Hazardous","#4a148c","#fff"

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

    st.title("🤖 Model Outputs")
    st.caption("Random Forest — diagnostics, feature importance, per-station performance, live prediction")
    st.divider()

    if gb_model is None:
        st.warning("Model artefacts not found in model_artefacts/. Run Task 3 notebook first to generate them.")
        st.stop()

    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    actual=test_preds["actual"]; predicted=test_preds["predicted"]
    rmse=np.sqrt(mean_squared_error(actual,predicted))
    mae=mean_absolute_error(actual,predicted)
    r2=r2_score(actual,predicted)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Test RMSE",   f"{rmse:.2f} µg/m³")
    c2.metric("Test MAE",    f"{mae:.2f} µg/m³")
    c3.metric("R2 Score",    f"{r2:.4f}")
    c4.metric("Test Samples",f"{len(actual):,}")

    st.divider()
    tab1,tab2,tab3=st.tabs(["📉 Diagnostics","🏆 Feature Importance","🔮 Live Prediction"])

    with tab1:
        residuals=actual.values-predicted.values
        col1,col2=st.columns(2)
        with col1:
            fig,ax=plt.subplots(figsize=(6,5))
            sc=ax.scatter(actual,predicted,alpha=0.12,s=6,rasterized=True,
                          c=np.abs(residuals),cmap="RdYlGn_r",vmin=0,vmax=100)
            plt.colorbar(sc,ax=ax,label="Abs Residual",shrink=0.8)
            mn,mx=actual.min(),actual.max()
            ax.plot([mn,mx],[mn,mx],"--",color="gray",linewidth=1.5,label="Perfect")
            ax.set_xlabel("Actual PM2.5 (µg/m³)"); ax.set_ylabel("Predicted PM2.5 (µg/m³)")
            ax.set_title("Actual vs Predicted"); ax.legend(fontsize=8); clean_ax(ax)
            st.pyplot(fig,use_container_width=True); plt.close()
        with col2:
            fig2,ax2=plt.subplots(figsize=(6,5))
            ax2.hist(residuals,bins=80,color="#3182ce",edgecolor="white",linewidth=0.3,alpha=0.85)
            ax2.axvline(0,color="gray",linestyle="--",linewidth=1.5)
            ax2.axvline(np.mean(residuals),color="red",linestyle="-",linewidth=1.2,label=f"Mean: {np.mean(residuals):.2f}")
            ax2.set_xlabel("Residual (µg/m³)"); ax2.set_ylabel("Count")
            ax2.set_title("Residual Distribution"); ax2.legend(fontsize=8); clean_ax(ax2)
            st.pyplot(fig2,use_container_width=True); plt.close()
        fig3,ax3=plt.subplots(figsize=(12,3.5))
        ax3.scatter(predicted,residuals,alpha=0.1,s=5,color="#38a169",rasterized=True)
        ax3.axhline(0,   color="gray",linestyle="--",linewidth=1.2)
        ax3.axhline( mae,color="red", linestyle=":", linewidth=1,label=f"+MAE ({mae:.1f})")
        ax3.axhline(-mae,color="red", linestyle=":", linewidth=1,label=f"-MAE ({mae:.1f})")
        ax3.set_xlabel("Predicted PM2.5 (µg/m³)"); ax3.set_ylabel("Residual")
        ax3.set_title("Residuals vs Predicted"); ax3.legend(fontsize=8); clean_ax(ax3)
        st.pyplot(fig3,use_container_width=True); plt.close()
        if "station" in test_preds.columns:
            st.divider(); st.markdown("**Per-Station Performance**")
            stn_m=test_preds.groupby("station").apply(lambda g: pd.Series({
                "RMSE":np.sqrt(mean_squared_error(g["actual"],g["predicted"])),
                "MAE": mean_absolute_error(g["actual"],g["predicted"]),
                "R2":  r2_score(g["actual"],g["predicted"]),
                "n":   len(g)
            })).reset_index()
            st.dataframe(stn_m.set_index("station").round(4),use_container_width=True)

    with tab2:
        if hasattr(gb_model,"feature_importances_"):
            imp=pd.Series(gb_model.feature_importances_,index=feat_names).sort_values()
            top_n=st.slider("Top N features",5,len(imp),min(15,len(imp)))
            top=imp.tail(top_n)
            fig,ax=plt.subplots(figsize=(9,max(4,top_n*0.45)))
            bar_colors=plt.cm.RdYlGn(top.values/top.values.max())
            top.plot(kind="barh",ax=ax,color=bar_colors,edgecolor="none")
            for i,(val,name) in enumerate(zip(top.values,top.index)):
                ax.text(val+0.002,i,f"{val:.4f}",va="center",fontsize=8)
            ax.set_xlabel("Importance Score"); ax.set_title(f"Top {top_n} Feature Importances")
            clean_ax(ax); st.pyplot(fig,use_container_width=True); plt.close()

    with tab3:
        st.markdown("Enter current sensor and weather readings to predict PM2.5:")
        WIND_OPT=["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW","calm","CALM"]
        with st.form("predict_form"):
            st.markdown("**Pollutant Readings**")
            rc1,rc2,rc3,rc4,rc5,rc6=st.columns(6)
            pm10=rc1.number_input("PM10",      0.0,2000.0, 80.0,step=1.0)
            so2 =rc2.number_input("SO2",       0.0, 500.0, 15.0,step=1.0)
            no2 =rc3.number_input("NO2",       0.0, 500.0, 50.0,step=1.0)
            co  =rc4.number_input("CO",        0.0,10000.0,600.0,step=10.0)
            o3  =rc5.number_input("O3",        0.0, 500.0, 40.0,step=1.0)
            lag1=rc6.number_input("Prev PM2.5",0.0,1000.0, 50.0,step=1.0)
            st.markdown("**Meteorological Conditions**")
            mc1,mc2,mc3,mc4=st.columns(4)
            temp=mc1.number_input("Temp (C)",  -30.0, 45.0, 15.0,step=0.5)
            pres=mc2.number_input("Pressure",  980.0,1060.0,1010.0,step=0.5)
            dewp=mc3.number_input("Dew Point", -40.0, 30.0,  5.0,step=0.5)
            wspm=mc4.number_input("Wind Speed",  0.0, 20.0,  2.0,step=0.1)
            ec1,ec2,ec3,ec4=st.columns(4)
            rain    =ec1.number_input("Rainfall",0.0,100.0,0.0,step=0.1)
            wd      =ec2.selectbox("Wind Dir",WIND_OPT)
            stn_type=ec3.selectbox("Station Type",["Urban","Suburban"])
            hour_in =ec4.slider("Hour",0,23,12)
            month_in=st.slider("Month",1,12,6)
            submitted=st.form_submit_button("PREDICT PM2.5",use_container_width=True)
        if submitted:
            is_urban=1 if stn_type=="Urban" else 0
            try:    wd_enc=le_wind.transform([wd])[0]
            except: wd_enc=0
            feat_map={
                "PM10":pm10,"SO2":so2,"NO2":no2,"CO":co,"O3":o3,
                "TEMP":temp,"PRES":pres,"DEWP":dewp,"WSPM":wspm,"RAIN":rain,
                "wd_encoded":wd_enc,"is_urban":is_urban,
                "hour_sin": np.sin(2*np.pi*hour_in/24),
                "hour_cos": np.cos(2*np.pi*hour_in/24),
                "month_sin":np.sin(2*np.pi*month_in/12),
                "month_cos":np.cos(2*np.pi*month_in/12),
                "PM2.5_lag1":lag1
            }
            X_pred    =np.array([[feat_map[f] for f in feat_names]])
            prediction=float(max(0.0,gb_model.predict(X_pred)[0]))
            label,bg_c,fg_c=aqi_info(prediction)
            st.divider()
            res1,res2=st.columns(2)
            with res1: st.metric("Predicted PM2.5",f"{prediction:.1f} µg/m³")
            with res2: st.metric("AQI Category",label)
            fig_bar,ax_bar=plt.subplots(figsize=(10,1.2))
            thresholds=[0,12,35.4,55.4,150.4,250.4,350]
            aqi_colors=["#00c853","#ffd600","#ff6d00","#d50000","#6a1b9a","#4a148c"]
            aqi_labels=["Good","Moderate","USG","Unhealthy","V.Unhealthy","Hazardous"]
            txt_colors=["black","black","black","white","white","white"]
            for i,(lo,hi) in enumerate(zip(thresholds[:-1],thresholds[1:])):
                ax_bar.barh(0,hi-lo,left=lo,height=0.6,color=aqi_colors[i],edgecolor="white",linewidth=0.5)
                ax_bar.text((lo+hi)/2,0,aqi_labels[i],ha="center",va="center",fontsize=7,color=txt_colors[i],fontweight="bold")
            ax_bar.axvline(min(prediction,340),color="black",linewidth=3,ymin=0.05,ymax=0.95)
            ax_bar.set_xlim(0,350); ax_bar.set_ylim(-0.5,0.5); ax_bar.axis("off")
            ax_bar.set_title(f"AQI Scale - {prediction:.1f} µg/m3",fontsize=10)
            st.pyplot(fig_bar,use_container_width=True); plt.close()
            advice={
                "Good":                  ("OK",          "Air quality is satisfactory. Outdoor activities are safe."),
                "Moderate":              ("Warning",     "Sensitive people should consider limiting outdoor exertion."),
                "Unhealthy (Sensitive)": ("Caution",     "Sensitive groups should reduce outdoor activity."),
                "Unhealthy":             ("Alert",       "Everyone should limit prolonged outdoor exertion."),
                "Very Unhealthy":        ("Health Alert","Avoid all outdoor physical activity."),
                "Hazardous":             ("Emergency",   "Stay indoors. Avoid all outdoor activity."),
            }
            lvl,msg=advice.get(label,("Info","Monitor conditions."))
            st.info(f"{lvl}: {msg}")
