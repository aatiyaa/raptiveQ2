import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Aatiya Ali, July 18th, 2026 
st.set_page_config(page_title="Space Weather Analysis", layout="wide")

st.title("Solar Flare & Particle Flux Analysis")
#st.write("This dashboard analyzes the correlation between solar flare intensity and high-energy particle flux.")
st.write("""Hypothesis: \n
Given that solar flares serve as primary indicators of intense solar activity, 
then flare intensity (soft X-ray flux) should exhibit a positive, non-linear correlation 
with increased particle flux (protons and electrons) in the solar wind, 
driven by the shared energetic processes of solar eruptions rather than a simple linear relationship.""")


@st.cache_data
def load_and_process_data():
    # GOES proton data at earth
    # >1 MeV 
    goes = pd.read_csv('./input_g.csv') 

    # GOES flare data at earth
    # 0.1 − 0.8 nm
    sxr = pd.read_csv('./input_sxr.csv') 

    # EPHIN electron data at L1 
    # 0.045 MeV to 0.062 MeV
    ephin = pd.read_csv('./input_e.csv') 
 
    goes.set_index('time_tag', inplace=True)
    sxr.set_index('time_tag', inplace=True)
    ephin.set_index('time_tag', inplace=True)

    # Merge
    df_merged = goes.join(sxr, lsuffix='_goes', rsuffix='_sxr', how='inner')
    df_merged = df_merged.join(ephin, rsuffix='_ephin', how='inner')

    # Downsample for github reqs
    TARGET_SIZE = 5000000
    if len(df_merged) >= TARGET_SIZE:
        start_idx = (len(df_merged) - TARGET_SIZE) // 2
        df_merged = df_merged.iloc[start_idx : start_idx + TARGET_SIZE]
    
    return df_merged

# Correlation Function
def get_correlation(df, col_x, col_y):
    subset = df[[col_x, col_y]].dropna()
    subset[col_x] = pd.to_numeric(subset[col_x], errors='coerce')
    subset[col_y] = pd.to_numeric(subset[col_y], errors='coerce')
    subset = subset.dropna()
    subset = subset[np.isfinite(subset[col_x]) & np.isfinite(subset[col_y])]
    
    if len(subset) < 2:
        return (np.nan, np.nan), (np.nan, np.nan)
        
    p_corr = stats.pearsonr(subset[col_x].values, subset[col_y].values)
    s_corr = stats.spearmanr(subset[col_x].values, subset[col_y].values)
    return p_corr, s_corr
 
df_show = load_and_process_data()
st.subheader("Visualizing Particle Flux vs. Flare Intensity")

fig, axes = plt.subplots(figsize=(5, 2.5))
sns.set_theme(style="ticks", context="notebook")

axes.scatter(df_show['E14.13_15.85'], df_show['xl'], 
             color='orange', alpha=0.5, s=20, label='EPHIN & Flare fluxes', edgecolor='none')
axes.scatter(df_show['p1_flux_ic'], df_show['xl'], 
             color='palevioletred', alpha=0.75, s=20, label='GOES & Flare fluxes', edgecolor='none')

axes.set_title('SXR Flare Intensity vs. High-Energy Proton & Electron Flux')
axes.set_xlabel("Proton Flux [pfu]")
axes.set_ylabel("Flare Intensity [W/m2]")
axes.set_yscale('log')
axes.set_xscale('log')
axes.set_ylim(1e-10,5e-5)
axes.grid(True, which="major", ls="-", alpha=0.5)
axes.legend(loc='lower center', ncol=2,fontsize=9)

st.pyplot(fig)
st.subheader("Statistical Analysis")

col1, col2 = st.columns(2)

# GOES Metrics
p_goes, s_goes = get_correlation(df_show, 'p1_flux_ic', 'xl')
with col1:
    st.markdown("**GOES Proton Correlation**")
    st.metric("Pearson", f"{p_goes[0]:.4f}")
    st.metric("Spearman", f"{s_goes[0]:.4f}")

# EPHIN Metrics
p_ephin, s_ephin = get_correlation(df_show, 'E14.13_15.85', 'xl')
with col2:
    st.markdown("**EPHIN Electron Correlation**")
    st.metric("Pearson", f"{p_ephin[0]:.4f}")
    st.metric("Spearman", f"{s_ephin[0]:.4f}")

st.divider()
 
st.markdown("""
### Summary of Findings
This Python pipeline analyzes multi-instrument space weather data, comparing GOES proton measurements, EPHIN electron data at L1, and solar flare signatures (GOES soft X-rays). The objective was to determine if solar flare intensity correlates with increased particle (proton and electron) flux. Applying both Pearson (linear) and Spearman (monotonic) analyses from Q1, we find correlation coefficients near zero for both datasets. The scatter plot confirms this, showing no discernible linear or monotonic relationship between flare intensity and the recorded particle fluxes. This indicates that flare intensity, in isolation and at the provided time-scales, does not serve as a direct proxy for real-time particle flux measurements. Time lag between events, instrumentation sensitivity, data preparation, and even background noise could all be contributing to the lack of correlation found. 
            """)
