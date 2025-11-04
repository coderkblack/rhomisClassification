import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="RHoMIS Analysis App", layout="wide")
st.text("RHoMIS Dataset Analysis: Livestock, Crops, and Market Orientation")

# -------------------------------
# Load & Clean Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/home/jakes/Documents/strathmore/Modules/Module 1/dataMining/project/Rhomis/final/RHoMIS_Indicators.csv", encoding="latin1")

# Numeric columns (expanded for livestock, crops, market)
    num_cols = [
        'HHsizemembers', 'HHsizeMAE', 'LandOwned', 'LandCultivated', 'LivestockHoldings',
        'NrofMonthsFoodInsecure', 'PPI_Likelihood', 'score_HDDS_GoodSeason',
        'score_HDDS_BadSeason', 'total_income_USD_PPP_pHH_Yr', 'offfarm_income_USD_PPP_pHH_Yr',
        'farm_income_USD_PPP_pHH_Yr', 'Food_Availability_kCal_MAE_day',
        'Market_Orientation', 'Livestock_Orientation', 'GHGEmissions', 'Gender_MaleControl',
        'Gender_FemaleControl', 'NFertInput', 'GPS_LAT', 'GPS_LON',
        # Livestock-specific
        'value_livestock_production_USD_PPP_pHH_Yr', 'value_livestock_prod_consumed_USD_PPP_pHH_Yr',
        'livestock_prodsales_USD_PPP_pHH_Yr',
        # Crop-specific
        'value_crop_produce_USD_PPP_pHH_Yr', 'value_crop_consumed_USD_PPP_pHH_Yr',
        'crop_sales_USD_PPP_pHH_Yr'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")

countries = sorted(df['ID_COUNTRY'].dropna().unique())
selected_countries = st.sidebar.multiselect("Countries", countries, default=countries[:1])

years = sorted(df['YEAR'].dropna().unique())
selected_years = st.sidebar.multiselect("Years", years, default=years)

# Filter
filtered_df = df[
    df['ID_COUNTRY'].isin(selected_countries) &
    df['YEAR'].isin(selected_years)
].copy()

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Livestock Analysis", "Crop Analysis", "Market Orientation", "Comparisons & Map"
])

# -------------------------------
# Tab 1: Overview
# -------------------------------
with tab1:
    st.header("Data Overview")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Households", len(filtered_df))
        st.metric("Countries", len(selected_countries))
    with col2:
        st.write(f"**Selected:** {', '.join(selected_countries)} | "
                 f"**Years:** {', '.join(map(str, selected_years))}")

    with st.expander("Sample Data"):
        st.dataframe(filtered_df.head(10))

# -------------------------------
# Tab 2: Livestock Analysis
# -------------------------------
with tab2:
    st.header("Livestock Analysis")

    livestock_features = [
        'LivestockHoldings', 'Livestock_Orientation', 'value_livestock_production_USD_PPP_pHH_Yr',
        'value_livestock_prod_consumed_USD_PPP_pHH_Yr', 'livestock_prodsales_USD_PPP_pHH_Yr',
        'GHGEmissions'
    ]

    sel_livestock = st.multiselect("Livestock Features", livestock_features,
                                   default=livestock_features[:3], key="livestock_sel")

    if sel_livestock:
        st.subheader("Livestock Summary Stats")
        st.dataframe(filtered_df[sel_livestock].describe())

    for i, feat in enumerate(sel_livestock):
        with st.expander(f"Distribution of {feat}"):
            fig_hist = px.histogram(
                filtered_df, x=feat, color='ID_COUNTRY',
                marginal="box", title=f"{feat} – Histogram + Box"
            )
            st.plotly_chart(fig_hist, use_container_width=True,
                            key=f"livestock_hist_{feat}_{i}")

            fig_box = px.box(filtered_df, x='ID_COUNTRY', y=feat, points="outliers")
            st.plotly_chart(fig_box, use_container_width=True,
                            key=f"livestock_box_{feat}_{i}")

    if len(sel_livestock) >= 2:
        with st.expander("Livestock Correlations"):
            corr = filtered_df[sel_livestock].corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig, use_container_width=True,
                            key="livestock_corr")

# -------------------------------
# Tab 3: Crop Analysis
# -------------------------------
with tab3:
    st.header("Crop Analysis")

    crop_features = [
        'LandOwned', 'LandCultivated', 'value_crop_produce_USD_PPP_pHH_Yr',
        'value_crop_consumed_USD_PPP_pHH_Yr', 'crop_sales_USD_PPP_pHH_Yr',
        'NFertInput'
    ]

    sel_crop = st.multiselect("Crop Features", crop_features,
                              default=crop_features[:3], key="crop_sel")

    if sel_crop:
        st.subheader("Crop Summary Stats")
        st.dataframe(filtered_df[sel_crop].describe())

    for i, feat in enumerate(sel_crop):
        with st.expander(f"Distribution of {feat}"):
            fig_hist = px.histogram(
                filtered_df, x=feat, color='ID_COUNTRY',
                marginal="box", title=f"{feat} – Histogram + Box"
            )
            st.plotly_chart(fig_hist, use_container_width=True,
                            key=f"crop_hist_{feat}_{i}")

            fig_box = px.box(filtered_df, x='ID_COUNTRY', y=feat, points="outliers")
            st.plotly_chart(fig_box, use_container_width=True,
                            key=f"crop_box_{feat}_{i}")

    if len(sel_crop) >= 2:
        with st.expander("Crop Correlations"):
            corr = filtered_df[sel_crop].corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig, use_container_width=True,
                            key="crop_corr")

# -------------------------------
# Tab 4: Market Orientation Analysis
# -------------------------------
with tab4:
    st.header("Market Orientation Analysis")

    st.text("Market orientation is the proportion of farm produce which is sold (where the 'amount' of farm produce is measured in cash value, not mass). The calculations is as follows: Farm Income (USD) per household per year/Value of farm produce (USD) per household per year")
    market_features = [
        'Market_Orientation', 'Livestock_Orientation', 'crop_sales_USD_PPP_pHH_Yr',
        'livestock_prodsales_USD_PPP_pHH_Yr', 'value_crop_consumed_USD_PPP_pHH_Yr',
        'value_livestock_prod_consumed_USD_PPP_pHH_Yr'
    ]

    sel_market = st.multiselect("Market Features", market_features,
                                default=market_features[:3], key="market_sel")

    if sel_market:
        st.subheader("Market Summary Stats")
        st.dataframe(filtered_df[sel_market].describe())

    for i, feat in enumerate(sel_market):
        with st.expander(f"Distribution of {feat}"):
            fig_hist = px.histogram(
                filtered_df, x=feat, color='ID_COUNTRY',
                marginal="box", title=f"{feat} – Histogram + Box"
            )
            st.plotly_chart(fig_hist, use_container_width=True,
                            key=f"market_hist_{feat}_{i}")

            fig_box = px.box(filtered_df, x='ID_COUNTRY', y=feat, points="outliers")
            st.plotly_chart(fig_box, use_container_width=True,
                            key=f"market_box_{feat}_{i}")

    if len(sel_market) >= 2:
        with st.expander("Market Correlations"):
            corr = filtered_df[sel_market].corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig, use_container_width=True,
                            key="market_corr")

    if 'Market_Orientation' in sel_market and len(sel_market) > 1:
        y_feat = st.selectbox(
            "Compare Market Orientation to",
            [f for f in sel_market if f != 'Market_Orientation'],
            key="market_scatter_y"
        )
        with st.expander(f"Scatter: Market Orientation vs {y_feat}"):
            fig = px.scatter(
                filtered_df, x='Market_Orientation', y=y_feat,
                color='ID_COUNTRY', hover_data=['ID_HH']
            )
            st.plotly_chart(fig, use_container_width=True,
                            key="market_scatter")

# -------------------------------
# Tab 5: Comparisons & Map
# -------------------------------
with tab5:
    st.header("Comparisons & Geography")

    all_features = list(set(livestock_features + crop_features + market_features))
    if len(selected_countries) > 1:
        with st.expander("Average Values by Country"):
            avg_cols = st.multiselect("Features to Compare", all_features,
                                      default=all_features[:3], key="avg_cols")
            if avg_cols:
                avg_df = filtered_df.groupby('ID_COUNTRY')[avg_cols].mean().reset_index()
                fig = px.bar(
                    avg_df.melt(id_vars='ID_COUNTRY'), x='ID_COUNTRY', y='value',
                    color='variable', barmode='group', title="Average Feature Values"
                )
                st.plotly_chart(fig, use_container_width=True,
                                key="avg_bar")

    # --- Interactive Map (Safe) ---
    map_df = filtered_df.dropna(subset=['GPS_LAT', 'GPS_LON', 'LandCultivated'])
    if not map_df.empty:
        with st.expander("Geographic Distribution"):
            size_vals = map_df['LandCultivated'].clip(lower=0.1) * 5
            fig_map = px.scatter_mapbox(
                map_df,
                lat='GPS_LAT', lon='GPS_LON',
                size=size_vals,
                color='ID_COUNTRY',
                hover_data=['ID_HH', 'LivestockHoldings', 'LandCultivated', 'Market_Orientation'],
                mapbox_style="open-street-map",
                zoom=5,
                title="Household Locations (Size = Land Cultivated)"
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True,
                            key="geo_map")
    else:
        st.info("No valid GPS + LandCultivated data for mapping.")

# -------------------------------
# Download
# -------------------------------
st.markdown("---")
st.header("Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"rhomis_{'_'.join(selected_countries)}_{'_'.join(map(str, selected_years))}.csv",
    mime="text/csv"
)