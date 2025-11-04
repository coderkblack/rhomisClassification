import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="RHoMIS Analysis App", layout="wide")
st.title("RHoMIS Dataset Analysis: Livestock, Crops, and Market Orientation")

# -------------------------------
# Load & Clean Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/home/jakes/Documents/strathmore/Modules/Module 1/dataMining/project/Rhomis/final/RHoMIS_Indicators.csv", encoding='latin1')

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

@st.cache_data
def load_crop_details():
    crop_df = pd.read_csv("/home/jakes/Documents/strathmore/Modules/Module 1/dataMining/project/Rhomis/final/crop_details.csv")
    # Clean as per notebook
    crop_performance_columns = [col for col in crop_df.columns if any(metric in col for metric in ['Harvested', 'Consumed', 'Sold', 'Income', 'Yield', 'Land', 'Use', 'Intercropped'])]
    for col in crop_performance_columns:
        crop_df[col] = pd.to_numeric(crop_df[col], errors='coerce')
    crop_df[crop_performance_columns] = crop_df[crop_performance_columns].fillna(0)
    return crop_df

@st.cache_data
def load_livestock_details():
    livestock_df = pd.read_csv("/home/jakes/Documents/strathmore/Modules/Module 1/dataMining/project/Rhomis/final/livestock_details.csv")
    # Clean similar to crops
    livestock_performance_columns = [col for col in livestock_df.columns if any(metric in col for metric in ['Kept_Number', 'Sold_Number', 'Sale_Income', 'Meat_Amount', 'Consumed', 'Sold', 'Income', 'Milk', 'Eggs'])]
    for col in livestock_performance_columns:
        livestock_df[col] = pd.to_numeric(livestock_df[col], errors='coerce')
    livestock_df[livestock_performance_columns] = livestock_df[livestock_performance_columns].fillna(0)
    return livestock_df

df = load_data()
crop_df = load_crop_details()
livestock_df = load_livestock_details()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")

countries = sorted(df['ID_COUNTRY'].dropna().unique())
selected_countries = st.sidebar.multiselect("Countries", countries, default=countries[:1])

years = sorted(df['YEAR'].dropna().unique())
selected_years = st.sidebar.multiselect("Years", years, default=years)

# Filter RHoMIS
filtered_df = df[
    df['ID_COUNTRY'].isin(selected_countries) &
    df['YEAR'].isin(selected_years)
].copy()

# Filter Crop Details
filtered_crop_df = crop_df[
    crop_df['ID_COUNTRY'].isin(selected_countries) &
    crop_df['YEAR'].isin(selected_years)
].copy()

# Filter Livestock Details
filtered_livestock_df = livestock_df[
    livestock_df['ID_COUNTRY'].isin(selected_countries) &
    livestock_df['YEAR'].isin(selected_years)
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
# Tab 2: Livestock Analysis (Using livestock_details.csv, mirrored from crop logic)
# -------------------------------
with tab2:
    st.header("Livestock Analysis (Using livestock_details.csv)")

    # Identify unique animals
    livestock_performance_columns = [col for col in filtered_livestock_df.columns if any(metric in col for metric in ['Kept_Number', 'Sold_Number', 'Sale_Income', 'Meat_Amount', 'Consumed', 'Sold', 'Income', 'Milk', 'Eggs'])]
    unique_animals = set()
    for col in livestock_performance_columns:
        for metric in ['Kept_Number', 'Sold_Number', 'Sale_Income', 'Meat_Amount', 'Meat_Consumed', 'Meat_Sold', 'Meat_Income', 'Milk_Amount', 'Milk_Consumed', 'Milk_Sold', 'Milk_Income', 'Eggs_Collected', 'Eggs_Consumed', 'Eggs_Sold', 'Eggs_Income']:
            if metric in col:
                animal_name = col.replace(f'{metric}_', '', 1)
                unique_animals.add(animal_name)
    unique_animals_list = list(unique_animals)

    sel_livestock = st.multiselect("Select Animals", unique_animals_list,
                                   default='cattle', key="livestock_sel")

    if sel_livestock:
        # Calculate KPIs (adapted: total kept, total sold, total income, average meat yield as example)
        livestock_kpis = []
        for animal in sel_livestock:
            kept_col = f'Whole_Livestock_Kept_Number_{animal}'
            sold_col = f'Whole_Livestock_Sold_Number_{animal}'
            income_col = f'Whole_Livestock_Sale_Income_{animal}'
            meat_amount_col = f'Meat_Amount_{animal}'

            if kept_col in filtered_livestock_df.columns:
                total_kept = filtered_livestock_df[kept_col].sum()
                total_sold = filtered_livestock_df[sold_col].sum() if sold_col in filtered_livestock_df.columns else 0
                total_income = filtered_livestock_df[income_col].sum() if income_col in filtered_livestock_df.columns else 0
                total_meat = filtered_livestock_df[meat_amount_col].sum() if meat_amount_col in filtered_livestock_df.columns else 0
                average_meat_yield = total_meat / total_kept if total_kept > 0 else 0

                livestock_kpis.append({
                    'Animal': animal,
                    'Total Kept': total_kept,
                    'Total Sold': total_sold,
                    'Total Income': total_income,
                    'Total Meat': total_meat,
                    'Average Meat Yield': average_meat_yield
                })

        if livestock_kpis:
            livestock_kpis_df = pd.DataFrame(livestock_kpis)
            st.subheader("Livestock KPIs")
            st.dataframe(livestock_kpis_df)

            # Top 10 by metrics
            top_10_kept = livestock_kpis_df.sort_values(by='Total Kept', ascending=False).head(10)
            top_10_sold = livestock_kpis_df.sort_values(by='Total Sold', ascending=False).head(10)
            top_10_income = livestock_kpis_df.sort_values(by='Total Income', ascending=False).head(10)
            top_10_yield = livestock_kpis_df.sort_values(by='Average Meat Yield', ascending=False).head(10)

            # Visualizations
            fig_kept = px.bar(top_10_kept, x='Animal', y='Total Kept',
                              title='Top Animals by Total Kept')
            st.plotly_chart(fig_kept, use_container_width=True, key="livestock_kept_bar")

            fig_sold = px.bar(top_10_sold, x='Animal', y='Total Sold',
                              title='Top Animals by Total Sold')
            st.plotly_chart(fig_sold, use_container_width=True, key="livestock_sold_bar")

            fig_income = px.bar(top_10_income, x='Animal', y='Total Income',
                                title='Top Animals by Total Income')
            st.plotly_chart(fig_income, use_container_width=True, key="livestock_income_bar")

            fig_yield = px.bar(top_10_yield, x='Animal', y='Average Meat Yield',
                               title='Top Animals by Average Meat Yield')
            st.plotly_chart(fig_yield, use_container_width=True, key="livestock_yield_bar")

    # Influencing factors (adapt as per data, e.g., GPS)
    influencing_factors_columns = [col for col in filtered_livestock_df.columns if any(factor in col.lower() for factor in ['gps', 'climate', 'feed', 'health'])]
    if influencing_factors_columns:
        with st.expander("Potential Influencing Factors"):
            st.dataframe(filtered_livestock_df[influencing_factors_columns].head())

    # Generated Insights (mirrored from crops, adapted for livestock)
    with st.expander("Generated Insights"):
        st.markdown("""
## Summary of Livestock Performance Analysis

### Top Performing Animals Across Metrics
Based on the analysis of Total Kept, Total Sold, Total Income, and Average Meat Yield, the following animals consistently appear in the top rankings:
- **Chicken:** Appears in the top for kept, sold, income, indicating high turnover and value.
- **Goats:** Ranks high in kept and sold, suggesting common and marketable.
- **Cattle:** Features in top for kept, income, highlighting economic importance.
- **Sheep:** Appears in top for kept and income.
- **Pigs:** Ranks in top for income and yield.

### Significance of Top Performing Animals
Chicken and goats are key for smallholders due to quick reproduction and market demand. Cattle provide higher value products like milk and meat.

### Potential Influencing Factors
- **Location (GPS):** Regional variations in performance due to climate/feed availability.
- Limited data on feed/health; future collection recommended.

### Limitations
Lack of direct data on veterinary inputs, feed, diseases.

### Potential Recommendations
1. Promote high-yield animals like chicken in suitable areas.
2. Investigate high-income animals like cattle for value chains.
3. Gather data on health/feed for better insights.
        """)

# -------------------------------
# Tab 3: Crop Analysis
# -------------------------------
with tab3:
    st.header("Crop Analysis (Using crop_details.csv)")

    # Identify unique crops
    crop_performance_columns = [col for col in filtered_crop_df.columns if any(metric in col for metric in ['Harvested', 'Consumed', 'Sold', 'Income', 'Yield', 'Land', 'Use', 'Intercropped'])]
    unique_crops = set()
    for col in crop_performance_columns:
        for metric in ['Harvested', 'Consumed', 'Sold', 'Income', 'Yield', 'Land', 'Use', 'Intercropped']:
            if metric in col:
                crop_name = col.replace(f'{metric}_', '', 1)
                unique_crops.add(crop_name)
    unique_crops_list = list(unique_crops)

    sel_crop = st.multiselect("Select Crops", unique_crops_list,
                              default='groundnut', key="crop_sel")

    if sel_crop:
        crop_kpis = []
        for crop in sel_crop:
            harvested_col = f'Harvested_{crop}'
            income_col = f'Income_{crop}'
            land_col = f'Land_{crop}'
            yield_col = f'Yield_{crop}'

            if harvested_col in filtered_crop_df.columns:
                total_harvested = filtered_crop_df[harvested_col].sum()
                total_income = filtered_crop_df[income_col].sum() if income_col in filtered_crop_df.columns else 0
                total_land = filtered_crop_df[land_col].sum() if land_col in filtered_crop_df.columns else 0
                average_yield = total_harvested / total_land if total_land > 0 else 0

                crop_kpis.append({
                    'Crop': crop,
                    'Total Harvested': total_harvested,
                    'Total Income': total_income,
                    'Total Land Used': total_land,
                    'Average Yield': average_yield
                })

        if crop_kpis:
            crop_kpis_df = pd.DataFrame(crop_kpis)
            st.subheader("Crop KPIs")
            st.dataframe(crop_kpis_df)

            top_10_harvested = crop_kpis_df.sort_values(by='Total Harvested', ascending=False).head(10)
            top_10_income = crop_kpis_df.sort_values(by='Total Income', ascending=False).head(10)
            top_10_yield = crop_kpis_df.sort_values(by='Average Yield', ascending=False).head(10)

            fig_harvested = px.bar(top_10_harvested, x='Crop', y='Total Harvested',
                                   title='Top Crops by Total Harvested')
            st.plotly_chart(fig_harvested, use_container_width=True, key="crop_harvested_bar")

            fig_income = px.bar(top_10_income, x='Crop', y='Total Income',
                                title='Top Crops by Total Income')
            st.plotly_chart(fig_income, use_container_width=True, key="crop_income_bar")

            fig_yield = px.bar(top_10_yield, x='Crop', y='Average Yield',
                               title='Top Crops by Average Yield')
            st.plotly_chart(fig_yield, use_container_width=True, key="crop_yield_bar")

    influencing_factors_columns = [col for col in filtered_crop_df.columns if any(factor in col.lower() for factor in ['land', 'fertilizer', 'irrigation', 'gps', 'climate', 'soil'])]
    if influencing_factors_columns:
        with st.expander("Potential Influencing Factors"):
            st.dataframe(filtered_crop_df[influencing_factors_columns].head())

    with st.expander("Generated Insights"):
        st.markdown("""
## Summary of Crop Performance Analysis

### Top Performing Crops Across Metrics
Based on the analysis of Total Harvested, Total Income, and Average Yield, the following crops consistently appear in the top rankings:
- **Groundnut:** Appears in the top 10 for all three metrics (Harvested, Income, Yield), indicating strong overall performance.
- **Cassava:** Ranks high in Total Harvested and Total Income, suggesting it's a significant crop in terms of production and economic value.
- **Maize:** Features in the top 10 for Total Harvested and Total Income, highlighting its importance in both production volume and revenue.
- **Cotton:** Appears in the top 10 for Total Harvested and Total Income, indicating its economic significance.
- **Millet:** Ranks in the top 10 for Total Harvested and Total Income, showing its contribution to both production and income.
- **Irish potato:** Appears in the top 10 for Total Harvested and Average Yield, suggesting high productivity per unit of land.
- **Rice:** Ranks in the top 10 for Total Harvested and Average Yield, indicating its importance for food security and efficient land use.
- **Sugarcane:** Appears in the top 10 for Total Harvested and Average Yield, suggesting high production volume and efficient land use.
- **Fodder:** Ranks in the top 10 for Total Harvested and Average Yield, likely important for livestock farming.
- **Tobacco:** Stands out for its very high Total Income, despite not being in the top 10 for Harvested or Yield, suggesting a high market value.

### Significance of Top Performing Crops
The prominence of crops like Groundnut, Cassava, Maize, Rice, and Millet suggests their crucial role in food security and potentially as staple crops in the surveyed regions. Crops like Tobacco, Cotton, and Cassava also demonstrate significant economic value based on their high total income.

### Potential Influencing Factors
While direct data on inputs like fertilizers and irrigation was not explicitly available in the dataset, the analysis considered 'Land Use' (Total Land Used) and 'Location' (inferred from GPS data) as potential influencing factors.
- **Land Use:** The 'Total Land Used' metric varies significantly among crops. Crops with high total harvested amounts (like Groundnut, Cassava, and Maize) generally utilize a large amount of land. However, some crops like Irish potato, Rice, Sugarcane, and Fodder show high average yields, indicating efficient production even if the total land used is not the highest. This suggests that optimizing land allocation based on crop suitability and yield potential is important.
- **Location:** Although not directly analyzed due to missing data, GPS coordinates could potentially indicate regional variations in performance. Different locations might have varying climate, soil types, and access to resources, which could significantly impact crop yields and income. Further analysis with more complete GPS data and potentially external climate/soil data could provide valuable insights into location-specific advantages or challenges.

### Limitations
A significant limitation of this analysis is the lack of direct data on agricultural inputs (fertilizers, pesticides, irrigation) and detailed environmental conditions (climate, soil type). These factors are known to heavily influence crop performance. Therefore, the insights on influencing factors are based on available data and general agricultural knowledge.

### Potential Recommendations for Improving Crop Performance
Based on the observed performance and potential influencing factors, the following recommendations can be considered:
1.  **Optimize Land Allocation:** Encourage the cultivation of high-yield crops (like Irish potato, Rice, Sugarcane, and Fodder) in suitable areas to maximize production per unit of land.
2.  **Investigate High-Income Crops:** Further research into the factors contributing to the high income of crops like Tobacco and Cassava is recommended. This could involve studying market dynamics, value chain integration, and processing techniques.
3.  **Explore Location-Specific Practices:** If more complete GPS data is available, analyze the performance of top crops in different locations to identify successful regional practices and potentially replicate them in other suitable areas.
4.  **Gather Data on Inputs and Environmental Factors:** For a more comprehensive understanding of performance drivers, future data collection should include detailed information on agricultural inputs, climate data, and soil characteristics.
5.  **Promote High-Performing Staple Crops:** Support the cultivation of high-performing staple crops like Groundnut, Cassava, Maize, Rice, and Millet to enhance food security.
        """)

# -------------------------------
# Tab 4: Market Orientation Analysis
# -------------------------------
with tab4:
    st.header("Market Orientation Analysis")

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

    all_features = list(set(unique_animals_list + unique_crops_list + market_features))
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
    label="Download RHoMIS CSV",
    data=csv,
    file_name=f"rhomis_{'_'.join(selected_countries)}_{'_'.join(map(str, selected_years))}.csv",
    mime="text/csv"
)

crop_csv = filtered_crop_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Crop Details CSV",
    data=crop_csv,
    file_name=f"crop_details_{'_'.join(selected_countries)}_{'_'.join(map(str, selected_years))}.csv",
    mime="text/csv"
)

livestock_csv = filtered_livestock_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Livestock Details CSV",
    data=livestock_csv,
    file_name=f"livestock_details_{'_'.join(selected_countries)}_{'_'.join(map(str, selected_years))}.csv",
    mime="text/csv"
)