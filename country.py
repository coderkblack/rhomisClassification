import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="RHoMIS Dataset Explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Data Loading
# -------------------------------
@st.cache_data
def load_data():
    # Update path if necessary
    file_path = "/home/jakes/Documents/strathmore/Modules/Module 1/dataMining/project/Rhomis/final/RHoMIS_Indicators.csv"
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Please check the path.")
        return pd.DataFrame()

    # Numeric columns conversion
    numeric_cols = [
        'HHsizemembers', 'HHsizeMAE', 'LandCultivated', 'LivestockHoldings',
        'NrofMonthsFoodInsecure', 'PPI_Likelihood', 'score_HDDS_GoodSeason',
        'score_HDDS_BadSeason', 'total_income_USD_PPP_pHH_Yr', 'offfarm_income_USD_PPP_pHH_Yr',
        'farm_income_USD_PPP_pHH_Yr', 'Food_Availability_kCal_MAE_day',
        'GHGEmissions', 'NFertInput', 'value_farm_produce_USD_PPP_pHH_Yr', 'crop_sales_USD_PPP_pHH_Yr'
    ]
    
    # Ensure columns exist before conversion
    existing_cols = [col for col in numeric_cols if col in df.columns]
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

df = load_data()

if df.empty:
    st.stop()

# -------------------------------
# Sidebar & Filters
# -------------------------------
st.sidebar.title("Configuration")
st.sidebar.markdown("Filter the dataset by country and year.")


# Country Filter
countries = sorted(df['ID_COUNTRY'].dropna().unique())

# Select All checkbox
select_all_countries = st.sidebar.checkbox("Select All Countries", value=False)

if select_all_countries:
    selected_countries = countries
else:
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=countries,
        default=countries[:1] if len(countries) > 0 else []
    )

# Year Filter
years = sorted(df['YEAR'].dropna().unique())
selected_years = st.sidebar.multiselect(
    "Select Years",
    options=years,
    default=years
)

# Apply Filters
filtered_df = df[
    df['ID_COUNTRY'].isin(selected_countries) &
    df['YEAR'].isin(selected_years)
].copy()

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# -------------------------------
# Main Layout
# -------------------------------
st.title("RHoMIS Dataset Explorer")
st.markdown("Explore household indicators, agricultural productivity, and food security metrics across different regions.")

# Tabs
tab_overview, tab_stats, tab_dist, tab_rel, tab_map, tab_model = st.tabs([
    "Overview", "Summary Statistics", "Distributions", "Relationships", "Comparisons & Map", "Modelling"
])

# -------------------------------
# Tab 1: Overview
# -------------------------------
with tab_overview:
    st.subheader("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Households", f"{len(filtered_df):,}")
    with col2:
        st.metric("Countries Selected", len(selected_countries))
    with col3:
        avg_hh_size = filtered_df['HHsizemembers'].mean()
        st.metric("Avg HH Size", f"{avg_hh_size:.1f}")
    with col4:
        avg_land = filtered_df['LandCultivated'].mean()
        st.metric("Avg Land Cultivated (ha)", f"{avg_land:.2f}")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    st.markdown("### Download Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered CSV",
        data=csv,
        file_name="rhomis_filtered_data.csv",
        mime="text/csv"
    )

# -------------------------------
# Tab 2: Summary Statistics
# -------------------------------
with tab_stats:
    st.subheader("Numeric Summary")
    
    numeric_features = [
        'HHsizemembers', 'HHsizeMAE', 'LandCultivated', 'LivestockHoldings',
        'NrofMonthsFoodInsecure', 'score_HDDS_GoodSeason', 'total_income_USD_PPP_pHH_Yr',
        'GHGEmissions', 'NFertInput'
    ]
    # Filter to only existing columns
    numeric_features = [col for col in numeric_features if col in filtered_df.columns]
    
    sel_num = st.multiselect("Select Numeric Features", numeric_features, default=numeric_features[:3])
    
    if sel_num:
        st.dataframe(filtered_df[sel_num].describe(), use_container_width=True)

    st.markdown("---")
    st.subheader("Categorical Summary")
    cat_features = ['HouseholdType', 'Head_EducationLevel', 'HFIAS_status']
    # Filter to only existing columns
    cat_features = [col for col in cat_features if col in filtered_df.columns]

    sel_cat = st.multiselect("Select Categorical Features", cat_features, default=cat_features[:2])
    
    if sel_cat:
        cols = st.columns(len(sel_cat))
        for idx, cat in enumerate(sel_cat):
            with cols[idx]:
                st.markdown(f"**{cat}**")
                st.dataframe(filtered_df[cat].value_counts(), use_container_width=True)

# -------------------------------
# Tab 3: Distributions
# -------------------------------
with tab_dist:
    st.subheader("Feature Distributions")
    
    dist_num = st.multiselect(
        "Select Features to Visualize", numeric_features, default=numeric_features[:2], key="dist_num"
    )
    
    for feat in dist_num:
        st.markdown(f"#### Distribution of {feat}")
        col_hist, col_box = st.columns(2)
        
        with col_hist:
            fig_hist = px.histogram(
                filtered_df, x=feat, color='ID_COUNTRY',
                marginal="box", 
                title=f"Histogram of {feat}",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_hist.update_layout(legend_title_text='Country', xaxis_title=feat, yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_box:
            fig_box = px.box(
                filtered_df, x='ID_COUNTRY', y=feat, 
                points="outliers",
                title=f"Box Plot of {feat} by Country",
                template="plotly_white",
                color='ID_COUNTRY',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_box.update_layout(showlegend=False, xaxis_title="Country", yaxis_title=feat)
            st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("---")

# -------------------------------
# Tab 4: Relationships
# -------------------------------
with tab_rel:
    st.subheader("Correlation Analysis")
    
    rel_num = st.multiselect(
        "Select Features for Correlation Matrix", numeric_features, default=numeric_features[:5], key="rel_num"
    )
    
    if len(rel_num) > 1:
        corr = filtered_df[rel_num].corr()
        fig_corr = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap",
            template="plotly_white"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("Scatter Plot Analysis")
    
    col_x, col_y = st.columns(2)
    with col_x:
        x_feat = st.selectbox("X-axis Feature", numeric_features, index=0)
    with col_y:
        y_feat = st.selectbox("Y-axis Feature", numeric_features, index=1 if len(numeric_features) > 1 else 0)
        
    if x_feat and y_feat:
        fig_scatter = px.scatter(
            filtered_df, x=x_feat, y=y_feat,
            color='ID_COUNTRY', 
            size='HHsizemembers' if 'HHsizemembers' in filtered_df.columns else None,
            hover_data=['ID_HH'] if 'ID_HH' in filtered_df.columns else None,
            title=f"{x_feat} vs {y_feat}",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe,
            opacity=0.7
        )
        fig_scatter.update_layout(legend_title_text='Country')
        st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------
# Tab 5: Comparisons & Map
# -------------------------------
with tab_map:
    st.subheader("Country Comparisons")
    
    if len(selected_countries) > 0:
        avg_cols = st.multiselect("Features to Compare (Average)", numeric_features, default=numeric_features[:3], key="avg_cols")
        if avg_cols:
            avg_df = filtered_df.groupby('ID_COUNTRY')[avg_cols].mean().reset_index()
            melted_avg = avg_df.melt(id_vars='ID_COUNTRY', var_name='Feature', value_name='Average Value')
            
            fig_bar = px.bar(
                melted_avg, x='ID_COUNTRY', y='Average Value',
                color='Feature', 
                barmode='group', 
                title="Average Feature Values by Country",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_bar.update_layout(xaxis_title="Country", legend_title_text='Feature')
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Select at least one country to view comparisons.")

    st.markdown("---")
    st.subheader("Geographic Distribution")
    
    # Check for GPS columns
    if 'GPS_LAT' in filtered_df.columns and 'GPS_LON' in filtered_df.columns:
        map_df = filtered_df.dropna(subset=['GPS_LAT', 'GPS_LON'])
        
        if not map_df.empty:
            # Determine size variable safely
            size_col = 'LandCultivated' if 'LandCultivated' in map_df.columns else None
            size_args = {}
            if size_col:
                # Ensure positive values for size and handle NaNs
                map_df['size_scaled'] = map_df[size_col].fillna(0.1).clip(lower=0.1)
                size_args = {'size': 'size_scaled'}

            fig_map = px.scatter_mapbox(
                map_df,
                lat='GPS_LAT', lon='GPS_LON',
                color='ID_COUNTRY',
                hover_data=[col for col in ['ID_HH', 'HFIAS_status', 'LandCultivated'] if col in map_df.columns],
                zoom=2,
                title="Household Locations",
                mapbox_style="open-street-map",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe,
                **size_args
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No valid GPS data available for the selected filters.")
    else:
        st.info("GPS coordinates (GPS_LAT, GPS_LON) not found in dataset.")

# -------------------------------
# Tab 6: Modelling
# -------------------------------
with tab_model:
    st.subheader("Food Security Prediction Model")
    st.markdown("Train a Random Forest model to predict if a household is **Food Secure** based on the filtered data.")

    # 1. Configuration
    col_config, col_params = st.columns(2)
    
    with col_config:
        st.markdown("#### Feature Selection")
        # Default features based on refined analysis (household-level only)
        default_features = [
            'NrofMonthsFoodInsecure', 'PPI_Likelihood', 'Food_Self_Sufficiency_kCal_MAE_day',
            'Food_Availability_kCal_MAE_day', 'LivestockHoldings', 'HHsizeMAE',
            'score_HDDS_GoodSeason', 'Head_EducationLevel', 'LandOwned', 'LandCultivated'
        ]
        
        # Exclude IDs, target, and country-level proxies
        excluded_features = [
            'HFIAS_status', 'ID_COUNTRY', 'ID_HH', 'GPS_LAT', 'GPS_LON', 'GPS_ALT',
            'YEAR', 'ITERATION', 'SURVEY_ID', 'ID_PROJ', 'Country', 'Region',
            'WorstFoodSecMonth', 'BestFoodSecMonth',
            'currency_conversion_factor', 'Altitude', 'TVA_USD_PPP_pmae_pday'  # Country-level proxies
        ]
        
        available_features = [col for col in filtered_df.columns if col not in excluded_features]
        
        # Pre-select defaults that are available
        pre_selected = [f for f in default_features if f in available_features]
        selected_features = st.multiselect("Select Predictors", available_features, default=pre_selected)

    with col_params:
        st.markdown("#### Model Hyperparameters")
        n_estimators = st.slider("Number of Trees", 10, 200, 100, step=10)
        max_depth = st.slider("Max Depth", 2, 20, 10)
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)

    # 2. Data Preparation & Training
    if st.button("Train Model", type="primary"):
        if not selected_features:
            st.error("Please select at least one feature.")
        elif len(filtered_df) < 50:
            st.error("Not enough data to train a model. Please select more countries/years.")
        else:
            with st.spinner("Training model..."):
                try:
                    # Prepare Data
                    model_df = filtered_df.dropna(subset=['HFIAS_status']).copy()
                    
                    if model_df.empty:
                        st.error("No data with valid HFIAS_status.")
                        st.stop()

                    # Target: 1 if FoodSecure, 0 otherwise
                    model_df['target'] = model_df['HFIAS_status'].apply(lambda x: 1 if x == 'FoodSecure' else 0)
                    
                    X = model_df[selected_features].copy()
                    y = model_df['target']
                    
                    # Handle Categorical
                    encoders = {}
                    for col in X.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        X[col] = X[col].astype(str)
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le
                    
                    # Handle Missing (Simple Imputation)
                    imputer = SimpleImputer(strategy='median')
                    X_imputed = imputer.fit_transform(X)
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
                    
                    # Train
                    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
                    rf.fit(X_train, y_train)
                    y_probs = rf.predict_proba(X_test)[:, 1]
                    
                    # Store in session state (we calculate metrics dynamically now)
                    st.session_state['rf_model'] = rf
                    st.session_state['rf_features'] = selected_features
                    st.session_state['rf_encoders'] = encoders
                    st.session_state['rf_imputer'] = imputer
                    st.session_state['rf_test_data'] = (y_test, y_probs)
                    st.session_state['rf_importances'] = rf.feature_importances_
                    st.session_state['rf_base_input'] = np.median(X_imputed, axis=0) # For simulator defaults
                    
                    st.success(f"Model Trained!")
                    
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")

    # 3. Display Results (if model exists)
    if 'rf_model' in st.session_state:
        y_test, y_probs = st.session_state['rf_test_data']
        importances = st.session_state['rf_importances']
        rf_features = st.session_state['rf_features']
        
        # Dynamic Prediction based on Threshold
        y_pred = (y_probs >= threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        st.markdown(f"**Model Accuracy:** {acc:.2%}")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("##### Confusion Matrix")
            fig_cm = px.imshow(cm, text_auto=True, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Insecure', 'Secure'], y=['Insecure', 'Secure'],
                               color_continuous_scale='Blues', template="plotly_white")
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("##### Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
            
        with col_res2:
            st.markdown("##### Feature Importance")
            feat_imp = pd.DataFrame({'Feature': rf_features, 'Importance': importances})
            feat_imp = feat_imp.sort_values(by='Importance', ascending=True)
            
            fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                             template="plotly_white", color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)

        # 4. Prediction Simulator
        st.markdown("---")
        st.subheader("Prediction Simulator")
        st.markdown("Adjust values to see the probability of being **Food Secure**.")
        
        # Get top 5 features for simulator
        top_features_df = feat_imp.sort_values(by='Importance', ascending=False).head(5)
        top_5_features = top_features_df['Feature'].tolist()
        
        sim_inputs = {}
        cols_sim = st.columns(len(top_5_features))
        
        base_input = st.session_state['rf_base_input']
        encoders = st.session_state['rf_encoders']
        
        for idx, feat in enumerate(top_5_features):
            # Find index in the original feature list
            feat_idx = rf_features.index(feat)
            col_idx = idx % 5
            
            with cols_sim[col_idx]:
                if feat in encoders:
                    # Categorical
                    classes = encoders[feat].classes_
                    # Default to median value class if possible, else first
                    default_int = int(base_input[feat_idx])
                    default_val = classes[default_int] if default_int < len(classes) else classes[0]
                    
                    val = st.selectbox(f"{feat}", classes, index=list(classes).index(default_val), key=f"sim_{feat}")
                    sim_inputs[feat_idx] = encoders[feat].transform([val])[0]
                else:
                    # Numeric
                    min_val = float(filtered_df[feat].min())
                    max_val = float(filtered_df[feat].max())
                    default_val = float(base_input[feat_idx])
                    
                    # Handle NaN in min/max
                    if np.isnan(min_val): min_val = 0.0
                    if np.isnan(max_val): max_val = 100.0
                    if np.isnan(default_val): default_val = min_val
                    
                    val = st.number_input(f"{feat}", min_value=min_val, max_value=max_val, value=default_val, key=f"sim_{feat}")
                    sim_inputs[feat_idx] = val

        # Construct full input vector
        final_input = base_input.copy()
        for idx, val in sim_inputs.items():
            final_input[idx] = val
        
        # Predict
        rf = st.session_state['rf_model']
        prob = rf.predict_proba([final_input])[0][1] # Probability of class 1 (Secure)
        
        st.metric("Probability of Food Security", f"{prob:.1%}")
        
        if prob > 0.5:
            st.success("Prediction: **Food Secure**")
        else:
            st.error("Prediction: **Food Insecure**")