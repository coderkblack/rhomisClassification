import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from pathlib import Path

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="RHOMIS Food Security Prediction 🚜", layout="wide")

# ==========================
# LOAD DATA + MODEL
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("RHoMIS_Indicators.csv", encoding="latin1")  # 👈 adjust path
    return df

# @st.cache_resource
# def load_model():
#     model = joblib.load("/home/jakes/Documents/strathmore/dataMining/project/Rhomis/final/rhomis_rf (1).pkl")  # 👈 adjust path
#     return model

@st.cache_resource
def load_models():
    base_url = "https://huggingface.co/coderkblack/rhomis-model/resolve/main/"
    model_files = {
        "Random Forest": "rhomis_small_rf (2).pkl",
        "XGBoost": "rhomis_small_xgb.pkl",
        "LightGBM": "rhomis_small_lgbm.pkl"
    }

    models = {}
    for name, filename in model_files.items():
        local_path = Path(filename)
        if not local_path.exists():  # Download if missing
            url = base_url + filename
            r = requests.get(url)
            local_path.write_bytes(r.content)
        models[name] = joblib.load(local_path)
    return models

# ==========================
# APP HEADER
# ==========================
st.title("🌾 RHOMIS Food Security Prediction Tool")
st.markdown("Predict **household food security status** using RHOMIS indicators.")

# ==========================
# SIDEBAR NAVIGATION
# ==========================
menu = st.sidebar.radio("📌 Navigation", ["Dataset Overview", "Model Performance", "Feature Importance", "Prediction Tool", "Recommendations"])

df = load_data()
models = load_models()
selected_model_name = st.sidebar.selectbox("🤖 Choose Model", list(models.keys()))
model = models[selected_model_name]

# ==========================
# 1. DATASET OVERVIEW
# ==========================
if menu == "📊 Dataset Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)

if menu == "Dataset Overview":
    st.header("📊 Dataset Overview")
    st.write("Preview of the dataset:")
    st.dataframe(df.head())

    st.subheader("Food Security Distribution (HFIAS_status)")
    fig1 = px.histogram(df, x="HFIAS_status", color="HFIAS_status", barmode="group")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Country-wise Household Counts")
    fig2 = px.histogram(df, x="Country", color="Country")
    st.plotly_chart(fig2, use_container_width=True)

    # Compute distribution of categories by country
    country_status = df.groupby(["Country", "HFIAS_status"]).size().reset_index(name="count")
    country_totals = df.groupby("Country").size().reset_index(name="total")
    country_status = country_status.merge(country_totals, on="Country")
    country_status["percentage"] = country_status["count"] / country_status["total"]

    # Choropleth of % Food Secure
    food_secure = country_status[country_status["HFIAS_status"] == "FoodSecure"]

    fig = px.choropleth(
        food_secure,
        locations="Country",
        locationmode="country names",
        color="percentage",
        hover_name="Country",
        hover_data={"percentage": ":.2%"},
        color_continuous_scale="Greens",
        title="Percentage of Food Secure Households by Country"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stacked bar chart for all categories
    st.markdown("### 📊 Food Security Levels per Country")
    fig = px.bar(
        country_status,
        x="Country",
        y="percentage",
        color="HFIAS_status",
        title="Food Security Distribution by Country",
        labels={"percentage": "Percentage", "Country": "Country"},
        barmode="stack",
        color_discrete_map={
            "FoodSecure": "#2E8B57",
            "MildlyFI": "#FFD700",
            "ModeratelyFI": "#FF8C00",
            "SeverelyFI": "#DC143C"
        }
    )
    fig.update_layout(xaxis_tickangle=-45, yaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

    # 💰 Income vs Food Security
    st.markdown("### 💰 Income vs Food Security")
    fig = px.box(
        df,
        x="HFIAS_status",
        y="total_income_USD_PPP_pHH_Yr",
        color="HFIAS_status",
        title="Household Total Income Distribution by Food Security",
        color_discrete_map={
            "FoodSecure": "#2E8B57",
            "MildlyFI": "#FFD700",
            "ModeratelyFI": "#FF8C00",
            "SeverelyFI": "#DC143C"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # 🗓️ Seasonal Food Security - Worst & Best Months
    st.markdown("### 🗓️ Seasonal Food Security")
    df_clean = df.replace({'WorstFoodSecMonth': 'na', 'BestFoodSecMonth': 'na'}, np.nan)
    df_clean = df_clean.dropna(subset=['WorstFoodSecMonth', 'BestFoodSecMonth'])
    # Create a DataFrame for "Worst" months and rename the column
    worst_months_df = df_clean[['WorstFoodSecMonth']].assign(Season_Type='Worst').rename(columns={'WorstFoodSecMonth': 'Month'})

    # Create a DataFrame for "Best" months and rename the column
    best_months_df = df_clean[['BestFoodSecMonth']].assign(Season_Type='Best').rename(columns={'BestFoodSecMonth': 'Month'})

    # Concatenate the two DataFrames with unique column names
    seasonal_df = pd.concat([worst_months_df, best_months_df])

    fig = px.histogram(
        seasonal_df,
        x="Month",
        color="Season_Type",
        barmode="group",
        title="Distribution of Best and Worst Food Secure Months",
        labels={"Month": "Month of the Year", "Season_Type": "Season Type"},
        category_orders={"Month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 🍽️ Months Food Insecure Distribution
    st.markdown("### 🍽️ Months Food Insecure Distribution")
    fig = px.violin(
        df,
        x="HFIAS_status",
        y="NrofMonthsFoodInsecure",
        color="HFIAS_status",
        box=True,
        points="all",
        title="Distribution of Food Insecurity Months by Status",
        color_discrete_map={
            "FoodSecure": "#2E8B57",
            "MildlyFI": "#FFD700",
            "ModeratelyFI": "#FF8C00",
            "SeverelyFI": "#DC143C"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # 🐄 Livestock by country & security
    st.markdown("### 🐄 Livestock Holdings by Country and Food Security")
    livestock_stats = df.groupby(["Country", "HFIAS_status"])["LivestockHoldings"].mean().reset_index()
    fig = px.bar(
        livestock_stats,
        x="Country",
        y="LivestockHoldings",
        color="HFIAS_status",
        barmode="group",
        title="Average Livestock Holdings by Country & Food Security",
        color_discrete_map={
            "FoodSecure": "#2E8B57",
            "MildlyFI": "#FFD700",
            "ModeratelyFI": "#FF8C00",
            "SeverelyFI": "#DC143C"
        }
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ==========================
# 2. MODEL PERFORMANCE
# ==========================
elif menu == "Model Performance":
    st.header("📈 Model Performance")

    # Store evaluation results for each model
    model_metrics = {
        "Random Forest": {
            "Accuracy": 0.80,
            "Macro F1-score": 0.80,
            "ROC AUC": 0.91,
            "Precision (avg)": 0.82,
            "Recall (avg)": 0.79
        },
        "LightGBM": {
            "Accuracy": 0.81,
            "Macro F1-score": 0.78,
            "ROC AUC": 0.87,
            "Precision (avg)": 0.80,
            "Recall (avg)": 0.77
        },
        "XGBoost": {
            "Accuracy": 0.82,
            "Macro F1-score": 0.79,
            "ROC AUC": 0.87,
            "Precision (avg)": 0.81,
            "Recall (avg)": 0.78
        },
    }

    # Get metrics for selected model
    metrics = model_metrics[selected_model_name]

    # Gauge plot for Accuracy
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics["Accuracy"] * 100,
        title={'text': f"{selected_model_name} Accuracy (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Show metrics in JSON format
    st.write("### 📊 Evaluation Report")
    st.json(metrics)

# ==========================
# 3. FEATURE IMPORTANCE
# ==========================
# elif menu == "Feature Importance":
#     st.header("🔑 Feature Importance (Random Forest)")
    
#     importances = model.named_steps['classifier'].feature_importances_
#     features = model.named_steps['preprocessor'].get_feature_names_out()

#     fi = pd.DataFrame({
#         "Feature": features,
#         "Importance": importances
#     }).sort_values(by="Importance", ascending=False).head(20)

#     fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Top 20 Important Features")
#     st.plotly_chart(fig, use_container_width=True)
#     st.dataframe(fi) 

elif menu == "Feature Importance":

    # Extract feature importances
    importances = model.named_steps['classifier'].feature_importances_
    features = model.named_steps['preprocessor'].get_feature_names_out()

    fi = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(20)

    # --- Display results ---
    st.subheader(f"📊 {selected_model_name}")
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 title=f"Top 20 Important Features – {selected_model_name}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fi)


# ==========================
# 4. PREDICTION TOOL
# ==========================
elif menu == "Prediction Tool":
    st.header("🧮 Make a Prediction")

    # --- Input fields with refined inline descriptions ---
    HHsizeMAE = st.number_input("Household Size (MAE)", min_value=1, value=4)
    st.caption("Household size expressed in Male Adult Equivalents (MAE), adjusting for age and consumption needs.")

    NrofMonthsFoodInsecure = st.number_input("Months Food Insecure", min_value=0, max_value=12, value=2)
    st.caption("Number of months in the past year when the household experienced a shortage of food.")

    PPI_Likelihood = st.slider("PPI Likelihood", 0.0, 100.0, 50.0)
    st.caption("Estimated probability (%) that the household will move out of poverty, based on the Poverty Probability Index (PPI).")

    Food_Self_Sufficiency = st.number_input("Food Self Sufficiency (kCal MAE/day)", min_value=0, value=2000)
    st.caption("Daily kilocalories per Male Adult Equivalent (MAE) that were produced and consumed by the household itself.")

    LivestockHoldings = st.number_input("Livestock Holdings (TLU)", min_value=0, value=5)
    st.caption("Total livestock owned, converted into Tropical Livestock Units (TLU) for comparability across species.")

    TVA_USD = st.number_input("TVA (USD PPP per MAE per day)", min_value=0.0, value=1.5)
    st.caption("Total Value of all household activities in Purchasing Power Parity (PPP) USD per MAE per day. Reflects productivity and value generation.")

    farm_income = st.number_input("Farm Income (USD PPP per HH per Yr)", min_value=0.0, value=500.0)
    st.caption("Total annual income from farming activities (crop and livestock sales), adjusted to USD PPP.")

    total_income = st.number_input("Total Income (USD PPP per HH per Yr)", min_value=0.0, value=1000.0)
    st.caption("Annual household income from all sources (farm, livestock, and off-farm), in USD PPP.")

    Country = st.selectbox("Country", sorted(df["Country"].unique()))
    st.caption("Full name of the country where the household was surveyed.")

    HouseholdType = st.selectbox("Household Type", df["HouseholdType"].unique())
    st.caption("Type of household, categorized by marital status of the household head (e.g., single man, single woman, couple).")

    # --- Create input dataframe ---
    input_data = pd.DataFrame({
        "HHsizeMAE": [HHsizeMAE],
        "NrofMonthsFoodInsecure": [NrofMonthsFoodInsecure],
        "PPI_Likelihood": [PPI_Likelihood],
        "Food_Self_Sufficiency_kCal_MAE_day": [Food_Self_Sufficiency],
        "LivestockHoldings": [LivestockHoldings],
        "TVA_USD_PPP_pmae_pday": [TVA_USD],
        "farm_income_USD_PPP_pHH_Yr": [farm_income],
        "total_income_USD_PPP_pHH_Yr": [total_income],
        "HouseholdType": [HouseholdType],
        "Country": [Country]
    })

    # --- Predict ---
    if st.button("Predict Food Security"):
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data).max() * 100

        label_map = {0: "Insecure", 1: "Secure"}
        pred_label = label_map.get(pred, str(pred))

        st.success(f"✅ Predicted Food Security ({selected_model_name}): {pred_label}")
        st.info(f"Confidence: {proba:.2f}%")

        # Download option
        csv = input_data.assign(Predicted_FoodSecurity=pred_label)
        st.download_button("📥 Download Prediction", csv.to_csv(index=False), "prediction.csv", "text/csv")
