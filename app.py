import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Smart Streetlight Dashboard",
    page_icon="💡",
    layout="wide"
)

# ==================================================
# DARK THEME STYLE
# ==================================================
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        h1, h2, h3, h4 {
            color: white;
        }
        .stMetric {
            background-color: #1c1f26;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD TRAINED MODEL
# ==================================================
model = joblib.load("energy_model.pkl")

# ==================================================
# TITLE
# ==================================================
st.title("💡 Smart Streetlight Energy Dashboard")
st.markdown("### AI-Based Energy Optimization System for Smart Cities")

# ==================================================
# SIDEBAR INPUTS
# ==================================================
st.sidebar.header("⚙ Streetlight Parameters")

zone_id = st.sidebar.slider("Zone ID", 1, 5, 3)

area_type = st.sidebar.selectbox(
    "Area Type",
    ["Residential (0)", "Commercial (1)", "Industrial (2)"]
)
area_type = int(area_type.split("(")[1].replace(")", ""))

hour = st.sidebar.slider("Hour", 0, 23, 12)
day_of_week = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 3)
month = st.sidebar.slider("Month", 1, 12, 6)

temperature = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 60.0)

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear (0)", "Cloudy (1)", "Rainy (2)", "Storm (3)"]
)
weather_condition = int(weather_condition.split("(")[1].replace(")", ""))

traffic_density = st.sidebar.slider("Traffic Density", 0.0, 100.0, 50.0)
pedestrian_density = st.sidebar.slider("Pedestrian Density", 0.0, 100.0, 50.0)
light_intensity = st.sidebar.slider("Light Intensity", 0.0, 100.0, 70.0)
lamp_age = st.sidebar.slider("Lamp Age (Years)", 0.0, 5.0, 2.0)
maintenance_score = st.sidebar.slider("Maintenance Score", 50.0, 100.0, 80.0)

# ==================================================
# CREATE INPUT DATAFRAME (FIXES WARNING)
# ==================================================
feature_names = [
    "zone_id", "area_type", "hour", "day_of_week",
    "month", "temperature", "humidity",
    "weather_condition", "traffic_density",
    "pedestrian_density", "light_intensity",
    "lamp_age", "maintenance_score"
]

input_df = pd.DataFrame([[
    zone_id, area_type, hour, day_of_week,
    month, temperature, humidity,
    weather_condition, traffic_density,
    pedestrian_density, light_intensity,
    lamp_age, maintenance_score
]], columns=feature_names)

prediction = model.predict(input_df)[0]

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "⏳ Time Series"])

# ==================================================
# TAB 1 - PREDICTION
# ==================================================
with tab1:
    st.subheader("Predicted Energy Consumption")

    st.metric("Energy Usage (kWh)", f"{prediction:.2f}")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Energy Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "cyan"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

# ==================================================
# TAB 2 - ANALYTICS
# ==================================================
with tab2:
    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_
    else:
        importance = None

    if importance is not None:
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=True)

        fig = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance not available for this model.")

# ==================================================
# TAB 3 - REAL-TIME SIMULATION
# ==================================================
with tab3:
    st.subheader("Real-Time Energy Simulation")

    chart_placeholder = st.empty()
    data = []

    for i in range(30):
        simulated_value = prediction + np.random.normal(0, 2)
        data.append(simulated_value)

        df_sim = pd.DataFrame({
            "Time": list(range(len(data))),
            "Energy": data
        })

        fig = px.line(
            df_sim,
            x="Time",
            y="Energy",
            title="Live Energy Usage Trend",
            markers=True
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(0.2)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown("🚀 Smart City Energy Optimization Dashboard | Streamlit + ML + Plotly")
