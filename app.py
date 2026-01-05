import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_processor import load_and_merge_data
from analytics import calculate_metrics, generate_forecast

st.set_page_config(layout="wide", page_title="Vistara-AI | UIDAI Hackathon", page_icon="🇮🇳")

# --- Custom CSS for "Gold Standard" Aesthetics ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1E3A8A; /* Aadhaar Blue */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #555;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #fafafa;
    }
    .ai-insight-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1E3A8A;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
        color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Header ---
st.title("🇮🇳 Vistara-AI: Predictive Demographic Analytics")
st.markdown("**Unlocking Societal Trends in Aadhaar Enrolment and Updates (Powered by ML)**")

# --- Data Loading (Cached) ---
@st.cache_data
def get_data():
    with st.spinner('Loading and processing Aadhaar datasets across all districts...'):
        df = load_and_merge_data()
        df = calculate_metrics(df)
    return df

try:
    df_main = get_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if df_main.empty:
    st.warning("No data loaded. Please check the dataset files.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🕹️ Control Panel")

# Date Filter
min_date = df_main['date'].min()
max_date = df_main['date'].max()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

mask_date = (df_main['date'] >= pd.to_datetime(start_date)) & (df_main['date'] <= pd.to_datetime(end_date))
df_filtered = df_main.loc[mask_date]

# State Filter
all_states = sorted(df_filtered['state'].unique())
selected_states = st.sidebar.multiselect("Select State(s)", all_states, default=all_states[:1] if all_states else [])

if selected_states:
    df_filtered = df_filtered[df_filtered['state'].isin(selected_states)]

# Risk Level Filter (V2.0)
st.sidebar.markdown("### ⚠️ Risk Filter")
risk_options = ['High', 'Medium', 'Low']
selected_risks = st.sidebar.multiselect("Select Risk Level", risk_options, default=['High', 'Medium', 'Low'])

if selected_risks:
    df_filtered = df_filtered[df_filtered['risk_level'].isin(selected_risks)]

st.sidebar.markdown("---")
st.sidebar.info(f"**Data Loaded:** {len(df_filtered)} records\n\n**Districts:** {df_filtered['district'].nunique()}")

# Download Report Button (V2.0)
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "📥 Download Filtered Report",
    csv,
    "vistara_ai_report.csv",
    "text/csv",
    key='download-csv'
)

# --- Metric Explanations ---
with st.expander("ℹ️ Understanding the Metrics"):
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Metric A: Update Velocity**\n\nSpikes in update activity. ML detects anomalies.")
    c2.markdown("**Metric B: Divergence Ratio**\n\nDemo/Bio Updates. >2.0: Labor Migration.\n<0.5: School Drives.")
    c3.markdown("**Metric C: Migration Index**\n\nUpdates/Enrolments. >50: Phantom Growth (High Churn).")

# --- AI Narrative (V2.0) ---
if not df_filtered.empty:
    high_risk_count = len(df_filtered[df_filtered['risk_level'] == 'High'])
    top_district = df_filtered.loc[df_filtered['velocity'].idxmax()] if not df_filtered.empty else None
    
    insight_text = f"<strong>AI Insight:</strong> Analysis of {len(df_filtered)} records reveals <strong>{high_risk_count} high-risk anomalies</strong>."
    if top_district is not None:
        insight_text += f"<br><br><strong>Critical Alert:</strong> District <strong>{top_district['district']}</strong> ({top_district['state']}) is showing a velocity of <strong>{top_district['velocity']:.2f}</strong>, which is a significant deviation from normal patterns."
        if top_district['divergence_ratio'] > 2.0:
            insight_text += " This divergence suggests potential <strong>adult labor migration</strong> flux."
    
    st.markdown(f'<div class="ai-insight-box">{insight_text}</div>', unsafe_allow_html=True)


# --- Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["🗺️ The Pulse Map", "🚨 Anomaly Hunter (ML)", "📈 Strategic Insights"])

# --- TAB 1: PULSE MAP (V2.0 Mapbox) ---
with tab1:
    st.subheader("Geospatial Stress Analysis (Mapbox)")
    
    # State Level Aggregation for Map Reliability (Coords for centroids would be best, but we rely on names)
    # Using a workaround: We can't use Scatter Mapbox easily without Lat/Lon.
    # We will stick to Choropleth but use a better style if possible, OR if user wants Mapbox,
    # we really need Lat/Lon.
    # Hackathon context: We assume we might NOT have external lat/lon file.
    # However, Plotly's `choropleth_mapbox` works with GeoJSON.
    
    state_map_data = df_filtered.groupby('state')[['total_updates', 'velocity', 'anomaly_score']].mean().reset_index()
    
    # Fix for GeoJSON Matching: The GeoJSON usually expects Title Case (e.g. "Andhra Pradesh")
    # Our data is UPPERCASE.
    state_map_data['state_mapped'] = state_map_data['state'].str.title()
    
    # Custom Replacements for common mismatches in Indian GeoJSONs
    state_replacements = {
        "Andaman And Nicobar Islands": "Andaman & Nicobar Islands",
        "Jammu And Kashmir": "Jammu & Kashmir",
        "Dadra And Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu", # formatting varies
        "Delhi": "NCT of Delhi", # sometimes differs
        "Telangana": "Telangana" # usually matches
    }
    state_map_data['state_mapped'] = state_map_data['state_mapped'].replace(state_replacements)
    
    # Use choropleth_map instead of choropleth_mapbox to avoid deprecation warning
    # Note: mapbox_style is valid in choropleth_mapbox, for choropleth_map we might need layout configuration.
    # Actually, px.choropleth_mapbox is the intended API for mapbox tiles. 
    # The warning says "Use choropleth_map instead". This is a very recent Plotly change (v5.24+).
    # Let's switch to standard choropleth if mapbox is troublesome, or update to new syntax.
    # New syntax: px.choropleth_map(..., map_style="carto-positron")
    
    try:
        fig_map = px.choropleth_map(
            state_map_data,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='state_mapped', # Use mapped names
            color='velocity',
            color_continuous_scale='Reds',
            range_color=(0, 2), # Cap at 2 for visibility
            map_style="carto-positron", # Replaces mapbox_style
            zoom=3, center = {"lat": 20.5937, "lon": 78.9629},
            opacity=0.7,
            labels={'velocity':'Avg Velocity'},
            title="Average Update Velocity by State (ML Risk Overlay)"
        )
    except AttributeError:
        # Fallback for older plotly versions if installed
        fig_map = px.choropleth_mapbox(
            state_map_data,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='state_mapped',
            color='velocity',
            color_continuous_scale='Reds',
            range_color=(0, 2),
            mapbox_style="carto-positron",
            zoom=3, center = {"lat": 20.5937, "lon": 78.9629},
            opacity=0.7,
            labels={'velocity':'Avg Velocity'},
            title="Average Update Velocity by State (ML Risk Overlay)"
        )
        
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### Top Districts by Stress (ML Model)")
    # Fix Styler.applymap deprecation -> Styler.map
    st.dataframe(
        df_filtered.sort_values(by='anomaly_score', ascending=True).head(10)[
            ['state', 'district', 'date', 'velocity', 'divergence_ratio', 'risk_level']
        ].style.map(lambda v: 'color: red; font-weight: bold;' if v == 'High' else '', subset=['risk_level'])
    )


# --- TAB 2: ANOMALY HUNTER (ML) & FORECAST ---
with tab2:
    st.subheader("ML Anomaly Detection & Forecasting")
    
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.markdown("#### High Risk Districts (Isolation Forest)")
        anomalies = df_filtered[df_filtered['risk_level'] == 'High'].sort_values('anomaly_score')
        st.dataframe(anomalies[['date', 'state', 'district', 'velocity', 'risk_level']], use_container_width=True, height=400)
        
    with col_r:
        st.markdown("#### 🔮 3-Month Forecast & What-If Simulation")
        st.caption("Projecting 'Total Updates' for Top High Risk District")
        
        if not anomalies.empty:
            # Forecast for the #1 anomalous district
            target_district = anomalies.iloc[0]['district']
            st.write(f"**District:** {target_district}")
            
            # --- What-If Simulator (Enhancement 1) ---
            st.markdown("##### 🛠️ Intervention Simulator")
            new_centers = st.slider("Deploy New Aadhaar Centers", min_value=0, max_value=10, value=0, help="Simulate the impact of adding new centers on update velocity.")
            
            # Get full history
            forecast_y, p = generate_forecast(df_main, target_district, periods=3)
            
            if forecast_y is not None:
                # Apply " What-If" Logic:
                # Formula: Adjusted Forecast = Forecast * (1 - (New_Centers * Capacity_Factor))
                # Assumption: Each new center reduces backlog/stress velocity by 2% (0.02)
                capacity_factor = 0.02
                reduction_multiplier = 1.0 - (new_centers * capacity_factor)
                
                # Apply to forecast only
                adjusted_forecast_y = forecast_y * reduction_multiplier
                
                # Plot
                hist_data = df_main[df_main['district'] == target_district].sort_values('date')
                y_hist = hist_data['total_updates'].values
                x_hist = np.arange(len(y_hist))
                
                # Combine for plotting
                feature_x = np.arange(len(y_hist), len(y_hist)+3)
                
                # Create DF for plot
                # We plot History, Base Forecast, and Intervention Forecast
                df_plot = pd.DataFrame({
                    'Month_Index': np.concatenate([x_hist, feature_x, feature_x]),
                    'Updates': np.concatenate([y_hist, forecast_y, adjusted_forecast_y]),
                    'Type': ['History']*len(x_hist) + ['Base Forecast']*3 + ['With Intervention']*3
                })
                
                fig_forecast = px.line(df_plot, x='Month_Index', y='Updates', color='Type', markers=True, 
                                       title=f"Trend Analysis: {target_district}")
                
                # Style lines
                fig_forecast.update_traces(patch={"line": {"dash": "dot"}}, selector={"legendgroup": "Base Forecast"})
                fig_forecast.update_traces(patch={"line": {"dash": "dash", "color": "green"}}, selector={"legendgroup": "With Intervention"})
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Metrics
                base_proj = int(forecast_y[-1])
                adj_proj = int(adjusted_forecast_y[-1])
                st.metric("Projected Updates (Month +3)", f"{adj_proj}", delta=f"{adj_proj - base_proj} (Impact)", delta_color="inverse")
                
            else:
                st.warning("Not enough data to forecast.")
        else:
            st.write("Select or Find High Risk Districts to see forecast.")

    # --- Automated Policy Brief (Enhancement 2) ---
    st.markdown("---")
    st.subheader("📄 Automated Policy Brief")
    
    if not anomalies.empty:
        top_risk = anomalies.iloc[0]
        # Generate dynamic text
        brief_title = f"Priority Alert: {top_risk['district']} District ({top_risk['state']})"
        brief_body = f"""
        **Executive Summary:**
        The district of **{top_risk['district']}** has been flagged with a Critical Anomaly Score of **{top_risk.get('anomaly_score', 'N/A'):.2f}**.
        
        **Observations:**
        - Current Velocity: **{top_risk['velocity']:.2f}** (High Spikes Detected)
        - Divergence Ratio: **{top_risk.get('divergence_ratio', 0):.2f}**
        
        **Projections:**
        Based on current trends, we anticipate a continued surge in biometric updates over the next quarter.
        
        **Recommendations:**
        1. **Immediate Action**: Deploy **3 Mobile Update Kits** to {top_risk['district']} High School zones.
        2. **Resource Allocation**: Divert redundant staff from neighboring low-stress districts.
        3. **Monitoring**: Enable daily velocity tracking for this region.
        
        *Generated by Vistara-AI Decision Engine*
        """
        
        # Display Box
        st.info(brief_body)
        
        # Download Button for Brief
        brief_text = f"{brief_title}\n\n{brief_body.replace('**', '').replace('*', '')}"
        st.download_button("📥 Download Policy Brief (TXT)", brief_text, f"Policy_Brief_{top_risk['district']}.txt")
    else:
        st.success("No critical alerts requiring policy intervention at this time.")


# --- TAB 3: STRATEGIC INSIGHTS ---
with tab3:
    st.subheader("Growth vs. Churn Analysis (With Risk Overlay)")
    
    agg_scatter = df_filtered.groupby(['state', 'district']).agg({
        'total_enrolments': 'sum', 
        'total_demo_updates': 'sum',
        'risk_level': lambda x: x.mode()[0] if not x.mode().empty else 'Low'
    }).reset_index()
    
    agg_scatter['total_enrolments'] += 1
    
    fig_scatter = px.scatter(
        agg_scatter,
        x="total_enrolments",
        y="total_demo_updates",
        color="risk_level", # Use ML Risk Level for color
        hover_name="district",
        log_x=True,
        log_y=True,
        title="Strategic Insights Grid (Risk-Based)",
        color_discrete_map={
            "High": "red",
            "Medium": "orange",
            "Low": "green"
        },
        size='total_demo_updates', # Bubble size
        size_max=20
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
