import pandas as pd
import numpy as np

def calculate_metrics(df):
    """
    Calculates Core Metrics:
    - Metric A: Update Velocity (V)
    - Metric B: Divergence Ratio (D)
    - Metric C: Migration Index (M)
    
    And performs Anomaly Detection.
    """
    if df.empty:
        return df

    # Ensure data is sorted for lag calculations
    df = df.sort_values(by=['state', 'district', 'date'])

    # Total Updates = Bio Updates + Demo Updates
    # Note: Column names from data_processor aggregation:
    # bio: bio_age_5_17, bio_age_17_
    # demo: demo_age_5_17, demo_age_17_
    # enrol: age_0_5, age_5_17, age_18_greater
    
    df['total_bio_updates'] = df['bio_age_5_17'] + df['bio_age_17_']
    df['total_demo_updates'] = df['demo_age_5_17'] + df['demo_age_17_']
    df['total_updates'] = df['total_bio_updates'] + df['total_demo_updates']
    
    df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']

    # --- Metric A: Update Velocity (V) ---
    # Formula: (Current - Previous) / Previous
    # usage groupby to shift within group
    # We will use 'total_updates' for velocity as "activity"
    
    df['prev_month_updates'] = df.groupby(['state', 'district'])['total_updates'].shift(1)
    
    # Fill NaN for first month with 0 (or handle gracefully - setting velocity to 0)
    df['prev_month_updates'] = df['prev_month_updates'].fillna(0)
    
    # Calculate Velocity. If prev is 0, we can define velocity as 0 or 100% depending on logic.
    # Standard approach: (Curr - Prev) / Prev. If Prev is 0 and Curr > 0, velocity is infinite.
    # We will mitigate by adding 1 or handling 0 case.
    # Handling div by zero: replace 0 with 1 in denominator for stability 
    # OR specific logic: if prev=0, velocity = 0 (start of tracking)
    
    # Let's use a safe division approach
    df['velocity'] = np.where(
        df['prev_month_updates'] > 0,
        (df['total_updates'] - df['prev_month_updates']) / df['prev_month_updates'],
        0.0 # If prev is 0, we treat velocity as 0 to avoid spikes on month 1
    )

    # --- Metric B: Divergence Ratio (D) ---
    # Formula: Total_Demographic_Updates / (Total_Biometric_Updates + 1)
    df['divergence_ratio'] = df['total_demo_updates'] / (df['total_bio_updates'] + 1)

    # --- Metric C: Migration Index (M) ---
    # Formula: Total_Demographic_Updates / (Total_New_Enrolments + 1)
    df['migration_index'] = df['total_demo_updates'] / (df['total_enrolments'] + 1)

    # --- Anomaly Detection (Machine Learning) ---
    from sklearn.ensemble import IsolationForest

    # Prepare features for ML
    # Handle NaNs and infinite values before fitting
    model_data = df[['velocity', 'divergence_ratio', 'migration_index']].fillna(0)
    model_data = model_data.replace([np.inf, -np.inf], 0)
    
    # Initialize Isolation Forest
    # contamination='auto' lets the model decide threshold, or we can set it e.g. 0.05
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    
    # Fit and Predict
    df['anomaly_label'] = iso_forest.fit_predict(model_data)
    # anomaly_score: offset_ is negative for anomalies, positive for inliers
    # We invert it so higher score = more anomalous for intuitive understanding? 
    # Actually, standard score_samples returns negative values. 
    # Let's use decision_function: lower is more anomalous. 
    # For UI, let's normalize or simply use the label.
    # The prompt asks for 'anomaly_score'.
    df['anomaly_score'] = iso_forest.decision_function(model_data)
    
    # Create Risk Levels based on score
    # Lower score = higher risk
    # We can categorize based on quantiles or label
    # Label: -1 is anomaly (High Risk), 1 is normal (Low Risk)
    # Let's refine:
    # High Risk: label == -1
    # Medium Risk: label == 1 but score is low (close to 0)
    # Low Risk: label == 1 and score is high
    
    def classify_risk(row):
        if row['anomaly_label'] == -1:
            return 'High'
        elif row['anomaly_score'] < 0.1: # Threshold tunable
            return 'Medium'
        else:
            return 'Low'
            
    df['risk_level'] = df.apply(classify_risk, axis=1)

    # Forecasting (Time Series)
    # We need to forecast 'total_updates' for next 3 months for top 5 high risk districts regarding velocity?
    # The request: "Select the top 5 'High Stress' districts."
    # We will provide a function to do this rather than row-level calc here, 
    # or we can pre-calculate forecasts for ALL districts? 
    # Doing it for all is expensive. Let's do it on the fly or just return this ready for app.
    # But prompt says "Upgrade the Brain... Upgrade the Vision".
    # Let's add a helper function for forecasting here.
    
    return df

def generate_forecast(df, district_name, periods=3):
    """
    Simple linear forecast using np.polyfit for a specific district.
    """
    district_df = df[df['district'] == district_name].sort_values('date')
    if len(district_df) < 2:
        return None
        
    y = district_df['total_updates'].values
    x = np.arange(len(y))
    
    # fit linear model
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # Predict next 3 months
    future_x = np.arange(len(y), len(y) + periods)
    forecast_y = p(future_x)
    
    return forecast_y, p

if __name__ == "__main__":
    from data_processor import load_and_merge_data
    print("Loading data...")
    df = load_and_merge_data()
    print("Calculating metrics & ML...")
    df_analyzed = calculate_metrics(df)
    
    print("Analysis Complete.")
    print("Top High Risk Districts:")
    print(df_analyzed[df_analyzed['risk_level'] == 'High'].head()[['district', 'velocity', 'anomaly_score', 'risk_level']])
