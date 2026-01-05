import pandas as pd
import glob
import os

def clean_state_name(state):
    """
    Standardizes state names to handle typos and variations.
    """
    if pd.isna(state):
        return state
        
    state = str(state).strip().upper()
    
    # Common mappings
    mapping = {
        "WEST BANGAL": "WEST BENGAL",
        "WEST BENGLI": "WEST BENGAL",
        "WESTBENGAL": "WEST BENGAL",
        "WEST  BENGAL": "WEST BENGAL",
        "WB": "WEST BENGAL",
        
        "TAMILNADU": "TAMIL NADU",
        "ORISSA": "ODISHA",
        "PONDICHERRY": "PUDUCHERRY",
        "UTTARANCHAL": "UTTARAKHAND",
        "CHHATISGARH": "CHHATTISGARH",
        
        "ANDAMAN & NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
        "A & N ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
        
        "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
        "J & K": "JAMMU AND KASHMIR",
        
        "DADRA & NAGAR HAVELI": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        "DADRA AND NAGAR HAVELI": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        "DAMAN & DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        "DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
        
        "LAKSHADWEEP": "LAKSHADWEEP", 
    }
    
    return mapping.get(state, state)

    return mapping.get(state, state)

def load_and_merge_data(base_path="datasets"):
    """
    Loads, cleans, and merges Aadhar datasets from the specified base path.
    """
    
    # helper to load all csvs in a directory
    def load_folder_csvs(folder_name):
        path = os.path.join(base_path, folder_name, "*.csv")
        files = glob.glob(path)
        dfs = []
        for f in files:
            try:
                # Read all columns as string first to handle potential mixed types, then convert
                df = pd.read_csv(f, dtype={'pincode': str}) 
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    # 1. Load Data
    print("Loading Biometric Data...")
    bio_df = load_folder_csvs("api_data_aadhar_biometric")
    
    print("Loading Demographic Data...")
    demo_df = load_folder_csvs("api_data_aadhar_demographic")
    
    print("Loading Enrolment Data...")
    enrol_df = load_folder_csvs("api_data_aadhar_enrolment")

    # 2. Data Cleaning & Standardization
    def process_dataset(df):
        if df.empty:
            return df
            
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Standardize Strings
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.strip().str.upper()
        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.strip().str.upper()
            
        # Robust filtering: Remove rows where State looks like a number
        if 'state' in df.columns:
             df = df[~df['state'].str.match(r'^\d+$')]
             
             # Apply State Name Cleaning
             df['state'] = df['state'].apply(clean_state_name)
             
        return df

    # Apply processing explicitly to update the variables
    bio_df = process_dataset(bio_df)
    demo_df = process_dataset(demo_df)
    enrol_df = process_dataset(enrol_df)

    # 3. Aggregation by District and Month-Year
    # We want to aggregate metrics at the District-Month level before merging
    
    # Biometric Aggregation
    if not bio_df.empty:
        bio_agg = bio_df.groupby(['state', 'district', 'date'])[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()
    else:
        bio_agg = pd.DataFrame(columns=['state', 'district', 'date', 'bio_age_5_17', 'bio_age_17_'])

    # Demographic Aggregation
    if not demo_df.empty:
        demo_agg = demo_df.groupby(['state', 'district', 'date'])[['demo_age_5_17', 'demo_age_17_']].sum().reset_index()
    else:
        demo_agg = pd.DataFrame(columns=['state', 'district', 'date', 'demo_age_5_17', 'demo_age_17_'])

    # Enrolment Aggregation
    if not enrol_df.empty:
        enrol_agg = enrol_df.groupby(['state', 'district', 'date'])[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
    else:
        enrol_agg = pd.DataFrame(columns=['state', 'district', 'date', 'age_0_5', 'age_5_17', 'age_18_greater'])

    # 4. Merging
    print("Merging Data...")
    merged_df = pd.merge(bio_agg, demo_agg, on=['state', 'district', 'date'], how='outer')
    merged_df = pd.merge(merged_df, enrol_agg, on=['state', 'district', 'date'], how='outer')

    # 5. Handle Missing Values
    merged_df = merged_df.fillna(0)
    
    # Sort by date
    merged_df = merged_df.sort_values(by=['state', 'district', 'date'])
    
    print(f"Data Pipeline Complete. Shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    df = load_and_merge_data()
    print(df.head())
    print(df.info())
