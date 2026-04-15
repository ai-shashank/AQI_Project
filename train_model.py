import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'pollution_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'aqi_model.pkl')

def train_optimized():
    if not os.path.exists(DATA_PATH):
        print("CSV file not found!")
        return

    # Load Data
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    # Column Mapping (verified for your Kaggle dataset)
    feature_map = {
        'temp': 'temp_2m_c',
        'hum': 'humidity_percent',
        'pm25': 'pm2_5_ugm3',
        'no2': 'no2_ugm3',
        'target': 'us_aqi'
    }

    # Clean missing values
    df_clean = df[list(feature_map.values())].dropna()

    X = df_clean[[feature_map['temp'], feature_map['hum'], feature_map['pm25'], feature_map['no2']]]
    y = df_clean[feature_map['target']]

    print(f"Training on {len(df_clean)} rows...")

    # OPTIMIZED MODEL: Limits depth to save memory
    model = RandomForestRegressor(
        n_estimators=50,      # Fewer trees
        max_depth=8,         # Shorter trees = smaller file size
        max_samples=0.1,      # Use only 10% of data for each tree
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)

    # Save
    joblib.dump(model, MODEL_PATH, compress=3) # High compression
    print(f"SUCCESS: Small, fast model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_optimized()