import pandas as pd
import numpy as np
import os
import joblib # Standard library for saving python objects (models/scalers)
from sklearn.preprocessing import RobustScaler, LabelEncoder
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion import load_data

def extract_battle_experience(text):
    """NLP: Count action words in history text."""
    if pd.isna(text) or text == "Unknown":
        return 0
    keywords = ['fight', 'battle', 'war', 'save', 'protect', 'kill', 'defeat', 
                'trained', 'master', 'army', 'lead', 'win', 'lose', 'enemy']
    text = text.lower()
    score = sum(text.count(word) for word in keywords)
    return score

def engineer_features(df):
    """
    Robust Feature Engineering Pipeline.
    Handles Outliers (RobustScaler), Nulls (Median), and Duplicates.
    """
    print("⚙️ Starting Hardened Feature Engineering...")
    
    # 1. Remove Duplicates (keep first occurrence)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['name'], keep='first')
    print(f"🧹 Removed {initial_count - len(df)} duplicate heroes.")

    # 2. Basic Cleaning
    df = df.dropna(subset=['name']).reset_index(drop=True)
    df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
    
    # 3. Handle Missing Values
    # Text: "Unknown"
    text_cols = ['history_text', 'powers_text']
    for col in text_cols:
        df[col] = df[col].fillna("Unknown")
        
    # Numeric: Median (Robust to outliers)
    numeric_cols = ['intelligence_score', 'strength_score', 'speed_score', 
                    'durability_score', 'power_score', 'combat_score']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 4. NLP Feature: Battle Experience
    print("📝 Extracting 'Battle Experience' from history text...")
    df['battle_experience_score'] = df['history_text'].apply(extract_battle_experience)

    # 5. Feature: Power Diversity
    power_cols = [col for col in df.columns if col.startswith('has_')]
    df[power_cols] = df[power_cols].fillna(0)
    df['power_diversity_score'] = df[power_cols].sum(axis=1)

    # 6. Encode Alignment
    print("🏷️ Encoding Alignment...")
    le = LabelEncoder()
    df['alignment'] = df['alignment'].fillna('Unknown')
    df['alignment_encoded'] = le.fit_transform(df['alignment'])

    # 7. Select Features for ML
    feature_cols = [
        'intelligence_score', 'strength_score', 'speed_score', 
        'durability_score', 'power_score', 'combat_score',
        'battle_experience_score', 'power_diversity_score', 'alignment_encoded'
    ]
    
    X = df[feature_cols].copy()
    
    # 8. Scaling: Using RobustScaler (Handles Outliers better than StandardScaler)
    print("📐 Scaling features using RobustScaler (Outlier-resistant)...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Combine with Name and original stats for reference
    final_df = pd.concat([df[['name', 'overall_score']], X_scaled_df], axis=1)
    
    print("✅ Feature Engineering Complete!")
    return final_df, scaler

def save_processed_data(df, filename='superheroes_processed.csv'):
    """Saves processed dataframe."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'data', 'processed', filename)
    df.to_csv(save_path, index=False)
    print(f"💾 Saved processed data to: {save_path}")

def save_scaler(scaler, filename='scaler.pkl'):
    """Saves the fitted scaler to ensure consistency during prediction."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'models', filename)
    
    joblib.dump(scaler, save_path)
    print(f"🔧 Saved Scaler object to: {save_path}")

if __name__ == "__main__":
    # Load
    raw_df = load_data()
    
    # Process
    processed_df, scaler_object = engineer_features(raw_df)
    
    # Save Data
    save_processed_data(processed_df)
    
    # Save Scaler (Crucial for Step 6)
    save_scaler(scaler_object)
    
    # Display sample
    print("\n--- Sample of Robustly Scaled Data ---")
    print(processed_df.head())