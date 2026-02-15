import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

# Add parent directory to path to import data_ingestion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion import load_data

def extract_battle_experience(text):
    """
    NLP: Count specific 'action' words in history text to gauge experience.
    """
    if pd.isna(text) or text == "Unknown":
        return 0
    
    # Define keywords associated with combat and experience
    keywords = ['fight', 'battle', 'war', 'save', 'protect', 'kill', 'defeat', 
                'trained', 'master', 'army', 'lead', 'win', 'lose', 'enemy']
    
    text = text.lower()
    # Count occurrences
    score = sum(text.count(word) for word in keywords)
    return score

def engineer_features(df):
    """
    Cleans data, creates new features, and scales numerical values.
    """
    print("⚙️ Starting Feature Engineering...")
    
    # 1. Basic Cleaning (Drop rows with missing names, convert overall_score)
    df = df.dropna(subset=['name']).reset_index(drop=True)
    df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
    
    # 2. Handle Missing Values (Fill text with "Unknown", fill numeric scores with median)
    text_cols = ['history_text', 'powers_text']
    for col in text_cols:
        df[col] = df[col].fillna("Unknown")
        
    numeric_cols = ['intelligence_score', 'strength_score', 'speed_score', 
                    'durability_score', 'power_score', 'combat_score']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 3. NLP Feature: Battle Experience from History Text
    print("📝 Extracting 'Battle Experience' from history text...")
    df['battle_experience_score'] = df['history_text'].apply(extract_battle_experience)

    # 4. Feature: Power Diversity (Sum of all has_... columns)
    # Identify columns starting with 'has_'
    power_cols = [col for col in df.columns if col.startswith('has_')]
    # Fill NaNs in power columns with 0 (assuming NaN means False/No power)
    df[power_cols] = df[power_cols].fillna(0)
    df['power_diversity_score'] = df[power_cols].sum(axis=1)

    # 5. Encode Alignment (Good/Bad/Neutral -> Numbers)
    print("🏷️ Encoding Alignment...")
    le = LabelEncoder()
    # Fill missing alignment with 'Unknown' so it handles uniformly
    df['alignment'] = df['alignment'].fillna('Unknown')
    df['alignment_encoded'] = le.fit_transform(df['alignment'])

    # 6. Select Features for ML
    feature_cols = [
        'intelligence_score', 'strength_score', 'speed_score', 
        'durability_score', 'power_score', 'combat_score',
        'battle_experience_score', 'power_diversity_score', 'alignment_encoded'
    ]
    
    # Create a dataframe for features
    X = df[feature_cols].copy()
    
    # 7. Scaling (Standardization)
    # Important so that Strength (0-100) and Experience (0-50) are on the same scale
    print("📐 Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame for readability
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Combine with Name and original stats for reference
    final_df = pd.concat([df[['name', 'overall_score']], X_scaled_df], axis=1)
    
    print("✅ Feature Engineering Complete!")
    return final_df, scaler

def save_processed_data(df, filename='superheroes_processed.csv'):
    """
    Saves the processed dataframe to the processed folder.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'data', 'processed', filename)
    
    df.to_csv(save_path, index=False)
    print(f"💾 Saved processed data to: {save_path}")

if __name__ == "__main__":
    # Load
    raw_df = load_data()
    
    # Process
    processed_df, scaler_object = engineer_features(raw_df)
    
    # Save
    save_processed_data(processed_df)
    
    # Display sample
    print("\n--- Sample of Processed Data ---")
    print(processed_df.head())