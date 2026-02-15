import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion import load_data

def clean_text(text):
    """Basic text cleaning for NLP."""
    if pd.isna(text) or text == "Unknown":
        return ""
    # Convert to lowercase and remove non-alphanumeric chars
    text = str(text).lower()
    return text

def engineer_features(df):
    """
    Advanced Feature Engineering using TF-IDF NLP.
    """
    print("⚙️ Starting Advanced NLP Feature Engineering...")
    
    # 1. Basic Cleaning
    df = df.drop_duplicates(subset=['name'], keep='first').reset_index(drop=True)
    df = df.dropna(subset=['name']).reset_index(drop=True)
    df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
    
    # Fill missing text
    df['history_text'] = df['history_text'].fillna("Unknown").apply(clean_text)
    df['powers_text'] = df['powers_text'].fillna("Unknown").apply(clean_text)
    
    # Combine history and powers for richer context
    df['combined_text'] = df['history_text'] + " " + df['powers_text']

    # 2. NLP: TF-IDF Vectorization (No Hardcoded Words)
    print("📝 Extracting NLP features using TF-IDF (Top 30 keywords)...")
    vectorizer = TfidfVectorizer(max_features=30, stop_words='english')
    
    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    
    # Create a DataFrame for the TF-IDF features
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"nlp_{word}" for word in vectorizer.get_feature_names_out()])
    
    # Reset index to concatenate safely
    df = df.reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)
    
    # Merge NLP features with original stats
    df = pd.concat([df, tfidf_df], axis=1)

    # 3. Handle Missing Numeric Values
    numeric_cols = ['intelligence_score', 'strength_score', 'speed_score', 
                    'durability_score', 'power_score', 'combat_score']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 4. Encode Alignment
    le = LabelEncoder()
    df['alignment'] = df['alignment'].fillna('Unknown')
    df['alignment_encoded'] = le.fit_transform(df['alignment'])

    # 5. Power Diversity
    power_cols = [col for col in df.columns if col.startswith('has_')]
    df[power_cols] = df[power_cols].fillna(0)
    df['power_diversity_score'] = df[power_cols].sum(axis=1)
    
    # 6. Battle Experience (Length of history text as proxy for lore depth)
    df['lore_depth_score'] = df['history_text'].apply(len)

    # 7. Select Features for ML (Stats + NLP + Encoded)
    # We exclude text columns and non-numeric data for the scaler
    feature_cols = numeric_cols + [
        'power_diversity_score', 'alignment_encoded', 'lore_depth_score'
    ] + list(tfidf_df.columns) # Add all NLP columns dynamically
    
    X = df[feature_cols].copy()
    
    # 8. Scaling
    print("📐 Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Combine with Name and overall_score
    final_df = pd.concat([df[['name', 'overall_score']], X_scaled_df], axis=1)
    
    print("✅ NLP Feature Engineering Complete!")
    return final_df, scaler, vectorizer, feature_cols

def save_processed_data(df, filename='superheroes_processed.csv'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'data', 'processed', filename)
    df.to_csv(save_path, index=False)
    print(f"💾 Saved processed data to: {save_path}")

def save_scaler(scaler, filename='scaler.pkl'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'models', filename)
    joblib.dump(scaler, save_path)
    print(f"🔧 Saved Scaler.")

def save_vectorizer(vectorizer, filename='tfidf_vectorizer.pkl'):
    """Save the NLP vectorizer so we can process new text exactly the same way."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'models', filename)
    joblib.dump(vectorizer, save_path)
    print(f"🧠 Saved TF-IDF Vectorizer.")

def save_feature_list(feature_cols, filename='feature_list.pkl'):
    """Save the list of features so model knows what columns to expect."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'models', filename)
    joblib.dump(feature_cols, save_path)
    print(f"📝 Saved Feature List.")

if __name__ == "__main__":
    raw_df = load_data()
    processed_df, scaler_obj, vectorizer_obj, feat_cols = engineer_features(raw_df)
    
    save_processed_data(processed_df)
    save_scaler(scaler_obj)
    save_vectorizer(vectorizer_obj)
    save_feature_list(feat_cols)
    
    print("\n--- Sample Data with NLP Features ---")
    print(processed_df[['name'] + [c for c in processed_df.columns if 'nlp_' in c][:5]].head())