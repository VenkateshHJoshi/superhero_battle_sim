import pandas as pd
import numpy as np
import os
import joblib
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_assets():
    """
    Loads the trained model, scaler, processed dataset, and the DYNAMIC feature list.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    model_path = os.path.join(project_root, 'models', 'best_battle_model.pkl')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
    data_path = os.path.join(project_root, 'data', 'processed', 'superheroes_processed.csv')
    feature_list_path = os.path.join(project_root, 'models', 'feature_list.pkl')
    
    # Load assets
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
    
    # Load the feature list generated during NLP Feature Engineering
    # This ensures we use the EXACT columns the model was trained on
    try:
        feature_cols = joblib.load(feature_list_path)
    except FileNotFoundError:
        print("⚠️ Warning: feature_list.pkl not found. Using fallback (This might cause errors).")
        feature_cols = [] 

    return model, scaler, df, feature_cols

def simulate_battle(hero_a_name, hero_b_name):
    """
    Takes two hero names, runs the ML prediction.
    """
    model, scaler, df, feature_cols = load_assets()
    
    # Check if feature list was loaded correctly
    if not feature_cols:
        return "Error: Feature list missing. Please re-run feature engineering.", None, None
    
    # Check existence
    if hero_a_name not in df['name'].values:
        return f"Error: Hero '{hero_a_name}' not found.", None, None
    if hero_b_name not in df['name'].values:
        return f"Error: Hero '{hero_b_name}' not found.", None, None
        
    # Get hero data (Already Scaled)
    # We use the dynamic feature_cols list here
    hero_a = df[df['name'] == hero_a_name][feature_cols].iloc[0]
    hero_b = df[df['name'] == hero_b_name][feature_cols].iloc[0]
    
    # Calculate Difference (A - B)
    diff = hero_a.values - hero_b.values
    diff = diff.reshape(1, -1)
    
    # Predict
    proba = model.predict_proba(diff)[0]
    prob_a_wins = proba[1]
    prob_b_wins = proba[0]
    
    winner = hero_a_name if prob_a_wins > 0.5 else hero_b_name
    
    result = {
        "winner": winner,
        "hero_a": hero_a_name,
        "hero_b": hero_b_name,
        "prob_a": prob_a_wins,
        "prob_b": prob_b_wins
    }
    
    return result, hero_a, hero_b

if __name__ == "__main__":
    print("🧪 Testing Game Logic (Dynamic Features)...")
    model, scaler, df, feature_cols = load_assets()
    
    h1 = df.iloc[0]['name']
    h2 = df.iloc[1]['name']
    
    print(f"Simulating battle: {h1} vs {h2}")
    res, _, _ = simulate_battle(h1, h2)
    
    if isinstance(res, dict):
        print(f"🏆 Winner: {res['winner']}")
        print(f"📊 Probability {res['hero_a']} wins: {res['prob_a']:.2%}")
        print(f"📊 Probability {res['hero_b']} wins: {res['prob_b']:.2%}")
    else:
        print(res)