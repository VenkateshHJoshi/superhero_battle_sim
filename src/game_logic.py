import pandas as pd
import numpy as np
import os
import joblib
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_assets():
    """
    Loads the trained model, scaler, and processed dataset.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    model_path = os.path.join(project_root, 'models', 'best_battle_model.pkl')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
    data_path = os.path.join(project_root, 'data', 'processed', 'superheroes_processed.csv')
    
    # Load
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
    
    return model, scaler, df

def simulate_battle(hero_a_name, hero_b_name):
    """
    Takes two hero names, runs the ML prediction.
    """
    model, scaler, df = load_assets()
    
    # IMPORTANT: This list must be IDENTICAL to the one in model_training.py
    feature_cols = [
        'intelligence_score', 'strength_score', 'speed_score', 
        'durability_score', 'power_score', 'combat_score',
        'battle_experience_score', 'power_diversity_score', 'alignment_encoded',
        'power_tier' 
    ]
    
    # Check existence
    if hero_a_name not in df['name'].values:
        return f"Error: Hero '{hero_a_name}' not found.", None, None
    if hero_b_name not in df['name'].values:
        return f"Error: Hero '{hero_b_name}' not found.", None, None
        
    # Get hero data (Already Scaled)
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
    print("🧪 Testing Game Logic...")
    _, _, df = load_assets()
    
    # Pick first two heroes
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