import pandas as pd
import numpy as np
import os
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_battle_dataset(df, num_samples, feature_cols):
    """
    Generates battles using POWER TIER as the primary deciding factor.
    This forces the model to learn: Cosmic (Tier 2) > Tech (Tier 1) > Human (Tier 0).
    """
    print(f"⚔️ Generating Battle Dataset (Strict Power Tier Mode)...")
    
    battles_X = []
    battles_y = []
    
    for _ in range(num_samples):
        hero_a = df.sample(n=1).iloc[0]
        hero_b = df.sample(n=1).iloc[0]
        
        stats_diff = hero_a[feature_cols] - hero_b[feature_cols]
        
        # --- STRICT TRAINING LOGIC ---
        # 1. Check Power Tier (Cosmic > Tech > Human)
        if hero_a['power_tier'] > hero_b['power_tier']:
            winner = 1 # A wins
        elif hero_a['power_tier'] < hero_b['power_tier']:
            winner = 0 # B wins
        else:
            # 2. If same tier, check Overall Score
            if hero_a['overall_score'] > hero_b['overall_score']:
                winner = 1
            elif hero_a['overall_score'] < hero_b['overall_score']:
                winner = 0
            else:
                winner = np.random.choice([0, 1])
        
        battles_X.append(stats_diff.values)
        battles_y.append(winner)
        
    return np.array(battles_X), np.array(battles_y)

def evaluate_models(X_train, X_test, y_train, y_test):
    print("🤖 Training and Evaluating Models...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = {}
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        results[name] = {"model": model, "accuracy": acc, "roc_auc": roc}
    return results

def run_training_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'processed', 'superheroes_processed.csv')
    
    df = pd.read_csv(data_path)
    
    # 2. K-Means Clustering for Power Tiers
    print("📊 Calculating Power Tiers...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    stat_cols = ['intelligence_score', 'strength_score', 'speed_score', 'durability_score', 'power_score', 'combat_score']
    df['power_tier'] = kmeans.fit_predict(df[stat_cols])
    
    # Label the tiers for our understanding (Optional, but good for sanity check)
    # Map cluster numbers to 0=Human, 1=Enhanced, 2=Cosmic based on mean stats
    tier_means = df.groupby('power_tier')[stat_cols].mean().sum(axis=1).sort_values()
    mapping = {tier_means.index[0]: 0, tier_means.index[1]: 1, tier_means.index[2]: 2}
    df['power_tier'] = df['power_tier'].map(mapping)
    
    df.to_csv(data_path, index=False)
    print(f"💾 Saved Tiers: 0=Street Level, 1=Enhanced/Tech, 2=Cosmic/God-like")
    
    feature_cols = [
        'intelligence_score', 'strength_score', 'speed_score', 
        'durability_score', 'power_score', 'combat_score',
        'battle_experience_score', 'power_diversity_score', 'alignment_encoded',
        'power_tier' 
    ]
    
    print(f"🛠️ Strict Mode: Training with {len(feature_cols)} features.")
    
    X, y = generate_battle_dataset(df, num_samples=10000, feature_cols=feature_cols)
    print(f"📊 Generated battle data shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*30)
    print("📋 MODEL COMPARISON")
    print("="*30)
    best_model_name = None
    best_score = -1
    for name, metrics in results.items():
        print(f"{name:<20} | Acc: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f}")
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model_name = name
            
    print(f"🏆 BEST MODEL: {best_model_name}")
    
    model_path = os.path.join(project_root, 'models', 'best_battle_model.pkl')
    joblib.dump(results[best_model_name]['model'], model_path) 
    print(f"💾 Model saved.")
    
    return best_model_name

if __name__ == "__main__":
    run_training_pipeline()