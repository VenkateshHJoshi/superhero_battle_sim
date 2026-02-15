import pandas as pd
import numpy as np
import os
import joblib
import sys
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def adjust_tiers_with_keywords(row):
    """
    STRICT Priority Logic:
    1. Tech/Suit users are CAPPED at Tier 1 (Enhanced).
    2. Only non-tech characters with Cosmic keywords become Tier 2.
    """
    text = str(row['combined_text']).lower()
    current_tier = row['power_tier']
    
    # --- PRIORITY 1: CHECK TECH ---
    # If they use a suit/tech/armor, they are NOT Cosmic entities.
    # They are Enhanced (Tier 1) at best, regardless of who they fight.
    tech_words = ['armor', 'suit', 'tech', 'robot', 'cyborg', 'mech', 'iron man', 'war machine', 'batsuit']
    if any(word in text for word in tech_words):
        return 1  # Force Tier 1 (Enhanced)

    # --- PRIORITY 2: CHECK COSMIC ---
    # Only applies if they are NOT Tech users.
    cosmic_words = ['cosmic', 'galaxy', 'galactus', 'eternal', 'universe', 'reality', 'deity', 'abstract', 'divine', 'surfer', 'herald']
    if any(word in text for word in cosmic_words):
        return 2  # Force Tier 2 (Cosmic)

    # Default to whatever K-Means said (Human or Enhanced)
    return current_tier

def generate_battle_dataset(df, num_samples, feature_cols):
    """Generate battles with STRICT Tier Enforcement."""
    print(f"⚔️ Generating Battle Dataset (Strict Enforcement)...")
    battles_X = []
    battles_y = []
    
    for _ in range(num_samples):
        hero_a = df.sample(n=1).iloc[0]
        hero_b = df.sample(n=1).iloc[0]
        
        if 'power_tier' not in hero_a.index or 'power_tier' not in hero_b.index:
            continue

        stats_diff = hero_a[feature_cols] - hero_b[feature_cols]
        
        # --- STRICT HIERARCHY ---
        if hero_a['power_tier'] > hero_b['power_tier']:
            winner = 1
        elif hero_a['power_tier'] < hero_b['power_tier']:
            winner = 0
        else:
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
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
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
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'superheroes_nlp_dataset.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'superheroes_processed.csv')
    feature_list_path = os.path.join(project_root, 'models', 'feature_list.pkl')
    
    # ---------------------------------------------------------
    # STEP 1: Load RAW Data & Calculate Tiers
    # ---------------------------------------------------------
    print("📊 Loading RAW data and applying STRICT Priority Logic...")
    raw_df = pd.read_csv(raw_data_path)
    
    # Combine text
    raw_df['history_text'] = raw_df['history_text'].fillna("")
    raw_df['powers_text'] = raw_df['powers_text'].fillna("")
    raw_df['combined_text'] = raw_df['history_text'] + " " + raw_df['powers_text']
    
    # Clean stats
    core_stats = ['intelligence_score', 'strength_score', 'speed_score', 'durability_score', 'power_score', 'combat_score']
    for col in core_stats:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(raw_df[col].median())
    
    # 1A. K-Means Baseline
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    raw_df['cluster'] = kmeans.fit_predict(raw_df[core_stats])
    
    # 1B. Map Cluster to Tier
    cluster_means = raw_df.groupby('cluster')[core_stats].mean().sum(axis=1).sort_values()
    tier_mapping = {
        cluster_means.index[0]: 0,
        cluster_means.index[1]: 1,
        cluster_means.index[2]: 2
    }
    raw_df['power_tier'] = raw_df['cluster'].map(tier_mapping)
    
    # 1C. APPLY STRICT PRIORITY LOGIC
    print("🔍 Applying Strict Priority (Tech > Context)...")
    raw_df['power_tier'] = raw_df.apply(adjust_tiers_with_keywords, axis=1)
    
    # Print Verification
    print("📝 FINAL VERIFICATION CHECK:")
    for hero in ['Silver Surfer', 'Iron Man', 'Superman', 'Batman', 'Thor']:
        if hero in raw_df['name'].values:
            h_data = raw_df[raw_df['name'] == hero].iloc[0]
            print(f"   {hero}: Tier {h_data['power_tier']}")

    # ---------------------------------------------------------
    # STEP 2: Merge Tiers
    # ---------------------------------------------------------
    print("🔗 Merging Tiers...")
    processed_df = pd.read_csv(processed_data_path)
    tier_map = raw_df.drop_duplicates(subset=['name']).set_index('name')['power_tier'].to_dict()
    processed_df['power_tier'] = processed_df['name'].map(tier_map).fillna(0).astype(int)
    
    # ---------------------------------------------------------
    # STEP 3: Update Features
    # ---------------------------------------------------------
    feature_cols = joblib.load(feature_list_path)
    if 'power_tier' not in feature_cols:
        feature_cols.append('power_tier')
        joblib.dump(feature_cols, feature_list_path)
        
    processed_df.to_csv(processed_data_path, index=False)
    
    # ---------------------------------------------------------
    # STEP 4: Train
    # ---------------------------------------------------------
    X, y = generate_battle_dataset(processed_df, num_samples=10000, feature_cols=feature_cols)
    print(f"📊 Data Shape: {X.shape}")
    
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