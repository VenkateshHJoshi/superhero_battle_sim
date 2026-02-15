import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.game_logic import load_assets, simulate_battle
from src.utils import create_custom_hero_vector

# --- Page Setup ---
st.set_page_config(page_title="Superhero Battle Simulator", page_icon="⚡", layout="centered")

st.markdown("""
<style>
    .big-font { font-size:45px !important; font-weight: bold; color: #FF4B4B; text-align: center; }
    .vs-font { font-size:30px !important; font-weight: bold; color: #AAAAAA; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Superhero Battle Simulator")
st.markdown("### AI-Powered (Database + Wikipedia)")

@st.cache_resource
def get_hero_list():
    """Loads hero list from the database."""
    _, _, df, _ = load_assets() 
    return sorted(df['name'].unique()), df

hero_list, df = get_hero_list()

# --- TABS ---
tab1, tab2 = st.tabs(["Standard Battle (DB)", "Mystery Heroes (Wikipedia)"])

# ================= TAB 1: STANDARD (DB vs DB) =================
with tab1:
    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        hero_a = st.selectbox("Hero A", hero_list, index=0)
    with col2:
        st.markdown('<div class="vs-font">VS</div>', unsafe_allow_html=True)
    with col3:
        hero_b = st.selectbox("Hero B", hero_list, index=1)

    if st.button("⚔️ SIMULATE BATTLE", type="primary", use_container_width=True):
        if hero_a == hero_b:
            st.error("⚠️ Please select two different heroes.")
        else:
            with st.spinner("Calculating..."):
                result, stats_a, stats_b = simulate_battle(hero_a, hero_b)
            
            if isinstance(result, str):
                st.error(result)
            else:
                winner = result['winner']
                if winner == hero_a:
                    st.markdown(f'<p class="big-font">{hero_a} WINS!</p>', unsafe_allow_html=True)
                    prob_a = result['prob_a']
                else:
                    st.markdown(f'<p class="big-font">{hero_b} WINS!</p>', unsafe_allow_html=True)
                    prob_a = result['prob_a']
                
                c1, c2 = st.columns(2)
                with c1: st.metric(f"Prob. {hero_a}", f"{prob_a:.2%}")
                with c2: st.metric(f"Prob. {hero_b}", f"{result['prob_b']:.2%}")

# ================= TAB 2: CUSTOM (Wiki vs Wiki) =================
# ================= TAB 2: CUSTOM (Wiki vs Wiki) =================
with tab2:
    st.write("Enter names for **BOTH** heroes. We will fetch their data from Wikipedia!")
    st.caption("Try: 'Kratos vs Goku' or 'Sherlock Holmes vs Iron Man'")
    
    col_c1, col_c2, col_c3 = st.columns([3, 1, 3])
    
    with col_c1:
        custom_name_a = st.text_input("Hero A (Any Name)", placeholder="e.g. Kratos")
        
    with col_c2:
        st.markdown('<div class="vs-font">VS</div>', unsafe_allow_html=True)
        
    with col_c3:
        custom_name_b = st.text_input("Hero B (Any Name)", placeholder="e.g. Goku")
        
    if st.button("🌍 FETCH & FIGHT", type="primary", use_container_width=True):
        if not custom_name_a or not custom_name_b:
            st.warning("Please enter names for both heroes.")
        elif custom_name_a == custom_name_b:
            st.error("Please select two different heroes.")
        else:
            with st.spinner(f"Searching Wikipedia..."):
                # Load necessary assets
                model, scaler, _, feature_cols = load_assets()
                current_dir = os.path.dirname(os.path.abspath(__file__))
                vectorizer_path = os.path.join(current_dir, 'models', 'tfidf_vectorizer.pkl')
                
                try:
                    vectorizer = joblib.load(vectorizer_path)
                except FileNotFoundError:
                    st.error("TF-IDF Vectorizer not found. Please run feature engineering.")
                    st.stop()
                
                # Create Vectors for BOTH heroes
                vector_a, msg_a = create_custom_hero_vector(custom_name_a, scaler, vectorizer, feature_cols)
                vector_b, msg_b = create_custom_hero_vector(custom_name_b, scaler, vectorizer, feature_cols)
                
                # --- ERROR HANDLING WITH FRIENDLY MESSAGES ---
                error_found = False
                
                if vector_a is None and vector_b is None:
                    st.error(msg_a) # Show error for A (since both failed usually same reason)
                    error_found = True
                elif vector_a is None:
                    st.error(msg_a) # Show specific error for Hero A
                    # If A failed, show success for B so user knows B was found
                    if vector_b is not None: st.success(msg_b)
                    error_found = True
                elif vector_b is None:
                    st.error(msg_b) # Show specific error for Hero B
                    if vector_a is not None: st.success(msg_a)
                    error_found = True
                
                # If errors occurred, stop here
                if error_found:
                    st.stop()
                
                # --- SUCCESS & PREDICTION ---
                st.success(msg_a)
                st.success(msg_b)
                
                # Calculate Difference (A - B)
                diff = vector_a - vector_b
                diff = diff.reshape(1, -1)
                
                # Predict
                proba = model.predict_proba(diff)[0]
                prob_a = proba[1]
                prob_b = proba[0]
                
                winner = custom_name_a if prob_a > 0.5 else custom_name_b
                
                st.markdown("---")
                if winner == custom_name_a:
                    st.markdown(f'<p class="big-font">{custom_name_a} WINS!</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="big-font">{custom_name_b} WINS!</p>', unsafe_allow_html=True)
                    
                c1, c2 = st.columns(2)
                with c1: st.metric(f"Prob. {custom_name_a}", f"{prob_a:.2%}")
                with c2: st.metric(f"Prob. {custom_name_b}", f"{prob_b:.2%}")
                
                st.info(f"Note: Both heroes used 'Average' baseline stats. The prediction is driven by their NLP (Text) features from Wikipedia.")
st.divider()
st.caption("Model: LightGBM + TF-IDF NLP + Wikipedia Integration.")