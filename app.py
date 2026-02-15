import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.game_logic import simulate_battle, load_assets

st.set_page_config(page_title="Superhero Battle Simulator", page_icon="⚡", layout="centered")

st.markdown("""
<style>
    .big-font { font-size:45px !important; font-weight: bold; color: #FF4B4B; text-align: center; }
    .vs-font { font-size:30px !important; font-weight: bold; color: #AAAAAA; text-align: center; }
    .reason-box { background-color: #F0F2F6; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Superhero Battle Simulator")
st.markdown("### AI-Powered Prediction (Strict Hierarchy Mode)")

@st.cache_resource
def get_hero_list():
    _, _, df = load_assets()
    return sorted(df['name'].unique())

hero_list = get_hero_list()

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
        with st.spinner("Analyzing power tiers and stats..."):
            result, stats_a, stats_b = simulate_battle(hero_a, hero_b)
        
        if isinstance(result, str):
            st.error(result)
        else:
            winner = result['winner']
            
            # Display Winner
            if winner == hero_a:
                st.markdown(f'<p class="big-font">{hero_a} WINS!</p>', unsafe_allow_html=True)
                prob_a = result['prob_a']
                prob_b = result['prob_b']
            else:
                st.markdown(f'<p class="big-font">{hero_b} WINS!</p>', unsafe_allow_html=True)
                prob_a = result['prob_a']
                prob_b = result['prob_b']
            
            # Metrics
            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"Prob. {hero_a}", f"{prob_a:.2%}")
            with c2:
                st.metric(f"Prob. {hero_b}", f"{prob_b:.2%}")

            # --- NEW: BATTLE ANALYSIS (WHY?) ---
            st.write("#### 🧠 Battle Analysis (Why did they win?)")
            
            # Calculate absolute differences
            diff = stats_a - stats_b
            # Find the stat with the largest absolute difference
            # We exclude 'alignment_encoded' from the explanation as it's categorical
            explainable_cols = [c for c in diff.index if 'alignment' not in c]
            max_diff_stat = diff[explainable_cols].abs().idxmax()
            max_diff_val = diff[max_diff_stat]
            
            winner_advantage = "Advantage" if max_diff_val > 0 else "Disadvantage"
            
            explanation = f"The key factor was **{max_diff_stat.replace('_', ' ').title()}**. "
            if max_diff_val > 0:
                explanation += f"{hero_a} had a significantly higher score in this area."
            else:
                explanation += f"{hero_b} had a significantly higher score in this area."
                
            st.info(explanation)
            
            # Visualization
            st.write("#### 📊 Detailed Stat Comparison")
            compare_df = pd.DataFrame({
                'Stat': stats_a.index,
                hero_a: stats_a.values,
                hero_b: stats_b.values
            })
            st.bar_chart(compare_df.set_index('Stat'))

st.divider()
st.caption("Model: LightGBM trained on Power Tiers (Cosmic > Tech > Human).")