import wikipedia
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_text(text):
    """Same cleaning used in training."""
    if pd.isna(text) or text == "Unknown":
        return ""
    text = str(text).lower()
    return text

def fetch_wikipedia_text(hero_name):
    """Fetches the summary text from Wikipedia with better error handling."""
    try:
        # Search for the page
        page = wikipedia.page(hero_name, auto_suggest=False)
        return page.summary, "Success"
        
    except wikipedia.exceptions.PageError:
        # The page does not exist
        return None, f"❌ No Wikipedia data found for '{hero_name}'.<br><br>💡 **Tip:** Try the **full name** (e.g., 'Tony Stark' instead of 'Iron Man')."
        
    except wikipedia.exceptions.DisambiguationError as e:
        # Multiple pages found (e.g., "Flash" could be Barry Allen or Wally West)
        # We try the first option automatically
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.summary, f"ℹ️ Multiple options found. Using '{page.title}'."
        except:
            return None, f"❌ Could not find a specific page for '{hero_name}'.<br>💡 Try adding their surname (e.g., 'Peter Parker')."
            
    except Exception as e:
        # Network errors or other issues
        return None, "⚠️ Connection Error. Please check your internet connection."

def create_custom_hero_vector(hero_name, scaler, vectorizer, feature_cols):
    """
    Creates a feature vector for a hero not in the DB.
    Handles the 'power_tier' scaling mismatch correctly.
    """
    # 1. Get Text
    text = fetch_wikipedia_text(hero_name)
    if text is None:
        return None, "Hero not found on Wikipedia or connection error."
    
    clean_txt = clean_text(text)
    
    # 2. Generate NLP features
    tfidf_matrix = vectorizer.transform([clean_txt])
    nlp_features = tfidf_matrix.toarray()[0]
    nlp_cols_from_vectorizer = [f"nlp_{word}" for word in vectorizer.get_feature_names_out()]
    
    # 3. Separate 'power_tier' from other features because the scaler doesn't know it.
    # The scaler expects N-1 columns (everything except power_tier).
    cols_to_scale = [c for c in feature_cols if c != 'power_tier']
    
    # 4. Build dictionary for the columns we need to SCALE
    scale_dict = {}
    
    for col in cols_to_scale:
        if 'nlp_' in col:
            # Fill NLP features
            if col in nlp_cols_from_vectorizer:
                idx = nlp_cols_from_vectorizer.index(col)
                scale_dict[col] = nlp_features[idx]
            else:
                scale_dict[col] = 0
        elif col == 'lore_depth_score':
            scale_dict[col] = len(clean_txt)
        elif col == 'power_diversity_score':
            scale_dict[col] = 0 # Average diversity
        elif col == 'alignment_encoded':
            scale_dict[col] = 0 # Unknown
        else:
            # Stats: 0 represents Median/Average
            scale_dict[col] = 0
            
    # 5. Scale the data (Only the columns the scaler knows)
    df_scale = pd.DataFrame([scale_dict], columns=cols_to_scale)
    scaled_data = scaler.transform(df_scale)[0]
    
    # 6. Reconstruct the FULL vector (Scaled Cols + Power Tier)
    final_vector = []
    scale_idx = 0
    
    for col in feature_cols:
        if col == 'power_tier':
            # Add the Tier Value (Unscaled, e.g., 0, 1, 2)
            # We default to Tier 1 (Enhanced) for unknown heroes
            final_vector.append(1) 
        else:
            # Add the Scaled Value
            final_vector.append(scaled_data[scale_idx])
            scale_idx += 1
            
    return np.array(final_vector), f"Data fetched for '{hero_name}'."