import pandas as pd
import joblib
import json
import sys
import re
import numpy as np

# Load the model
try:
    model = joblib.load("fake_review_model.joblib")
except FileNotFoundError:
    print(json.dumps({"error": "Model file not found. Please run train.py first."}))
    sys.exit(1)

# Helper function for human-readable reasons
def get_human_reason(term, direction):
    """Maps technical tokens to user-friendly explanations."""
    term = term.lower()
    
    if direction == "FAKE":
        if any(x in term for x in ["plot", "character", "story", "arc", "narrative"]):
            return "Generic storytelling keywords (common in AI generation)"
        if any(x in term for x in ["highly recommend", "must read", "amazing", "excellent"]):
            return "Overly generic praise without specifics"
        if any(x in term for x in ["structure", "writing", "author", "book"]):
            return "Formal or analytical tone typical of AI"
        return f"suspiciously generic usage of '{term}'"
    else:
        # GENUINE indicators
        if any(x in term for x in ["refund", "waste", "garbage", "trash", "worst"]):
            return "Strong, specific negative sentiment"
        if any(x in term for x in ["bought", "purchased", "arrived", "shipping", "box"]):
            return "Specific logistical details"
        return f"Specific detail: '{term}'"

# Load metadata
try:
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    RECOMMENDED_THRESHOLD = metadata.get("recommended_threshold", 0.5)
except FileNotFoundError:
    RECOMMENDED_THRESHOLD = 0.5

ALLOWED_CATEGORIES = {
    "Kindle_Store_5",
    "Books_5",
    "Pet_Supplies_5",
    "Home_and_Kitchen_5",
    "Electronics_5",
    "Sports_and_Outdoors_5",
    "Tools_and_Home_Improvement_5",
    "Clothing_Shoes_and_Jewelry_5",
    "Toys_and_Games_5",
    "Movies_and_TV_5"
}

def validate_input(text, rating, category):
    errors = []
    
    # Text validation: single line and enclosed in quotes
    if not isinstance(text, str):
        errors.append("Text must be a string.")
    else:
        if '\n' in text:
            errors.append("Text must be a single line.")
        if not (text.startswith('"') and text.endswith('"')) and not (text.startswith("'") and text.endswith("'")):
             # The requirement says "enclosed in quotation marks". 
             # We can be lenient and accept if passed as argument, but strict per requirements.
             # However, usually CLI args strip quotes. Let's assume input string provided *contains* quotes?
             # Or is it about the input format in a file?
             # The prompt says: "Review text must be a single line and enclosed in quotes".
             # If passed via CLI, user might pass:  predict_review '"Clean code"', 5.0, Books_5
             # Let's check if the string itself has quotes.
             if not (text.strip().startswith('"') and text.strip().endswith('"')):
                 errors.append("Text must be enclosed in quotation marks.")

    # Rating validation
    try:
        r = float(rating)
        if not (1.0 <= r <= 5.0):
             errors.append("Rating must be between 1.0 and 5.0.")
    except ValueError:
        errors.append("Rating must be a decimal number.")

    # Category validation
    if category not in ALLOWED_CATEGORIES:
        errors.append(f"Category must be one of: {', '.join(ALLOWED_CATEGORIES)}")

    return errors

def get_explanations(pipeline, text):
    # Access the vectorizer and classifier
    try:
        vectorizer = pipeline.named_steps['features'].named_transformers_['text']
        classifier = pipeline.named_steps['clf']
    except Exception as e:
        return ["Could not extract explanations from model structure."]

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Transform input
    # Note: We need to pass it as a list, and the column transformer expects a dict or df usually, 
    # but here we accessed the vectorizer directly.
    # However, the vectorizer expects raw text.
    
    # We strip the quotes for TF-IDF if they are part of the wrapper, 
    # but the model was trained on "text_" column. 
    # If the training data had quotes, we should keep them. 
    # The requirement says "enclosed in quotes", but standard CSVs usually don't have quotes INSIDE the field unless escaped.
    # I'll assume standard CSV reading in training didn't include surrounding quotes as part of the text content unless the CSV was malformed.
    # But the validation requires them. I will strip them for the model processing to match typical text.
    clean_text = text.strip('"').strip("'")
    
    tfidf_matrix = vectorizer.transform([clean_text])
    
    # Get coefficients
    # LogisticRegression coef_ is (n_classes, n_features)
    # For binary, it's (1, n_features) - positive class (FAKE) coefs
    if hasattr(classifier, 'coef_'):
        coefs = classifier.coef_[0]
    else:
        return ["Model does not support coefficient extraction."]

    # Calculate contribution: tfidf * coef
    # tfidf_matrix is sparse
    indices = tfidf_matrix.indices
    data = tfidf_matrix.data
    
    contributions = []
    for idx, split_val in zip(indices, data):
        feature_name = feature_names[idx]
        score = split_val * coefs[idx]
        contributions.append((feature_name, score))
        
    # Sort by absolute contribution or just contribution?
    # We want to know WHY it's fake (positive contribution) or GENUINE (negative contribution).
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_features = contributions[:5]
    
    reasons = []
    for feat, score in top_features:
        direction = "FAKE" if score > 0 else "GENUINE"
        human_label = get_human_reason(feat, direction)
        
        reasons.append({
            "term": feat,
            "direction": direction,
            "score": round(float(score), 3),
            "human_label": human_label
        })
        
    return reasons

def predict_review(text, rating, category):
    # Validate
    validation_errors = validate_input(text, rating, category)
    if validation_errors:
        return {
            "error": "Validation Failed",
            "details": validation_errors
        }

    # Prepare input for model
    # Model expects a DataFrame with text_, rating, category
    # Remove quotes for the model if they are just formatting wrapper
    clean_text = text.strip('"').strip("'")
    
    input_df = pd.DataFrame([{
        "text_": clean_text,
        "rating": float(rating),
        "category": category
    }])

    # Predict
    # The pipeline handles preprocessing
    pred_prob = model.predict_proba(input_df)[0][1] # Probability of FAKE (class 1)
    
    # Use recommended threshold for the verdict if available to optimize precision
    threshold = RECOMMENDED_THRESHOLD
    is_fake = pred_prob >= threshold

    result = {
        "label": "FAKE" if is_fake else "GENUINE",
        "fake_probability": round(float(pred_prob), 3),
        "threshold_used": threshold,
        "reasons": get_explanations(model, text)
    }
    
    return result

def predict_batch_df(df):
    """
    Efficiently predict for a DataFrame.
    Expected columns: text, rating, category
    """
    # 1. Preprocess Vectorized
    # We need to ensure columns match what the model pipeline expects.
    # The pipeline step 'features' -> ColumnTransformer expects 'text_', 'rating', 'category'.
    
    # Clean text (remove outer quotes if present)
    # Using pandas apply is faster than loop but slower than pure string ops, but string ops on series are fast.
    # df['text'] should be the source.
    
    # Create input_df for model
    input_df = df.copy()
    
    # Rename 'text' to 'text_' if needed or ensure text_ exists
    if 'text_' not in input_df.columns:
        if 'text' in input_df.columns:
             input_df['text_'] = input_df['text'].astype(str).str.strip('"').str.strip("'")
        else:
             return {"error": "Missing 'text' column"}
             
    # Ensure types
    input_df['rating'] = pd.to_numeric(input_df['rating'], errors='coerce').fillna(5.0)
    input_df['category'] = input_df['category'].fillna("Electronics_5")
    
    # 2. Batch Predict
    try:
        # Get probabilities
        probas = model.predict_proba(input_df)[:, 1]
        
        # 3. Format Results
        results = []
        
        # We'll use the recommended threshold from metadata
        threshold = RECOMMENDED_THRESHOLD
        
        # We need to iterate to create the result list, but we have the probas pre-calculated.
        # This loop is just for dictionary creation, which is much faster than running the model loop.
        
        # To get explanations, doing it per-row is still expensive.
        # For batch, maybe we skip explanations? Or we only generate them for the high-risk ones?
        # Generating explanations requires feature extraction for each text.
        # Let's Skip explanations for batch to keep it fast, or maybe only for the top ones later in app.py?
        # For now, let's leave reasons empty in batch or simple placeholder.
        
        # Let's iterate index and proba
        for idx, prob in enumerate(probas):
            is_fake = prob >= threshold
            
            # Helper to get original text for display
            original_text = str(df.iloc[idx].get('text', ''))
            
            results.append({
                "text": original_text[:100] + "..." if len(original_text) > 100 else original_text,
                "rating": float(input_df.iloc[idx]['rating']),
                "category": str(input_df.iloc[idx]['category']),
                "label": "FAKE" if is_fake else "GENUINE",
                "fake_probability": round(float(prob), 3),
                "threshold_used": threshold,
                 "reasons": [] # Skipping detailed reasons for batch performance
            })
            
        return results
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Simple CLI
    if len(sys.argv) == 4:
        text_arg = sys.argv[1]
        rating_arg = sys.argv[2]
        category_arg = sys.argv[3]
        
        print(json.dumps(predict_review(text_arg, rating_arg, category_arg), indent=2))
    else:
        print("Usage: python predictor.py \"<text>\" <rating> <category>")
        # Example for interactive testing if no args
        print("\n--- Interactive Mode ---")
        t = input("Review text (in quotes): ")
        r = input("Rating: ")
        c = input("Category: ")
        print(json.dumps(predict_review(t, r, c), indent=2))
