import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

def load_data(filepath):
    df = pd.read_csv(filepath)
    df["label"] = df["label"].map({"OR": 0, "CG": 1}) # FAKE is 1
    return df

def train_model():
    print("Loading data...")
    df = load_data("fake_reviews_dataset.csv")
    
    # Preprocessing
    # We use a refined TfidfVectorizer to handle stopwords better and include n-grams
    text_transformer = TfidfVectorizer(
        ngram_range=(1, 3), 
        min_df=5, # Ignore very rare words
        max_df=0.9, # Ignore very common words (domain specific stopwords)
        sublinear_tf=True,
        stop_words='english' # Remove common English stopwords
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "text_"),
            ("rating", numerical_transformer, ["rating"]),
            ("category", categorical_transformer, ["category"])
        ]
    )

    # Pipeline with Logistic Regression
    # We use LogisticRegression because it's interpretable and works well for text
    pipeline = Pipeline([
        ("features", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')) 
    ])

    X = df[["text_", "rating", "category"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Grid Search for C parameter
    print("Tuning hyperparameters...")
    grid = GridSearchCV(
        pipeline,
        {"clf__C": [0.01, 0.1, 1, 10]},
        cv=3,
        scoring="f1", # Optimize for F1-score to balance precision and recall
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"Best parameters: {grid.best_params_}")

    # Calibration (optional but good for probabilities)
    print("Calibrating model...")
    calibrated = CalibratedClassifierCV(
        best_model,
        method="sigmoid",
        cv="prefit" # Use the already fitted model
    )
    calibrated.fit(X_test, y_test) # Note: ideally we should have a validation set for calibration, but for hackathon split is okay-ish or cross-val

    # Re-fit on full training set if we want, or just use the best estimator
    # For simplicity and speed, let's use the best_estimator_ directly and see if probas are good enough
    # Actually, CalibratedClassifierCV with cv=5 (internal CV) is better if we have time, 
    # but let's stick to the grid search result which is robust enough. 
    # If we need probability calibration, we can wrap the *entire* pipeline in CalibratedClassifierCV
    # properly or just use LogisticRegression's predict_proba. 
    # LogisticRegression (liblinear) produces decent probabilities.
    
    final_model = best_model

    # Evaluation
    print("\nEvaluating model...")
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["GENUINE", "FAKE"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Precision-Recall Analysis
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"Average Precision: {ap:.3f}")

    # Find threshold for high precision (e.g., 0.95)
    target_precision = 0.90
    idx = np.where(precision >= target_precision)[0][0]
    suggested_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    print(f"Suggested threshold for >= {target_precision} precision: {suggested_threshold:.3f}")
    print(f"Recall at this threshold: {recall[idx]:.3f}")

    # Save
    print("Saving model...")
    joblib.dump(final_model, "fake_review_model.joblib")
    
    # Save vectorizer explicitly if needed for easy feature lookup, 
    # but it is inside the pipeline: final_model.named_steps['features'].named_transformers_['text']
    
    print("Done!")

if __name__ == "__main__":
    train_model()
