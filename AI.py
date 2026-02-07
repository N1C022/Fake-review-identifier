import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("fake_reviews_dataset.csv")
df["label"] = df["label"].map({"OR": 0, "CG": 1})  # FAKE is 1 now


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        ), "text_"),

        ("rating", Pipeline([
            ("scaler", StandardScaler())
        ]), ["rating"]),

        ("category", OneHotEncoder(handle_unknown="ignore"), ["category"])
    ]
)

pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight='balanced'))
])

X = df[["text_", "rating", "category"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

grid = GridSearchCV(
    pipeline,
    {"clf__C": [0.005, 0.01, 0.05, 0.1]},
    cv=5,
    scoring="precision",
    n_jobs=-1
)

grid.fit(X_train, y_train)



calibrated = CalibratedClassifierCV(
    grid.best_estimator_,
    method="sigmoid",
    cv=5
)

calibrated.fit(X_train, y_train)


print(classification_report(y_test, calibrated.predict(X_test)))


joblib.dump(calibrated, "fake_review_model.joblib")
print("Model saved as fake_review_model.joblib")
