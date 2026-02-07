# Fake Review Radar

A hackathon MVP for detecting fake reviews with explainability.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to generate the model and view evaluation metrics.
```bash
python train.py
```
This will create `fake_review_model.joblib`.
**Performance**: ~91% F1-Score, >90% Precision.

### 3. Run Predictions
Use the CLI to predict if a review is FAKE or GENUINE.

**Syntax**:
```bash
python predictor.py "'Review Text'" <Rating> <Category>
```
*Note: The review text must be enclosed in double quotes inside single quotes (or escaped quotes) to ensure it is treated as a single string argument containing quotes.*

**Example**:
```bash
python predictor.py "'This product is amazing!'" 5.0 Electronics_5
```

**Output**:
```json
{
  "label": "FAKE",
  "fake_probability": 0.85,
  "reasons": [
    "'amazing' (FAKE, score: 0.45)",
    ...
  ]
}
```

### 4. Run Demo Cases
Check `demo_cases.json` for examples of inputs to try.

### 5. Run the Web UI (New!)
Launch the interactive web interface to test reviews in real-time.

**Windows:**
Double-click `run_ui.bat` or run:
```cmd
run_ui.bat
```

**Mac/Linux/Bash:**
```bash
./run_ui.sh
```

Or manually:
```bash
python -m uvicorn app:app --reload
```
Open your browser at **http://localhost:8000**.


## Model Details
- **Algorithm**: Logistic Regression with TF-IDF (1-3 ngrams).
- **Preprocessing**: Stopword removal, standard scaling for ratings.
- **Explainability**: Top contributing n-grams (positive coefficients imply FAKE, negative imply GENUINE).

### Key Metrics
We intentionally optimise for high precision on FAKE to minimise false accusations, accepting lower recall.

**Confusion Matrix:**
```
[[3709  335]  (GENUINE: TN, FP)
 [ 389 3654]] (FAKE:    FN, TP)
```

**Precision/Recall at Thresholds:**
| Threshold | Precision | Recall |
| :--- | :--- | :--- |
| **0.7** | 0.957 | 0.838 |
| **0.8** | 0.975 | 0.778 |
| **0.9** | 0.988 | 0.671 |

**Recommended Threshold:** 0.439 (Privacy > 90%, Recall ~92%)

## Batch Upload
You can now upload a CSV file containing reviews to get batch predictions.
1. Open the Web UI.
2. Switch to the **Batch Upload** tab.
3. Upload a CSV with at least a `text` column.
4. View the results table sorted by risk (highest fake probability first).
