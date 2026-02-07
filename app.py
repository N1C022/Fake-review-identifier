from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import predictor
import json
import os
import csv
import io
import pandas as pd

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ReviewInput(BaseModel):
    text: str
    rating: float
    category: str

@app.get("/")
def read_root():
    response = FileResponse('static/index.html')
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.post("/predict")
def predict(input_data: ReviewInput):
    try:
        # The predictor expects text in quotes for consistency with CLI/training data quirks
        # We'll wrap it if it's not already, although the predictor logic we saw handles stripping too.
        # Let's just pass it as is, but ensure we handle the string format.
        # predictor.predict_review args: text, rating, category
        
        # We must ensure the text is treated as a string with quotes if the predictor logic strictly requires it for validation
        # The predictor validation logic:
        # if not (text.startswith('"') and text.endswith('"')) and not (text.startswith("'") and text.endswith("'")):
        #      errors.append("Text must be enclosed in quotation marks.")
        
        # So we ensure we wrap it in double quotes for the predictor call
        formatted_text = f'"{input_data.text}"'
        
        result = predictor.predict_review(formatted_text, input_data.rating, input_data.category)
        
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["details"])
             
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Handle 'text_' column from fake_reviews_dataset.csv
        if "text" not in df.columns and "text_" in df.columns:
            df.rename(columns={"text_": "text"}, inplace=True)
            
        if "text" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'text' (or 'text_') column")
            
        results = []
        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            rating = float(row.get("rating", 5.0))
            category = str(row.get("category", "Electronics_5"))
            
            # Ensure text format is compatible with predictor
            formatted_text = f'"{text}"' if not (text.startswith('"') or text.startswith("'")) else text
            
            pred = predictor.predict_review(formatted_text, rating, category)
            if "error" not in pred:
                results.append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "rating": rating,
                    "category": category,
                    "label": pred["label"],
                    "fake_probability": pred["fake_probability"],
                    "threshold_used": pred["threshold_used"]
                })
        
        # Sort by risk (fake_probability descending)
        results.sort(key=lambda x: x["fake_probability"], reverse=True)
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/demo-cases")
def get_demo_cases():
    try:
        with open("demo_cases.json", "r") as f:
            cases = json.load(f)
        return cases
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Demo cases not found")
