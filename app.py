from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import predictor
import json
import os

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

@app.get("/demo-cases")
def get_demo_cases():
    try:
        with open("demo_cases.json", "r") as f:
            cases = json.load(f)
        return cases
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Demo cases not found")
