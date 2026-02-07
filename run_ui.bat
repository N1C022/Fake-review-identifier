@echo off
echo Starting Fake Review Radar UI...
echo Open your browser to http://localhost:8000
python -m uvicorn app:app --reload
pause
