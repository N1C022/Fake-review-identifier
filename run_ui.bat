@echo off
echo Starting Fake Review Radar UI...
echo Open your browser to http://localhost:8080
python -m uvicorn app:app --reload --port 8080
pause
