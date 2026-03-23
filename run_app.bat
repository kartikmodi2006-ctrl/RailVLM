@echo off
echo Starting RailVLM Server...
C:\Users\hardi\AppData\Local\Programs\Python\Python310\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
pause
