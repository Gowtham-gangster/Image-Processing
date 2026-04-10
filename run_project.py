#!/usr/bin/env python
"""
Run the full project: install dependencies and start backend API.
"""
import subprocess
import sys
import os

def main():
    # Install requirements
    print("Installing dependencies...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=os.getcwd())
    if result.returncode != 0:
        print("Failed to install dependencies. Continuing anyway...")
    
    # Start backend API
    print("\n" + "="*60)
    print("Starting Backend API on http://localhost:8000")
    print("="*60)
    subprocess.run([sys.executable, "-m", "uvicorn", "api.index:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])

if __name__ == "__main__":
    main()
