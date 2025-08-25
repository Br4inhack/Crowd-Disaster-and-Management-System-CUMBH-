#!/usr/bin/env python3
"""
Startup script for CUMBHv2 - Streamlit Crowd Analyzer
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import torch
        import cv2
        import numpy
        from PIL import Image
        import plotly
        import pandas
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "outputs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def start_streamlit():
    """Start the Streamlit app"""
    print("🚀 Starting CUMBHv2 Streamlit App...")
    print("📍 App will be available at: http://localhost:8501")
    print("📍 Press Ctrl+C to stop the app")
    print("\n" + "="*50)
    
    try:
        # Start the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {e}")

def main():
    print("🎯 CUMBHv2 - Streamlit Crowd Analyzer")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create directories
    create_directories()
    
    # Start Streamlit app
    start_streamlit()

if __name__ == "__main__":
    main()
