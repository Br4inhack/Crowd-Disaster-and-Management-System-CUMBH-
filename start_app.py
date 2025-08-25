#!/usr/bin/env python3
"""
Startup script for CUMBHv2 - FastAPI Crowd Analyzer
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import torch
        import cv2
        import numpy
        from PIL import Image
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

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI Backend Server...")
    print("📍 Backend will be available at: http://localhost:8000")
    print("📍 WebSocket endpoint: ws://localhost:8000/ws")
    print("📍 API documentation: http://localhost:8000/docs")
    print("\n" + "="*50)
    
    try:
        # Start the FastAPI server
        subprocess.run([sys.executable, "backend_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def main():
    print("🎯 CUMBHv2 - FastAPI Crowd Analyzer")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create directories
    create_directories()
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main()
