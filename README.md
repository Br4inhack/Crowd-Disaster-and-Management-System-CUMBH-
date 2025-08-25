# CUMBHv2 - Crowd Analyzer (FastAPI + Streamlit)

This repository contains two versions of the Crowd Management Dashboard:
1. **FastAPI Version** - WebSocket-based real-time communication
2. **Streamlit Version** - Interactive web application (Recommended)

## Features

- **CSRNet Model**: AI-powered crowd density estimation
- **Real-time Analysis**: Live video processing with progress tracking
- **Alert System**: Zone-based density alerts with configurable thresholds
- **Interactive Dashboard**: Beautiful UI with charts and analytics
- **Video Processing**: Heatmap overlay with crowd count visualization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose Your Version

#### Option A: Streamlit App (Recommended)
```bash
python start_streamlit.py
```
Or directly:
```bash
streamlit run streamlit_app.py
```

#### Option B: FastAPI Backend
```bash
python start_app.py
```

### 3. Access the Application

#### Streamlit App
- **URL**: http://localhost:8501
- **Features**: Interactive dashboard, real-time analysis, charts, alerts

#### FastAPI Backend
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws

## Streamlit App Features

### 🎛️ Interactive Controls
- **Model Status**: Real-time model loading status
- **Alert Thresholds**: Adjustable high/medium alert levels
- **Device Information**: CPU/GPU usage display

### 📹 Video Analysis
- **Drag & Drop Upload**: Easy video file upload
- **Progress Tracking**: Real-time processing progress
- **Original Video Preview**: View uploaded video before analysis

### 📊 Real-time Metrics
- **Crowd Count Statistics**: Average, max, min counts
- **Alert Summary**: High and medium alert counts
- **Live Updates**: Metrics update during processing

### 📈 Analysis Results
- **Processed Video**: Download processed video with heatmap
- **Alert History**: Detailed alert log with timestamps
- **Crowd Count Chart**: Interactive time-series visualization
- **Zone Analysis**: Per-zone alert breakdown
- **Summary Report**: Comprehensive analysis overview

## Directory Structure

```
CUMBHv2/
├── streamlit_app.py           # Streamlit application (NEW)
├── start_streamlit.py         # Streamlit startup script (NEW)
├── backend_server.py          # FastAPI backend server
├── requirements.txt           # Python dependencies
├── start_app.py              # FastAPI startup script
├── CSRNet_shanghaitechA.pth.tar  # Model weights
├── app/                      # Frontend components (FastAPI version)
├── components/               # UI components (FastAPI version)
├── uploads/                  # Uploaded videos (created automatically)
└── outputs/                  # Processed videos (created automatically)
```

## How It Works

### Streamlit Version
1. **Video Upload**: Drag & drop video file
2. **Model Loading**: CSRNet model loads with pre-trained weights
3. **Frame Processing**: Each frame analyzed for crowd density
4. **Real-time Updates**: Progress bar and metrics update live
5. **Results Display**: Interactive tabs show all analysis data
6. **Download**: Processed video available for download

### FastAPI Version
1. **Video Upload**: Upload via API endpoint
2. **WebSocket Communication**: Real-time progress updates
3. **Frame Processing**: Backend processes video frame by frame
4. **Results**: Processed video and alert data returned

## Model Information

- **Model**: CSRNet (Crowd Scene Regression Network)
- **Weights**: Pre-trained on ShanghaiTech dataset
- **Input**: Video frames (RGB)
- **Output**: Density maps and person counts
- **Zones**: 3x3 grid analysis for localized alerts

## Configuration

### Alert Thresholds
- **High Alert**: Default 0.1 (adjustable in Streamlit sidebar)
- **Medium Alert**: Default 0.05 (adjustable in Streamlit sidebar)

### Supported Video Formats
- MP4, AVI, MOV, MKV

## Troubleshooting

### Common Issues
- **Port 8501 in use**: Change port in `start_streamlit.py`
- **Model loading issues**: Ensure `CSRNet_shanghaitechA.pth.tar` is in the directory
- **Dependencies**: Run `pip install -r requirements.txt`
- **Memory issues**: Use shorter videos for testing

### Performance Tips
- **GPU Usage**: Automatically detected and used if available
- **Video Size**: Larger videos take longer to process
- **Batch Processing**: Consider processing videos in smaller chunks

## Development

### Streamlit Version
- **Framework**: Streamlit
- **Charts**: Plotly
- **Data**: Pandas
- **AI Model**: CSRNet (PyTorch)

### FastAPI Version
- **Backend**: FastAPI
- **WebSockets**: Real-time communication
- **Frontend**: React/Next.js
- **AI Model**: CSRNet (PyTorch)

## API Endpoints (FastAPI Version)

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /upload/` - Upload video file
- `WebSocket /ws` - Real-time communication for analysis progress
