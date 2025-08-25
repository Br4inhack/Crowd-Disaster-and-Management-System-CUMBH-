# backend_server.py
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import sys
from PIL import Image
from datetime import datetime
import asyncio
import nest_asyncio

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Apply patch for running asyncio in environments like Jupyter/Colab if needed
nest_asyncio.apply()

# Constants
ZONAL_GRID_ROWS = 3
ZONAL_GRID_COLS = 3
OUT_VIDEO_PATH = "outputs/alert_video.mp4"
THRESHOLDS = {
    'zone_density': {
        'high': 0.1,
        'medium': 0.05
    }
}

# --- CSRNet Model and Helper Functions ---
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        from torchvision import models
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self._initialize_weights()
            frontend_dict = self.frontend.state_dict()
            model_dict = mod.features.state_dict()
            frontend_dict.update({k: v for k, v in model_dict.items() if k in frontend_dict})
            self.frontend.load_state_dict(frontend_dict)
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        if dilation: d_rate = 2
        else: d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M': layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm: layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else: layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def get_zonal_densities(density_map_np, frame_width, frame_height, rows, cols):
    h, w = density_map_np.shape
    dh, dw = h // rows, w // cols
    zone_area_pixels = (frame_width / cols) * (frame_height / rows)
    zone_densities = []
    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * dh, c * dw
            y2, x2 = (r + 1) * dh, (c + 1) * dw
            zone_slice = density_map_np[y1:y2, x1:x2]
            count = np.sum(zone_slice)
            density = count / zone_area_pixels if zone_area_pixels > 0 else 0
            zone_densities.append(density)
    return zone_densities

def check_alerts(zone_densities, thresholds, frame_count):
    alerts = []
    high_thresh = thresholds['zone_density']['high']
    medium_thresh = thresholds['zone_density']['medium']
    for i, density in enumerate(zone_densities):
        level = None
        if density >= high_thresh: level = "HIGH"
        elif density >= medium_thresh: level = "MEDIUM"
        if level:
            row, col = i // 3 + 1, i % 3 + 1
            zone_name = f'Zone {row}-{col}'
            alerts.append({
                'zone_id': i, 
                'level': level, 
                'zone_name': zone_name, 
                'timestamp': datetime.now().isoformat()
            })
    return alerts

# Model loading function
def load_model():
    """Load the CSRNet model with pre-trained weights"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CSRNet(load_weights=False)
    
    # Try to load pre-trained weights
    weights_path = "CSRNet_shanghaitechA.pth.tar"
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Pre-trained weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load pre-trained weights: {e}")
            print("Using model with ImageNet weights")
    else:
        print("⚠️ Pre-trained weights file not found")
        print("Using model with ImageNet weights")
    
    model.to(device)
    model.eval()
    return model, device

# Global model instance
model, device = load_model()

# --- Modified Analysis Function for Streaming ---
async def run_analysis(video_path: str, websocket: WebSocket):
    # Setup
    if not os.path.exists(video_path):
        await websocket.send_json({"type": "error", "message": "Video file not found"})
        return
    
    # Video processing
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_count += 1
        
        # --- Inference ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad(): 
            density_map = model(input_tensor)
        
        # --- Insights ---
        total_person_count = int(density_map.detach().cpu().sum().numpy())
        density_map_np = density_map.squeeze(0).squeeze(0).cpu().numpy()
        zone_densities = get_zonal_densities(density_map_np, width, height, ZONAL_GRID_ROWS, ZONAL_GRID_COLS)
        alerts = check_alerts(zone_densities, THRESHOLDS, frame_count)
        
        # --- Send data over WebSocket ---
        progress = (frame_count / total_frames) * 100
        await websocket.send_json({
            "type": "progress",
            "frame": frame_count,
            "total_frames": total_frames,
            "progress": progress,
            "crowd_count": total_person_count
        })
        
        if alerts:
            await websocket.send_json({"type": "alerts", "data": alerts})
        
        # --- Visualization ---
        density_map_resized = cv2.resize(density_map_np, (width, height))
        density_map_norm = (density_map_resized - np.min(density_map_resized)) / (np.max(density_map_resized) - np.min(density_map_resized) + 1e-6)
        heatmap = (density_map_norm * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        
        # Draw alerts on overlay
        dh, dw = height // ZONAL_GRID_ROWS, width // ZONAL_GRID_COLS
        for alert in alerts:
            zone_id, level = alert['zone_id'], alert['level']
            r, c = zone_id // ZONAL_GRID_COLS, zone_id % ZONAL_GRID_COLS
            y_pos, x_pos = r * dh + dh // 2, c * dw + dw // 2
            alert_color = (0, 0, 255) if level == 'HIGH' else (0, 165, 255)
            cv2.putText(overlay, level, (x_pos - 40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 4, cv2.LINE_AA)
            cv2.putText(overlay, level, (x_pos - 40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, alert_color, 2, cv2.LINE_AA)
        
        cv2.putText(overlay, f'Crowd Count: {total_person_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(overlay)
    
    cap.release()
    out.release()
    await websocket.send_json({"type": "complete", "video_url": f"/outputs/alert_video.mp4"})

# --- FastAPI App ---
app = FastAPI(title="Crowd Analyzer API", version="2.0.0")

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the output videos statically
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
async def root():
    return {"message": "Crowd Analyzer API v2.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for a message (e.g., a file upload signal)
            message = await websocket.receive_text()
            if message == "start_analysis":
                # This assumes the file is already uploaded
                await run_analysis("uploads/uploaded_video.mp4", websocket)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = "uploads/uploaded_video.mp4"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"status": "File uploaded successfully", "file_path": file_path}

if __name__ == "__main__":
    print("🚀 Starting CUMBHv2 FastAPI Server...")
    print("📍 API: http://localhost:8000")
    print("📍 Docs: http://localhost:8000/docs")
    print("📍 WebSocket: ws://localhost:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000)