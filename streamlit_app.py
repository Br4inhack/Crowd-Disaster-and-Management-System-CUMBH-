# streamlit_app.py
import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import sys
import csv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from datetime import datetime
import time
import tempfile
import base64

# Page configuration
st.set_page_config(
    page_title="CUMBHv2 - Crowd Analyzer",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
ZONAL_GRID_ROWS = 3
ZONAL_GRID_COLS = 3
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
                'timestamp': datetime.now().isoformat(),
                'density': density
            })
    return alerts

# --- CSV Logging Functions ---
def initialize_csv_loggers(video_name):
    """Initialize CSV files for logging analysis data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main analysis log
    analysis_log_path = f"analysis_log_{video_name}_{timestamp}.csv"
    analysis_headers = [
        'frame_number', 'timestamp', 'total_crowd_count', 'overall_density',
        'zone_1_1_density', 'zone_1_2_density', 'zone_1_3_density',
        'zone_2_1_density', 'zone_2_2_density', 'zone_2_3_density', 
        'zone_3_1_density', 'zone_3_2_density', 'zone_3_3_density',
        'alerts_count', 'high_alerts', 'medium_alerts'
    ]
    
    # Alerts log
    alerts_log_path = f"alerts_log_{video_name}_{timestamp}.csv"
    alerts_headers = [
        'frame_number', 'timestamp', 'zone_id', 'zone_name', 
        'alert_level', 'density_value', 'threshold_exceeded'
    ]
    
    # Zone statistics log
    zone_stats_path = f"zone_stats_{video_name}_{timestamp}.csv"
    zone_stats_headers = [
        'frame_number', 'timestamp', 'zone_id', 'zone_name',
        'density', 'crowd_count_estimate', 'status'
    ]
    
    # Initialize CSV files with headers
    with open(analysis_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(analysis_headers)
    
    with open(alerts_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(alerts_headers)
        
    with open(zone_stats_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(zone_stats_headers)
    
    return analysis_log_path, alerts_log_path, zone_stats_path

def log_frame_analysis(analysis_log_path, frame_count, total_count, zone_densities, alerts):
    """Log comprehensive frame analysis data"""
    timestamp = datetime.now().isoformat()
    overall_density = sum(zone_densities) / len(zone_densities)
    
    high_alerts = sum(1 for alert in alerts if alert['level'] == 'HIGH')
    medium_alerts = sum(1 for alert in alerts if alert['level'] == 'MEDIUM')
    
    row_data = [
        frame_count, timestamp, total_count, overall_density,
        *zone_densities,  # All 9 zone densities
        len(alerts), high_alerts, medium_alerts
    ]
    
    with open(analysis_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

def log_alerts(alerts_log_path, frame_count, alerts, thresholds):
    """Log alert-specific data"""
    timestamp = datetime.now().isoformat()
    
    for alert in alerts:
        threshold_type = 'high' if alert['level'] == 'HIGH' else 'medium'
        threshold_value = thresholds['zone_density'][threshold_type]
        
        row_data = [
            frame_count, timestamp, alert['zone_id'], alert['zone_name'],
            alert['level'], alert['density'], threshold_value
        ]
        
        with open(alerts_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

def log_zone_statistics(zone_stats_path, frame_count, zone_densities, total_count):
    """Log detailed zone-wise statistics"""
    timestamp = datetime.now().isoformat()
    
    for i, density in enumerate(zone_densities):
        row = i // 3 + 1
        col = i % 3 + 1
        zone_name = f'Zone {row}-{col}'
        
        # Estimate crowd count per zone (proportional to density)
        zone_crowd_estimate = int((density / sum(zone_densities)) * total_count) if sum(zone_densities) > 0 else 0
        
        status = 'normal'
        if density >= 0.1:
            status = 'high_density'
        elif density >= 0.05:
            status = 'medium_density'
        
        row_data = [
            frame_count, timestamp, i, zone_name,
            density, zone_crowd_estimate, status
        ]
        
        with open(zone_stats_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

@st.cache_resource
def load_model():
    """Load the CSRNet model with pre-trained weights"""
    # Force GPU usage if available, otherwise show error
    if torch.cuda.is_available():
        device = torch.device('cuda')
        st.success(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        st.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        st.error("⚠️ CUDA not available! Using CPU (will be slow)")
        st.info("For GPU acceleration, ensure CUDA and PyTorch GPU version are installed")
    
    model = CSRNet(load_weights=False)
    
    # Try to load pre-trained weights
    weights_path = "CSRNet_shanghaitechA.pth.tar"
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            st.success("✅ Pre-trained weights loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ Could not load pre-trained weights: {e}")
            st.info("Using model with ImageNet weights")
    else:
        st.warning("⚠️ Pre-trained weights file not found")
        st.info("Using model with ImageNet weights")
    
    model.to(device)
    model.eval()
    
    # Display device info
    st.info(f"Model loaded on: {device}")
    if device.type == 'cuda':
        st.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    return model, device

def analyze_video(video_path, progress_bar, status_text):
    """Analyze video and return results"""
    model, device = load_model()
    
    # Video processing
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize CSV loggers with proper path handling
    video_name = os.path.splitext(os.path.basename(video_path))[0] if video_path else "streamlit_video"
    try:
        analysis_log_path, alerts_log_path, zone_stats_path = initialize_csv_loggers(video_name)
        st.info(f"📊 CSV logging initialized: {analysis_log_path}")
    except Exception as e:
        st.error(f"CSV logging initialization failed: {e}")
        # Create fallback paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_log_path = f"analysis_log_{video_name}_{timestamp}.csv"
        alerts_log_path = f"alerts_log_{video_name}_{timestamp}.csv"
        zone_stats_path = f"zone_stats_{video_name}_{timestamp}.csv"
    
    # Create output video file with H.264 codec for better web compatibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"processed_video_{video_name}_{timestamp}.mp4"
    
    # Use H.264 codec for better Streamlit compatibility
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # If H.264 fails, try other codecs
    if not out.isOpened():
        st.warning("H.264 codec failed, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.warning("mp4v codec failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = f"processed_video_{video_name}_{timestamp}.avi"
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                st.error("All video codecs failed. Video processing cannot continue.")
                return None
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frame_count = 0
    all_alerts = []
    crowd_counts = []
    all_zone_densities = []
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_count += 1
        
        # Update progress
        progress = (frame_count / total_frames) * 100
        progress_bar.progress(progress / 100)
        status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # --- Inference ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Ensure model and input are on the same device
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear GPU cache for memory efficiency
        
        with torch.no_grad(): 
            density_map = model(input_tensor)
            
        # Move result back to CPU for processing
        density_map_cpu = density_map.cpu()
        
        # --- Insights ---
        total_person_count = int(density_map_cpu.detach().sum().numpy())
        density_map_np = density_map_cpu.squeeze(0).squeeze(0).numpy()
        zone_densities = get_zonal_densities(density_map_np, width, height, ZONAL_GRID_ROWS, ZONAL_GRID_COLS)
        alerts = check_alerts(zone_densities, THRESHOLDS, frame_count)
        
        crowd_counts.append(total_person_count)
        all_alerts.extend(alerts)
        all_zone_densities.append(zone_densities)
        
        # --- CSV Logging ---
        try:
            log_frame_analysis(analysis_log_path, frame_count, total_person_count, zone_densities, alerts)
            if alerts:
                log_alerts(alerts_log_path, frame_count, alerts, THRESHOLDS)
            log_zone_statistics(zone_stats_path, frame_count, zone_densities, total_person_count)
        except Exception as e:
            st.warning(f"CSV logging error at frame {frame_count}: {e}")
        
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
    
    # Verify output files exist and convert video if needed
    if not os.path.exists(output_path):
        st.error(f"Output video file not created: {output_path}")
    else:
        st.success(f"✅ Video processed: {output_path}")
        
        # Convert to web-compatible format if needed
        if output_path.endswith('.avi'):
            try:
                import subprocess
                web_output_path = output_path.replace('.avi', '_web.mp4')
                # Try to convert using ffmpeg if available
                result = subprocess.run(['ffmpeg', '-i', output_path, '-c:v', 'libx264', '-preset', 'fast', web_output_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(web_output_path):
                    output_path = web_output_path
                    st.success(f"✅ Video converted for web compatibility: {web_output_path}")
                else:
                    st.warning("FFmpeg conversion failed, using original AVI file")
            except Exception as e:
                st.warning(f"Video conversion failed: {e}")
    
    # Verify CSV files exist
    csv_status = []
    for name, path in [("Analysis", analysis_log_path), ("Alerts", alerts_log_path), ("Zone Stats", zone_stats_path)]:
        if os.path.exists(path):
            csv_status.append(f"✅ {name}: {path}")
        else:
            csv_status.append(f"❌ {name}: {path} (not found)")
    
    st.info("CSV Files Status:\n" + "\n".join(csv_status))
    
    return {
        'output_video_path': output_path,
        'alerts': all_alerts,
        'crowd_counts': crowd_counts,
        'zone_densities': all_zone_densities,
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'analysis_log_path': analysis_log_path,
        'alerts_log_path': alerts_log_path,
        'zone_stats_path': zone_stats_path
    }

def get_video_download_link(file_path, file_name):
    """Generate a download link for the video file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{file_name}">Download Processed Video</a>'
    return href

# --- Data Visualization Functions ---
def load_csv_data(analysis_log_path, alerts_log_path, zone_stats_path):
    """Load and return CSV data as DataFrames"""
    try:
        analysis_df = pd.read_csv(analysis_log_path)
        alerts_df = pd.read_csv(alerts_log_path) if os.path.exists(alerts_log_path) else pd.DataFrame()
        zone_stats_df = pd.read_csv(zone_stats_path)
        return analysis_df, alerts_df, zone_stats_df
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def create_crowd_trend_chart(analysis_df):
    """Create interactive crowd count trend chart"""
    if analysis_df.empty:
        return None
    
    fig = go.Figure()
    
    # Main crowd count line
    fig.add_trace(go.Scatter(
        x=analysis_df['frame_number'],
        y=analysis_df['total_crowd_count'],
        mode='lines+markers',
        name='Crowd Count',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='Frame: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add moving average
    if len(analysis_df) > 10:
        window = min(10, len(analysis_df) // 4)
        moving_avg = analysis_df['total_crowd_count'].rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=analysis_df['frame_number'],
            y=moving_avg,
            mode='lines',
            name=f'Moving Average ({window} frames)',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Frame: %{x}<br>Avg: %{y:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Crowd Count Over Time",
        xaxis_title="Frame Number",
        yaxis_title="Crowd Count",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_zone_heatmap(zone_stats_df):
    """Create zone density heatmap"""
    if zone_stats_df.empty:
        return None
    
    # Calculate average density per zone
    zone_avg = zone_stats_df.groupby('zone_name')['density'].mean().reset_index()
    
    # Create 3x3 grid
    heatmap_data = np.zeros((3, 3))
    for _, row in zone_avg.iterrows():
        zone_name = row['zone_name']
        if 'Zone' in zone_name:
            parts = zone_name.split(' ')[1].split('-')
            r, c = int(parts[0]) - 1, int(parts[1]) - 1
            heatmap_data[r, c] = row['density']
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=['Col 1', 'Col 2', 'Col 3'],
        y=['Row 1', 'Row 2', 'Row 3'],
        colorscale='Reds',
        hoverongaps=False,
        hovertemplate='Zone %{y}-%{x}<br>Avg Density: %{z:.6f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Average Zone Density Heatmap",
        height=400
    )
    
    return fig

def create_alerts_timeline(alerts_df):
    """Create alerts timeline chart"""
    if alerts_df.empty:
        return None
    
    # Convert timestamp to datetime
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    
    # Create scatter plot
    fig = go.Figure()
    
    # High alerts
    high_alerts = alerts_df[alerts_df['alert_level'] == 'HIGH']
    if not high_alerts.empty:
        fig.add_trace(go.Scatter(
            x=high_alerts['frame_number'],
            y=[1] * len(high_alerts),
            mode='markers',
            name='High Alerts',
            marker=dict(color='red', size=10, symbol='triangle-up'),
            text=high_alerts['zone_name'],
            hovertemplate='Frame: %{x}<br>Zone: %{text}<br>Level: HIGH<extra></extra>'
        ))
    
    # Medium alerts
    medium_alerts = alerts_df[alerts_df['alert_level'] == 'MEDIUM']
    if not medium_alerts.empty:
        fig.add_trace(go.Scatter(
            x=medium_alerts['frame_number'],
            y=[0.5] * len(medium_alerts),
            mode='markers',
            name='Medium Alerts',
            marker=dict(color='orange', size=8, symbol='circle'),
            text=medium_alerts['zone_name'],
            hovertemplate='Frame: %{x}<br>Zone: %{text}<br>Level: MEDIUM<extra></extra>'
        ))
    
    fig.update_layout(
        title="Alerts Timeline",
        xaxis_title="Frame Number",
        yaxis=dict(
            tickvals=[0.5, 1],
            ticktext=['Medium', 'High'],
            range=[0, 1.5]
        ),
        height=300,
        showlegend=True
    )
    
    return fig

def create_zone_comparison_chart(zone_stats_df):
    """Create zone-wise comparison chart"""
    if zone_stats_df.empty:
        return None
    
    # Calculate statistics per zone
    zone_stats = zone_stats_df.groupby('zone_name').agg({
        'density': ['mean', 'max', 'std'],
        'crowd_count_estimate': 'mean'
    }).round(6)
    
    zone_stats.columns = ['avg_density', 'max_density', 'density_std', 'avg_crowd']
    zone_stats = zone_stats.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Density', 'Maximum Density', 'Density Variation', 'Average Crowd Estimate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Average density
    fig.add_trace(
        go.Bar(x=zone_stats['zone_name'], y=zone_stats['avg_density'], name='Avg Density', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Max density
    fig.add_trace(
        go.Bar(x=zone_stats['zone_name'], y=zone_stats['max_density'], name='Max Density', marker_color='red'),
        row=1, col=2
    )
    
    # Density variation
    fig.add_trace(
        go.Bar(x=zone_stats['zone_name'], y=zone_stats['density_std'], name='Density Std', marker_color='orange'),
        row=2, col=1
    )
    
    # Average crowd estimate
    fig.add_trace(
        go.Bar(x=zone_stats['zone_name'], y=zone_stats['avg_crowd'], name='Avg Crowd', marker_color='green'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Zone-wise Analysis Comparison")
    return fig

def create_density_distribution(analysis_df):
    """Create density distribution histogram"""
    if analysis_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=analysis_df['overall_density'],
        nbinsx=30,
        name='Density Distribution',
        marker_color='skyblue',
        opacity=0.7
    ))
    
    # Add threshold lines
    fig.add_vline(x=0.05, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Threshold")
    fig.add_vline(x=0.1, line_dash="dash", line_color="red", 
                  annotation_text="High Threshold")
    
    fig.update_layout(
        title="Overall Density Distribution",
        xaxis_title="Density Value",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def generate_insights_summary(analysis_df, alerts_df, zone_stats_df):
    """Generate automated insights from the data"""
    insights = []
    
    if not analysis_df.empty:
        # Crowd count insights
        avg_crowd = analysis_df['total_crowd_count'].mean()
        max_crowd = analysis_df['total_crowd_count'].max()
        peak_frame = analysis_df.loc[analysis_df['total_crowd_count'].idxmax(), 'frame_number']
        
        insights.append(f"📊 **Average crowd count**: {avg_crowd:.1f} people")
        insights.append(f"🔝 **Peak crowd count**: {max_crowd} people at frame {peak_frame}")
        
        # Density insights
        avg_density = analysis_df['overall_density'].mean()
        insights.append(f"📈 **Average density**: {avg_density:.6f}")
        
        # Alert insights
        total_alerts = analysis_df['alerts_count'].sum()
        high_alerts = analysis_df['high_alerts'].sum()
        medium_alerts = analysis_df['medium_alerts'].sum()
        
        insights.append(f"🚨 **Total alerts**: {total_alerts} ({high_alerts} high, {medium_alerts} medium)")
        
        # Alert frequency
        alert_rate = (total_alerts / len(analysis_df)) * 100
        insights.append(f"⚡ **Alert frequency**: {alert_rate:.1f}% of frames triggered alerts")
    
    if not zone_stats_df.empty:
        # Zone insights
        zone_avg = zone_stats_df.groupby('zone_name')['density'].mean()
        hottest_zone = zone_avg.idxmax()
        coolest_zone = zone_avg.idxmin()
        
        insights.append(f"🔥 **Hottest zone**: {hottest_zone} (avg density: {zone_avg.max():.6f})")
        insights.append(f"❄️ **Coolest zone**: {coolest_zone} (avg density: {zone_avg.min():.6f})")
    
    return insights

def main():
    # Header
    st.title("👥 CUMBHv2 - Crowd Management Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Additional Analysis")

    if st.sidebar.button("🗺️ Crowd-Aware Route Planner"):
        st.switch_page("pages/route_planner.py")

    if st.sidebar.button("📹 Real-time Crowd Monitor"):
        st.switch_page("pages/crowd_monitor.py")
        
    if st.sidebar.button("🚨 Enhanced Crowd Monitor"):
        st.switch_page("pages/stampede_detector.py")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Emergency System")

    if st.sidebar.button("📱 Emergency Alerts"):
        st.switch_page("pages/emergency_alerts.py")
        
    # Model status
    st.sidebar.subheader("🤖 Model Status")
    try:
        model, device = load_model()
        st.sidebar.success(f"✅ Model Loaded")
        st.sidebar.info(f"Device: {device}")
        if device.type == 'cuda':
            st.sidebar.success("🚀 GPU Acceleration Enabled")
            st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.warning("⚠️ Using CPU (Slow)")
    except Exception as e:
        st.sidebar.error(f"❌ Model Error: {e}")
        return
    
    # Threshold settings
    st.sidebar.subheader("⚙️ Alert Thresholds")
    high_threshold = st.sidebar.slider("High Alert Threshold", 0.01, 0.5, 0.1, 0.01)
    medium_threshold = st.sidebar.slider("Medium Alert Threshold", 0.01, 0.3, 0.05, 0.01)
    
    # Update thresholds
    THRESHOLDS['zone_density']['high'] = high_threshold
    THRESHOLDS['zone_density']['medium'] = medium_threshold
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Upload & Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze crowd density"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
            
            # Display original video
            st.video(uploaded_file)
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary"):
                # Create progress elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run analysis
                with st.spinner("Analyzing video..."):
                    results = analyze_video(video_path, progress_bar, status_text)
                
                # Store results in session state
                st.session_state.analysis_results = results
                st.session_state.analysis_complete = True
                
                # Clean up
                os.unlink(video_path)
                
                st.success("✅ Analysis completed!")
    
    with col2:
        st.subheader("📊 Real-time Metrics")
        
        if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
            results = st.session_state.analysis_results
            
            # Crowd count statistics
            crowd_counts = results['crowd_counts']
            avg_count = np.mean(crowd_counts)
            max_count = np.max(crowd_counts)
            min_count = np.min(crowd_counts)
            
            st.metric("Average Crowd Count", f"{avg_count:.0f}")
            st.metric("Maximum Crowd Count", f"{max_count}")
            st.metric("Minimum Crowd Count", f"{min_count}")
            
            # Alert summary
            alerts = results['alerts']
            high_alerts = [a for a in alerts if a['level'] == 'HIGH']
            medium_alerts = [a for a in alerts if a['level'] == 'MEDIUM']
            
            st.metric("High Alerts", len(high_alerts))
            st.metric("Medium Alerts", len(medium_alerts))
    
    # Results section
    if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
        st.markdown("---")
        st.subheader("📈 Analysis Results")
        
        results = st.session_state.analysis_results
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎬 Processed Video", "🚨 Alerts", "📊 Analytics Dashboard", "🔥 Zone Heatmap", "📈 Insights", "📋 CSV Data"])
        
        with tab1:
            st.subheader("Processed Video with Heatmap")
            
            # Display processed video
            video_path = results['output_video_path']
            if os.path.exists(video_path):
                st.write(f"**Video file**: {video_path}")
                st.write(f"**File size**: {os.path.getsize(video_path)} bytes")
                
                # Try multiple methods to display video
                try:
                    # Method 1: Direct file path
                    st.video(video_path)
                except Exception as e:
                    st.warning(f"Direct video display failed: {e}")
                    try:
                        # Method 2: Read as bytes
                        with open(video_path, 'rb') as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                    except Exception as e2:
                        st.error(f"Video bytes display failed: {e2}")
                        st.info("Video file exists but cannot be displayed in browser. You can download it below.")
                
                # Download link
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_bytes,
                    file_name=os.path.basename(video_path),
                    mime="video/mp4" if video_path.endswith('.mp4') else "video/avi"
                )
            else:
                st.error(f"Processed video not found: {video_path}")
                # List files in current directory for debugging
                st.write("**Files in current directory:**")
                current_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi'))]
                if current_files:
                    for file in current_files:
                        st.write(f"- {file}")
                else:
                    st.write("No video files found")
        
        with tab2:
            st.subheader("🚨 Alert History")
            
            alerts = results['alerts']
            if alerts:
                # Convert to DataFrame for better display
                import pandas as pd
                df_alerts = pd.DataFrame(alerts)
                df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
                df_alerts = df_alerts.sort_values('timestamp', ascending=False)
                
                # Color code by alert level
                def color_alert(val):
                    if val == 'HIGH':
                        return 'background-color: #ffcccc'
                    elif val == 'MEDIUM':
                        return 'background-color: #fff2cc'
                    return ''
                
                st.dataframe(
                    df_alerts.style.applymap(color_alert, subset=['level']),
                    use_container_width=True
                )
                
                # Alert statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Alerts", len(alerts))
                with col2:
                    st.metric("High Alerts", len([a for a in alerts if a['level'] == 'HIGH']))
                with col3:
                    st.metric("Medium Alerts", len([a for a in alerts if a['level'] == 'MEDIUM']))
            else:
                st.info("No alerts detected during analysis.")
        
        with tab3:
            st.subheader("📊 Analytics Dashboard")
            
            # Load CSV data for advanced analytics
            analysis_df, alerts_df, zone_stats_df = load_csv_data(
                results['analysis_log_path'], 
                results['alerts_log_path'], 
                results['zone_stats_path']
            )
            
            if not analysis_df.empty:
                # Row 1: Crowd trends and alerts timeline
                col1, col2 = st.columns(2)
                
                with col1:
                    crowd_chart = create_crowd_trend_chart(analysis_df)
                    if crowd_chart:
                        st.plotly_chart(crowd_chart, use_container_width=True)
                
                with col2:
                    alerts_timeline = create_alerts_timeline(alerts_df)
                    if alerts_timeline:
                        st.plotly_chart(alerts_timeline, use_container_width=True)
                    else:
                        st.info("No alerts detected during analysis")
                
                # Row 2: Zone comparison and density distribution
                col3, col4 = st.columns(2)
                
                with col3:
                    zone_comparison = create_zone_comparison_chart(zone_stats_df)
                    if zone_comparison:
                        st.plotly_chart(zone_comparison, use_container_width=True)
                
                with col4:
                    density_dist = create_density_distribution(analysis_df)
                    if density_dist:
                        st.plotly_chart(density_dist, use_container_width=True)
            else:
                st.error("No analysis data available for visualization")
        
        with tab4:
            st.subheader("🔥 Zone Density Heatmap")
            
            # Load CSV data for zone heatmap
            analysis_df, alerts_df, zone_stats_df = load_csv_data(
                results['analysis_log_path'], 
                results['alerts_log_path'], 
                results['zone_stats_path']
            )
            
            if not zone_stats_df.empty:
                zone_heatmap = create_zone_heatmap(zone_stats_df)
                if zone_heatmap:
                    st.plotly_chart(zone_heatmap, use_container_width=True)
                
                # Zone statistics table
                st.subheader("Zone Statistics Summary")
                zone_summary = zone_stats_df.groupby('zone_name').agg({
                    'density': ['mean', 'max', 'min', 'std'],
                    'crowd_count_estimate': ['mean', 'max'],
                    'status': lambda x: (x != 'normal').sum()
                }).round(6)
                
                zone_summary.columns = ['Avg Density', 'Max Density', 'Min Density', 'Density Std', 'Avg Crowd', 'Max Crowd', 'Alert Frames']
                zone_summary = zone_summary.reset_index()
                
                st.dataframe(zone_summary, use_container_width=True)
            else:
                st.error("No zone statistics data available")
        
        with tab5:
            st.subheader("📈 AI-Generated Insights")
            
            # Load CSV data for insights
            analysis_df, alerts_df, zone_stats_df = load_csv_data(
                results['analysis_log_path'], 
                results['alerts_log_path'], 
                results['zone_stats_path']
            )
            
            if not analysis_df.empty:
                insights = generate_insights_summary(analysis_df, alerts_df, zone_stats_df)
                
                st.markdown("### 🤖 Automated Analysis Summary")
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                # Risk assessment
                st.markdown("### ⚠️ Risk Assessment")
                total_alerts = analysis_df['alerts_count'].sum()
                high_alerts = analysis_df['high_alerts'].sum()
                alert_rate = (total_alerts / len(analysis_df)) * 100
                
                if alert_rate > 50:
                    risk_level = "🔴 **HIGH RISK**"
                    risk_desc = "Frequent alerts detected. Immediate attention required."
                elif alert_rate > 20:
                    risk_level = "🟡 **MEDIUM RISK**"
                    risk_desc = "Moderate alert frequency. Monitor closely."
                else:
                    risk_level = "🟢 **LOW RISK**"
                    risk_desc = "Low alert frequency. Situation appears stable."
                
                st.markdown(f"**Overall Risk Level**: {risk_level}")
                st.markdown(f"**Assessment**: {risk_desc}")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                recommendations = []
                
                if high_alerts > 0:
                    recommendations.append("🚨 **High-priority zones identified** - Deploy additional security personnel")
                
                if alert_rate > 30:
                    recommendations.append("📊 **High alert frequency** - Consider crowd flow management")
                
                if not zone_stats_df.empty:
                    zone_avg = zone_stats_df.groupby('zone_name')['density'].mean()
                    if zone_avg.max() > zone_avg.mean() * 2:
                        recommendations.append("⚖️ **Uneven crowd distribution** - Implement crowd redistribution strategies")
                
                if not recommendations:
                    recommendations.append("✅ **Situation stable** - Continue regular monitoring")
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.error("No analysis data available for insights generation")
        
        with tab6:
            st.subheader("📋 CSV Data & Downloads")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Video Information:**")
                st.write(f"- Duration: {results['total_frames'] / results['fps']:.2f} seconds")
                st.write(f"- Frames: {results['total_frames']}")
                st.write(f"- FPS: {results['fps']}")
                st.write(f"- Resolution: {results['width']}x{results['height']}")
            
            with col2:
                st.write("**Analysis Results:**")
                st.write(f"- Average Crowd Count: {np.mean(results['crowd_counts']):.0f}")
                st.write(f"- Peak Crowd Count: {np.max(results['crowd_counts'])}")
                st.write(f"- Total Alerts: {len(results['alerts'])}")
                st.write(f"- High Risk Alerts: {len([a for a in results['alerts'] if a['level'] == 'HIGH'])}")
            
            # CSV file information and download links
            st.markdown("### 📊 Generated CSV Files")
            
            csv_files = [
                ("Analysis Log", results['analysis_log_path'], "Complete frame-by-frame analysis data"),
                ("Alerts Log", results['alerts_log_path'], "Detailed alert information"),
                ("Zone Statistics", results['zone_stats_path'], "Zone-wise density and crowd statistics")
            ]
            
            for name, path, description in csv_files:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        csv_data = f.read()
                    
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.write(f"**{name}**")
                        st.write(f"_{description}_")
                    with col_b:
                        st.write(f"Size: {len(csv_data)} bytes")
                    with col_c:
                        st.download_button(
                            label="📥 Download",
                            data=csv_data,
                            file_name=os.path.basename(path),
                            mime="text/csv"
                        )
                else:
                    st.error(f"CSV file not found: {name}")
            
            # Display sample data
            st.markdown("### 👀 Sample Data Preview")
            
            # Load and display sample CSV data
            analysis_df, alerts_df, zone_stats_df = load_csv_data(
                results['analysis_log_path'], 
                results['alerts_log_path'], 
                results['zone_stats_path']
            )
            
            if not analysis_df.empty:
                st.write("**Analysis Data (First 10 rows):**")
                st.dataframe(analysis_df.head(10), use_container_width=True)
            
            if not alerts_df.empty:
                st.write("**Alerts Data (First 10 rows):**")
                st.dataframe(alerts_df.head(10), use_container_width=True)
            
            if not zone_stats_df.empty:
                st.write("**Zone Statistics (First 10 rows):**")
                st.dataframe(zone_stats_df.head(10), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>CUMBHv2 - Crowd Management Dashboard | Powered by CSRNet AI Model</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
