import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import os
import csv

# Page configuration
st.set_page_config(
    page_title="Real-time Crowd Monitor",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Real-time Crowd Monitor with YOLO")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
if st.sidebar.button("← Back to Dashboard"):
    st.switch_page("streamlit_app.py")

# Configuration Section
st.sidebar.subheader("⚙️ Configuration")

# Zone Configuration
st.sidebar.write("**Zone Settings:**")
zones = []
num_zones = st.sidebar.number_input("Number of Zones", min_value=1, max_value=10, value=3)

for i in range(num_zones):
    with st.sidebar.expander(f"Zone {i+1}"):
        zone_id = st.text_input(f"Zone {i+1} Name", value=f"Zone {i+1}", key=f"zone_name_{i}")
        area_sqm = st.number_input(f"Area (sqm)", min_value=100, max_value=5000, value=1000, key=f"zone_area_{i}")
        zones.append({'id': zone_id, 'area_sqm': area_sqm})

# Threshold Configuration
st.sidebar.write("**Density Thresholds (people/sqm):**")
safe_threshold = st.sidebar.slider("Safe Threshold", 0.1, 10.0, 4.2, 0.1)
warning_threshold = st.sidebar.slider("Warning Threshold", 0.1, 10.0, 5.4, 0.1)
critical_threshold = st.sidebar.slider("Critical Threshold", 0.1, 10.0, 6.0, 0.1)

THRESHOLDS = {
    'safe': safe_threshold,
    'warning': warning_threshold,
    'critical': critical_threshold
}

# Model Selection
model_size = st.sidebar.selectbox("YOLO Model Size", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"], index=2)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Video Processing")
    
    # Upload video
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if video_file:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
        
        # Load YOLO model
        @st.cache_resource
        def load_yolo_model(model_name):
            return YOLO(model_name)
        
        model = load_yolo_model(model_size)
        
        # Processing controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            process_video = st.button("🚀 Start Monitoring", type="primary")
        with col_ctrl2:
            frame_skip = st.number_input("Frame Skip", min_value=1, max_value=30, value=5)
        with col_ctrl3:
            max_frames = st.number_input("Max Frames", min_value=10, max_value=1000, value=100)
        
        if process_video:
            # Initialize CSV logging
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"crowd_monitor_log_{timestamp_str}.csv"
            
            # Create CSV file
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Frame", "Zone", "Density", "Status", "Total_People", "Alert"])
            
            # Video processing
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Data storage
            monitoring_data = []
            frame_count = 0
            processed_frames = 0
            
            # Video display placeholder
            video_placeholder = st.empty()
            
            while processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    continue
                
                processed_frames += 1
                
                # Update progress
                progress = processed_frames / max_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} (Processed: {processed_frames}/{max_frames})")
                
                # Run YOLO detection
                results = model(frame, verbose=False)
                
                # Filter for people (class 0 in COCO dataset)
                people_boxes = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        if int(box.cls) == 0:  # Person class
                            people_boxes.append(box)
                
                crowd_count = len(people_boxes)
                timestamp = datetime.now()
                
                # Process zones and calculate densities
                zones_data = []
                alerts = []
                
                for zone in zones:
                    density = crowd_count / zone['area_sqm']
                    
                    # Determine status
                    if density <= THRESHOLDS['safe']:
                        status = 'normal'
                        status_color = 'green'
                    elif density <= THRESHOLDS['warning']:
                        status = 'warning'
                        status_color = 'orange'
                    else:
                        status = 'critical'
                        status_color = 'red'
                    
                    zones_data.append({
                        'id': zone['id'],
                        'density': round(density, 3),
                        'status': status,
                        'color': status_color
                    })
                    
                    # Generate alerts
                    alert_msg = ""
                    if density > THRESHOLDS['warning']:
                        alert_msg = f"⚠️ Overcrowding at {zone['id']}! Density: {density:.2f} people/sqm"
                        alerts.append({
                            'zone': zone['id'],
                            'message': alert_msg,
                            'density': density,
                            'severity': 'critical' if density > THRESHOLDS['critical'] else 'warning'
                        })
                    
                    # Log to CSV
                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp.isoformat(),
                            frame_count,
                            zone['id'],
                            round(density, 3),
                            status,
                            crowd_count,
                            alert_msg
                        ])
                
                # Store monitoring data
                monitoring_data.append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'total_people': crowd_count,
                    'zones': zones_data,
                    'alerts': alerts
                })
                
                # Draw detection boxes on frame
                annotated_frame = frame.copy()
                for box in people_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, 'Person', (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add crowd count text
                cv2.putText(annotated_frame, f'Total People: {crowd_count}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display frame
                video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                
                # Small delay for visualization
                time.sleep(0.1)
            
            cap.release()
            os.unlink(video_path)  # Clean up temp file
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results in session state
            st.session_state.monitoring_data = monitoring_data
            st.session_state.csv_filename = csv_filename
            st.session_state.zones_config = zones
            
            st.success(f"✅ Monitoring completed! Processed {processed_frames} frames. Data saved to {csv_filename}")

with col2:
    st.subheader("📊 Live Monitoring")
    
    if 'monitoring_data' in st.session_state and st.session_state.monitoring_data:
        data = st.session_state.monitoring_data
        latest_data = data[-1]
        
        # Current Statistics
        st.metric("Current People Count", latest_data['total_people'])
        st.metric("Frames Processed", len(data))
        
        # Zone Status
        st.write("**Zone Status:**")
        for zone_data in latest_data['zones']:
            status_emoji = "🟢" if zone_data['status'] == 'normal' else "🟡" if zone_data['status'] == 'warning' else "🔴"
            st.write(f"{status_emoji} **{zone_data['id']}**: {zone_data['density']} people/sqm ({zone_data['status']})")
        
        # Active Alerts
        if latest_data['alerts']:
            st.write("**🚨 Active Alerts:**")
            for alert in latest_data['alerts']:
                if alert['severity'] == 'critical':
                    st.error(alert['message'])
                else:
                    st.warning(alert['message'])
        else:
            st.success("✅ No active alerts")
        
        # Real-time Chart
        if len(data) > 1:
            st.write("**People Count Trend:**")
            
            # Prepare data for plotting
            frames = [d['frame'] for d in data]
            people_counts = [d['total_people'] for d in data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frames,
                y=people_counts,
                mode='lines+markers',
                name='People Count',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="Real-time People Count",
                xaxis_title="Frame Number",
                yaxis_title="People Count",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a video and start monitoring to see live data")

# Results Analysis Section
if 'monitoring_data' in st.session_state and st.session_state.monitoring_data:
    st.markdown("---")
    st.subheader("📈 Analysis Results")
    
    data = st.session_state.monitoring_data
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏢 Zone Analysis", "🚨 Alerts Log", "📋 Data Export"])
    
    with tab1:
        st.subheader("Monitoring Overview")
        
        # Summary statistics
        total_frames = len(data)
        avg_people = np.mean([d['total_people'] for d in data])
        max_people = max([d['total_people'] for d in data])
        total_alerts = sum([len(d['alerts']) for d in data])
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Frames", total_frames)
        with col_stat2:
            st.metric("Avg People", f"{avg_people:.1f}")
        with col_stat3:
            st.metric("Peak People", max_people)
        with col_stat4:
            st.metric("Total Alerts", total_alerts)
        
        # People count over time
        frames = [d['frame'] for d in data]
        people_counts = [d['total_people'] for d in data]
        
        fig = px.line(x=frames, y=people_counts, 
                     title="People Count Over Time",
                     labels={'x': 'Frame Number', 'y': 'People Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Zone-wise Analysis")
        
        # Prepare zone data
        zone_analysis = {}
        for zone in st.session_state.zones_config:
            zone_analysis[zone['id']] = {
                'densities': [],
                'statuses': [],
                'alerts': 0
            }
        
        for frame_data in data:
            for zone_data in frame_data['zones']:
                zone_id = zone_data['id']
                if zone_id in zone_analysis:
                    zone_analysis[zone_id]['densities'].append(zone_data['density'])
                    zone_analysis[zone_id]['statuses'].append(zone_data['status'])
            
            for alert in frame_data['alerts']:
                if alert['zone'] in zone_analysis:
                    zone_analysis[alert['zone']]['alerts'] += 1
        
        # Zone comparison chart
        zone_names = list(zone_analysis.keys())
        avg_densities = [np.mean(zone_analysis[zone]['densities']) if zone_analysis[zone]['densities'] else 0 
                        for zone in zone_names]
        max_densities = [max(zone_analysis[zone]['densities']) if zone_analysis[zone]['densities'] else 0 
                        for zone in zone_names]
        alert_counts = [zone_analysis[zone]['alerts'] for zone in zone_names]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Average Density', x=zone_names, y=avg_densities, marker_color='lightblue'))
        fig.add_trace(go.Bar(name='Max Density', x=zone_names, y=max_densities, marker_color='red'))
        
        fig.update_layout(
            title="Zone Density Comparison",
            xaxis_title="Zones",
            yaxis_title="Density (people/sqm)",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Zone statistics table
        zone_stats = []
        for zone in zone_names:
            densities = zone_analysis[zone]['densities']
            if densities:
                zone_stats.append({
                    'Zone': zone,
                    'Avg Density': f"{np.mean(densities):.3f}",
                    'Max Density': f"{max(densities):.3f}",
                    'Min Density': f"{min(densities):.3f}",
                    'Alerts': zone_analysis[zone]['alerts']
                })
        
        if zone_stats:
            df_zones = pd.DataFrame(zone_stats)
            st.dataframe(df_zones, use_container_width=True)
    
    with tab3:
        st.subheader("Alerts Timeline")
        
        # Collect all alerts with timestamps
        all_alerts = []
        for frame_data in data:
            for alert in frame_data['alerts']:
                all_alerts.append({
                    'Frame': frame_data['frame'],
                    'Timestamp': frame_data['timestamp'].strftime("%H:%M:%S"),
                    'Zone': alert['zone'],
                    'Message': alert['message'],
                    'Density': alert['density'],
                    'Severity': alert['severity']
                })
        
        if all_alerts:
            df_alerts = pd.DataFrame(all_alerts)
            
            # Color code by severity
            def color_severity(val):
                if val == 'critical':
                    return 'background-color: #ffcccc'
                elif val == 'warning':
                    return 'background-color: #fff2cc'
                return ''
            
            styled_df = df_alerts.style.applymap(color_severity, subset=['Severity'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Alert frequency chart
            alert_frames = [alert['Frame'] for alert in all_alerts]
            fig = px.histogram(x=alert_frames, nbins=20, title="Alert Frequency Distribution")
            fig.update_xaxis(title="Frame Number")
            fig.update_yaxis(title="Number of Alerts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alerts were generated during monitoring")
    
    with tab4:
        st.subheader("Data Export")
        
        # CSV download
        if 'csv_filename' in st.session_state:
            csv_file = st.session_state.csv_filename
            if os.path.exists(csv_file):
                with open(csv_file, 'rb') as f:
                    csv_data = f.read()
                
                st.download_button(
                    label="📥 Download CSV Log",
                    data=csv_data,
                    file_name=csv_file,
                    mime="text/csv"
                )
                
                # Show sample data
                st.write("**Sample Data Preview:**")
                df_sample = pd.read_csv(csv_file)
                st.dataframe(df_sample.head(10), use_container_width=True)
        
        # Summary report
        st.write("**Monitoring Summary:**")
        summary_text = f"""
        **Monitoring Session Report**
        - Total Frames Processed: {len(data)}
        - Average People Count: {np.mean([d['total_people'] for d in data]):.1f}
        - Peak People Count: {max([d['total_people'] for d in data])}
        - Total Alerts Generated: {sum([len(d['alerts']) for d in data])}
        - Zones Monitored: {len(st.session_state.zones_config)}
        - Session Duration: {len(data)} frames
        """
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name=f"monitoring_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>👥 Real-time Crowd Monitor | Powered by YOLO Object Detection</p>
    </div>
    """,
    unsafe_allow_html=True
)
