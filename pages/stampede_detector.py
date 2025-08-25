import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import tempfile
import os
import time
import sys
from collections import Counter

# Add lib directory to path for emergency alert system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.emergency_alert_system import EmergencyAlertSystem, AlertLevel, DepartmentType

# Page configuration
st.set_page_config(
    page_title="Enhanced Crowd Monitor",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Enhanced Crowd Monitor")
st.markdown("**Real-time object detection and crowd monitoring with YOLO**")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
if st.sidebar.button("← Back to Dashboard"):
    st.switch_page("streamlit_app.py")

# Configuration Section
st.sidebar.subheader("⚙️ Detection Parameters")

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

# Emergency Alert Configuration
st.sidebar.write("**🚨 Emergency Alerts:**")
enable_alerts = st.sidebar.checkbox("Enable SMS Alerts", value=False)
if enable_alerts:
    alert_cooldown = st.sidebar.number_input("Alert Cooldown (minutes)", min_value=1, max_value=60, value=5)
    auto_escalate = st.sidebar.checkbox("Auto Escalate Critical Alerts", value=True)

THRESHOLDS = {
    'safe': safe_threshold,
    'warning': warning_threshold,
    'critical': critical_threshold
}

# Model Selection
model_size = st.sidebar.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"], index=2)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Video Analysis")
    
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
        
        # Initialize emergency alert system if enabled
        alert_system = None
        last_alert_time = {}
        if enable_alerts:
            try:
                alert_system = EmergencyAlertSystem()
                st.sidebar.success("🚨 Emergency alerts enabled")
            except Exception as e:
                st.sidebar.error(f"⚠️ Alert system error: {str(e)}")
                enable_alerts = False
        
        # Processing controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            start_analysis = st.button("🚀 Start Analysis", type="primary")
        with col_ctrl2:
            frame_skip = st.number_input("Frame Skip", min_value=1, max_value=30, value=2)
        with col_ctrl3:
            max_frames = st.number_input("Max Frames", min_value=10, max_value=1000, value=200)
        
        if start_analysis:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Initialize data storage
            records = []
            frame_num = 0
            processed_frames = 0
            alert_frames = 0
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Real-time display placeholders
            video_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # CSV filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"crowd_monitor_full_log_{timestamp_str}.csv"
            
            while processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Skip frames for performance
                if frame_num % frame_skip != 0:
                    continue
                
                processed_frames += 1
                
                # Update progress
                progress = processed_frames / max_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_num}/{total_frames} (Analyzed: {processed_frames}/{max_frames})")
                
                # YOLO detection for all objects
                results = model(frame, verbose=False)
                detections = results[0].boxes
                
                if detections is not None:
                    classes = [int(box.cls[0]) for box in detections]
                    confidences = [float(box.conf[0]) for box in detections]
                    
                    # Count all objects with confidence filtering
                    filtered_detections = [(cls, conf) for cls, conf in zip(classes, confidences) if conf > 0.5]
                    object_counts = Counter([cls for cls, conf in filtered_detections])
                    
                    # Create detailed object summary
                    if object_counts:
                        readable_counts = ", ".join([
                            f"{count} {results[0].names[cls]}" for cls, count in sorted(object_counts.items())
                        ])
                        total_objects = sum(object_counts.values())
                    else:
                        readable_counts = "No objects detected (low confidence)"
                        total_objects = 0
                    
                    # People count (class 0 = person)
                    crowd_count = object_counts.get(0, 0)
                    
                    # Vehicle count (cars, trucks, buses, motorcycles)
                    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
                    vehicle_count = sum(object_counts.get(cls, 0) for cls in vehicle_classes)
                    
                    # Other objects count
                    other_count = total_objects - crowd_count - vehicle_count
                else:
                    classes = []
                    object_counts = Counter()
                    readable_counts = "No objects detected"
                    crowd_count = 0
                    vehicle_count = 0
                    other_count = 0
                    total_objects = 0
                
                # Process zones and calculate densities
                zones_data = []
                alerts = []
                timestamp = datetime.now()
                
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
                        alert_msg = f"⚠ Overcrowding at {zone['id']}! Density: {density:.2f} people/sqm."
                        severity = 'critical' if density > THRESHOLDS['critical'] else 'warning'
                        alerts.append({
                            'zone': zone['id'],
                            'message': alert_msg,
                            'density': density,
                            'severity': severity
                        })
                        alert_frames += 1
                        
                        # Send emergency SMS alert if enabled
                        if enable_alerts and alert_system:
                            zone_key = f"{zone['id']}_{severity}"
                            current_time = datetime.now()
                            
                            # Check cooldown period
                            if (zone_key not in last_alert_time or 
                                (current_time - last_alert_time[zone_key]).total_seconds() > alert_cooldown * 60):
                                
                                # Determine alert level and departments
                                if severity == 'critical':
                                    alert_level = AlertLevel.CRITICAL
                                    departments = [DepartmentType.FIRE, DepartmentType.POLICE, DepartmentType.MEDICAL]
                                else:
                                    alert_level = AlertLevel.HIGH
                                    departments = [DepartmentType.SECURITY, DepartmentType.MANAGEMENT]
                                
                                # Create detailed alert message
                                emergency_msg = f"CROWD DENSITY ALERT: {alert_msg} Immediate attention required."
                                
                                additional_info = {
                                    "People Count": crowd_count,
                                    "Total Objects": total_objects,
                                    "Vehicles Present": vehicle_count,
                                    "Zone Area": f"{zone['area_sqm']} sqm",
                                    "Frame Number": frame_num,
                                    "Detection Confidence": "High"
                                }
                                
                                try:
                                    # Send emergency alert
                                    alert_result = alert_system.send_emergency_alert(
                                        alert_level=alert_level,
                                        message=emergency_msg,
                                        departments=departments,
                                        location=f"Monitoring Zone: {zone['id']}",
                                        additional_info=additional_info
                                    )
                                    
                                    if alert_result["success"]:
                                        last_alert_time[zone_key] = current_time
                                        st.sidebar.success(f"🚨 Emergency alert sent for {zone['id']}")
                                        
                                        # Schedule escalation for critical alerts
                                        if severity == 'critical' and auto_escalate:
                                            alert_system.schedule_escalation_alert(alert_result["alert_id"])
                                    else:
                                        st.sidebar.error(f"⚠️ Failed to send alert: {alert_result.get('reason', 'Unknown error')}")
                                        
                                except Exception as e:
                                    st.sidebar.error(f"⚠️ Alert system error: {str(e)}")
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                # Draw detection boxes for all objects with improved visualization
                if detections is not None:
                    for i, box in enumerate(detections):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Skip low confidence detections
                        if conf < 0.5:
                            continue
                            
                        label = results[0].names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Enhanced color coding
                        if cls == 0:  # Person
                            color = (0, 255, 0)  # Green
                        elif cls in [2, 3, 5, 7]:  # Vehicles
                            color = (0, 165, 255)  # Orange
                        else:  # Other objects
                            color = (255, 0, 0)  # Blue
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label with confidence
                        label_text = f"{label} {conf:.2f}"
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        # Background for text
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add status text based on alerts
                if alerts:
                    alert_text = f"⚠ {len(alerts)} ALERTS ACTIVE"
                    if enable_alerts and alert_system:
                        alert_text += " - SMS SENT"
                    cv2.putText(annotated_frame, alert_text, (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    cv2.putText(annotated_frame, "✅ All Zones Normal", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Add comprehensive object counts
                cv2.putText(annotated_frame, f"Total Objects: {total_objects}", (50, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"People: {crowd_count} | Vehicles: {vehicle_count} | Other: {other_count}", (50, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Details: {readable_counts[:60]}{'...' if len(readable_counts) > 60 else ''}", (50, 150),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
                
                # Display frame
                video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                
                # Store data for each zone
                for zone_data in zones_data:
                    record = {
                        "timestamp": timestamp.isoformat(),
                        "frame": frame_num,
                        "total_people": crowd_count,
                        "total_vehicles": vehicle_count,
                        "total_objects": total_objects,
                        "other_objects": other_count,
                        "object_breakdown": readable_counts,
                        "zone": zone_data['id'],
                        "density": zone_data['density'],
                        "status": zone_data['status'],
                        "alert": next((alert['message'] for alert in alerts if alert['zone'] == zone_data['id']), ""),
                        "sms_alert_sent": enable_alerts and any(alert['zone'] == zone_data['id'] and alert['severity'] in ['warning', 'critical'] for alert in alerts)
                    }
                    records.append(record)
                
                # Small delay for visualization
                time.sleep(0.05)
            
            cap.release()
            os.unlink(video_path)  # Clean up temp file
            
            progress_bar.empty()
            status_text.empty()
            
            # Save results to CSV
            if records:
                df = pd.DataFrame(records)
                df.to_csv(csv_filename, index=False)
                
                # Store in session state
                st.session_state.monitor_data = records
                st.session_state.monitor_csv = csv_filename
                st.session_state.monitor_stats = {
                    'total_frames': processed_frames,
                    'alert_frames': alert_frames,
                    'alert_rate': (alert_frames / processed_frames) * 100 if processed_frames > 0 else 0,
                    'zones_monitored': len(zones)
                }
                
                st.success(f"✅ Analysis completed! Processed {processed_frames} frames. Results saved to {csv_filename}")

with col2:
    st.subheader("📊 Real-time Metrics")
    
    if 'monitor_data' in st.session_state and st.session_state.monitor_data:
        data = st.session_state.monitor_data
        stats = st.session_state.monitor_stats
        
        # Get latest data by frame
        latest_frame = max([d['frame'] for d in data])
        latest_data = [d for d in data if d['frame'] == latest_frame]
        
        if latest_data:
            # Current Status
            alerts_active = any(d['alert'] for d in latest_data)
            if alerts_active:
                st.error("🚨 ALERTS ACTIVE")
            else:
                st.success("✅ All Zones Normal")
            
            # Key Metrics
            people_count = latest_data[0]['total_people']
            vehicle_count = latest_data[0].get('total_vehicles', 0)
            total_objects = latest_data[0].get('total_objects', 0)
            object_breakdown = latest_data[0].get('object_breakdown', 'No data')
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("People", people_count)
            with col_m2:
                st.metric("Vehicles", vehicle_count)
            with col_m3:
                st.metric("Total Objects", total_objects)
            
            st.write(f"**Object Breakdown:** {object_breakdown}")
            
            # Zone Status
            st.write("**Zone Status:**")
            for zone_data in latest_data:
                status_emoji = "🟢" if zone_data['status'] == 'normal' else "🟡" if zone_data['status'] == 'warning' else "🔴"
                st.write(f"{status_emoji} **{zone_data['zone']}**: {zone_data['density']} people/sqm ({zone_data['status']})")
            
            # Overall Statistics
            st.write("**Session Statistics:**")
            st.metric("Total Frames", stats['total_frames'])
            st.metric("Alert Frames", stats['alert_frames'])
            st.metric("Alert Rate", f"{stats['alert_rate']:.1f}%")
            st.metric("Zones Monitored", stats['zones_monitored'])
            
            # Real-time trend
            if len(data) > 10:
                st.write("**People Count Trend:**")
                
                # Get unique frames and their people counts
                frame_data = {}
                for d in data:
                    if d['frame'] not in frame_data:
                        frame_data[d['frame']] = d['total_people']
                
                recent_frames = sorted(frame_data.keys())[-20:]  # Last 20 frames
                recent_counts = [frame_data[f] for f in recent_frames]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recent_frames,
                    y=recent_counts,
                    mode='lines+markers',
                    name='People Count',
                    line=dict(color='red' if alerts_active else 'green')
                ))
                
                fig.update_layout(
                    height=250,
                    xaxis_title="Frame",
                    yaxis_title="People Count",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a video and start analysis to see real-time metrics")

# Results Analysis Section
if 'monitor_data' in st.session_state and st.session_state.monitor_data:
    st.markdown("---")
    st.subheader("📈 Analysis Results")
    
    data = st.session_state.monitor_data
    df = pd.DataFrame(data)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏢 Zone Analysis", "🚨 Alerts Log", "📋 Data Export"])
    
    with tab1:
        st.subheader("Analysis Overview")
        
        # Summary statistics
        unique_frames = len(set([d['frame'] for d in data]))
        alert_records = len([d for d in data if d['alert']])
        max_people = max([d['total_people'] for d in data])
        avg_people = np.mean([d['total_people'] for d in data])
        zones_count = len(set([d['zone'] for d in data]))
        
        # Enhanced statistics with object counts
        max_objects = max([d.get('total_objects', 0) for d in data])
        avg_objects = np.mean([d.get('total_objects', 0) for d in data])
        total_vehicles = sum([d.get('total_vehicles', 0) for d in data])
        
        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5, col_stat6 = st.columns(6)
        with col_stat1:
            st.metric("Unique Frames", unique_frames)
        with col_stat2:
            st.metric("Alert Records", alert_records)
        with col_stat3:
            st.metric("Avg People", f"{avg_people:.1f}")
        with col_stat4:
            st.metric("Peak People", max_people)
        with col_stat5:
            st.metric("Peak Objects", max_objects)
        with col_stat6:
            st.metric("Total Vehicles", total_vehicles)
        
        # Multi-object tracking over time
        frame_data = {}
        for d in data:
            if d['frame'] not in frame_data:
                frame_data[d['frame']] = {
                    'people': d['total_people'],
                    'vehicles': d.get('total_vehicles', 0),
                    'total_objects': d.get('total_objects', 0)
                }
        
        frames = sorted(frame_data.keys())
        people_counts = [frame_data[f]['people'] for f in frames]
        vehicle_counts = [frame_data[f]['vehicles'] for f in frames]
        total_object_counts = [frame_data[f]['total_objects'] for f in frames]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frames, y=people_counts, mode='lines+markers', 
                               name='People', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=frames, y=vehicle_counts, mode='lines+markers', 
                               name='Vehicles', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=frames, y=total_object_counts, mode='lines+markers', 
                               name='Total Objects', line=dict(color='blue')))
        
        fig.update_layout(
            title="Object Detection Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Object Count",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Zone Analysis")
        
        # Zone-wise analysis
        zone_analysis = {}
        for zone_name in set([d['zone'] for d in data]):
            zone_data = [d for d in data if d['zone'] == zone_name]
            zone_analysis[zone_name] = {
                'densities': [d['density'] for d in zone_data],
                'alerts': len([d for d in zone_data if d['alert']])
            }
        
        # Zone comparison chart
        zone_names = list(zone_analysis.keys())
        avg_densities = [np.mean(zone_analysis[zone]['densities']) for zone in zone_names]
        max_densities = [max(zone_analysis[zone]['densities']) for zone in zone_names]
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
        
        # Collect all alerts
        alert_data = [d for d in data if d['alert']]
        
        if alert_data:
            st.write(f"**{len(alert_data)} alert records found:**")
            
            # Create DataFrame for display
            alerts_df = pd.DataFrame([{
                'Frame': d['frame'],
                'Zone': d['zone'],
                'Density': f"{d['density']:.3f}",
                'Status': d['status'],
                'Alert': d['alert']
            } for d in alert_data])
            
            # Color code by status
            def color_status(val):
                if val == 'critical':
                    return 'background-color: #ffcccc'
                elif val == 'warning':
                    return 'background-color: #fff2cc'
                return ''
            
            styled_df = alerts_df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Alert frequency chart
            alert_frames = [d['frame'] for d in alert_data]
            fig = px.histogram(x=alert_frames, nbins=20, title="Alert Frequency Distribution")
            fig.update_xaxis(title="Frame Number")
            fig.update_yaxis(title="Number of Alerts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No alerts were generated during monitoring")
    
    with tab4:
        st.subheader("Data Export")
        
        # CSV download
        if 'monitor_csv' in st.session_state:
            csv_file = st.session_state.monitor_csv
            if os.path.exists(csv_file):
                with open(csv_file, 'rb') as f:
                    csv_data = f.read()
                
                st.download_button(
                    label="📥 Download Monitoring Log (CSV)",
                    data=csv_data,
                    file_name=csv_file,
                    mime="text/csv"
                )
        
        # Sample data preview with better formatting
        st.write("**Sample Data Preview:**")
        sample_data = data[:10] if len(data) >= 10 else data
        display_columns = ['timestamp', 'frame', 'total_people', 'total_vehicles', 
                          'total_objects', 'zone', 'density', 'status', 'object_breakdown']
        
        sample_records = []
        for d in sample_data:
            record = {col: d.get(col, 'N/A') for col in display_columns}
            # Truncate long object breakdown for display
            if 'object_breakdown' in record and len(str(record['object_breakdown'])) > 50:
                record['object_breakdown'] = str(record['object_breakdown'])[:50] + '...'
            sample_records.append(record)
        
        sample_df = pd.DataFrame(sample_records)
        st.dataframe(sample_df, use_container_width=True)
        
        # Analysis summary
        summary_report = f"""
**Enhanced Crowd Monitor Analysis Report**

Configuration:
- Zones Monitored: {len(set([d['zone'] for d in data]))}
- Safe Threshold: {THRESHOLDS['safe']} people/sqm
- Warning Threshold: {THRESHOLDS['warning']} people/sqm
- Critical Threshold: {THRESHOLDS['critical']} people/sqm

Results Summary:
- Unique Frames Analyzed: {len(set([d['frame'] for d in data]))}
- Total Records: {len(data)}
- Alert Records: {len([d for d in data if d['alert']])}
- Average People Count: {np.mean([d['total_people'] for d in data]):.1f}
- Peak People Count: {max([d['total_people'] for d in data])}
- Average Total Objects: {np.mean([d.get('total_objects', 0) for d in data]):.1f}
- Peak Total Objects: {max([d.get('total_objects', 0) for d in data])}
- Total Vehicle Detections: {sum([d.get('total_vehicles', 0) for d in data])}
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=summary_report,
            file_name=f"crowd_monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>👥 Enhanced Crowd Monitor | Powered by YOLO Multi-Object Detection</p>
    </div>
    """,
    unsafe_allow_html=True
)
