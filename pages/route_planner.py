import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from skimage.graph import route_through_array
import tempfile
import os
import requests
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="Crowd-Aware Route Planner",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Crowd-Aware Route Planner (Video Input)")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
if st.sidebar.button("← Back to Dashboard"):
    st.switch_page("streamlit_app.py")

# GPS API Integration Section
st.sidebar.subheader("🌍 GPS Integration")
use_gps = st.sidebar.checkbox("Enable GPS Coordinates")

if use_gps:
    st.sidebar.write("**Location Settings:**")
    latitude = st.sidebar.number_input("Latitude", value=40.7128, format="%.6f")
    longitude = st.sidebar.number_input("Longitude", value=-74.0060, format="%.6f")
    zoom_level = st.sidebar.slider("Map Zoom Level", 10, 20, 15)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Video Processing")
    
    # Upload video
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if video_file:
        st.info("Processing video for detections...")
        temp_file = f"temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(video_file.read())

        # Load YOLO
        @st.cache_resource
        def load_yolo_model():
            return YOLO("yolov8n.pt")
        
        model = load_yolo_model()
        cap = cv2.VideoCapture(temp_file)

        max_detections = 0
        best_frame = None
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Progress bar for video processing
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process all frames to find the one with most detections
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")

            results = model(frame)
            total_objs = sum([len(r.boxes.xyxy) for r in results])
            if total_objs > max_detections:
                max_detections = total_objs
                best_frame = frame.copy()

        cap.release()
        os.unlink(temp_file)  # Clean up temp file
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Selected frame with max detections: {max_detections}")

        if best_frame is not None:
            # Convert frame to grayscale and binary crowd map
            gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
            crowd_map = (gray > 150).astype(np.float32)  # white=navigable, black=obstacles
            
            # Display original frame and crowd map
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(best_frame, caption="Original Frame", use_column_width=True)
            with col_b:
                st.image(crowd_map, caption="Navigation map (white=navigable paths, black=obstacles)", use_column_width=True)

            # User selects start/end points
            st.subheader("🎯 Route Planning")
            st.write("Set start and end points (pixels in free white areas):")
            
            col_start, col_end = st.columns(2)
            with col_start:
                st.write("**Start Point:**")
                start_x = st.number_input("Start X", min_value=0, max_value=crowd_map.shape[1]-1, value=0)
                start_y = st.number_input("Start Y", min_value=0, max_value=crowd_map.shape[0]-1, value=0)
            
            with col_end:
                st.write("**End Point:**")
                end_x = st.number_input("End X", min_value=0, max_value=crowd_map.shape[1]-1, value=crowd_map.shape[1]-1)
                end_y = st.number_input("End Y", min_value=0, max_value=crowd_map.shape[0]-1, value=crowd_map.shape[0]-1)

            if st.button("🚀 Compute Safe Route", type="primary"):
                with st.spinner("Computing optimal route..."):
                    try:
                        # Validate start and end points are in navigable areas
                        if crowd_map[start_y, start_x] == 0:
                            st.error(f"❌ Start point ({start_x}, {start_y}) is in an obstacle area (black). Please select a white area.")
                            st.stop()
                        
                        if crowd_map[end_y, end_x] == 0:
                            st.error(f"❌ End point ({end_x}, {end_y}) is in an obstacle area (black). Please select a white area.")
                            st.stop()
                        
                        # Enhanced pathfinding: white areas (1) = navigable (cost=1), black areas (0) = obstacles (cost=inf)
                        # Add small buffer around obstacles for safer routing
                        from scipy import ndimage
                        
                        # Create obstacle buffer (dilate black areas slightly)
                        obstacle_mask = (crowd_map == 0)
                        buffered_obstacles = ndimage.binary_dilation(obstacle_mask, structure=np.ones((3,3)))
                        
                        # Create cost map with buffered obstacles
                        cost_map = np.ones_like(crowd_map, dtype=np.float32)
                        cost_map[buffered_obstacles] = np.inf
                        cost_map[obstacle_mask] = np.inf  # Original obstacles remain infinite cost
                        
                        # Ensure start and end points are still accessible after buffering
                        cost_map[start_y, start_x] = 1
                        cost_map[end_y, end_x] = 1
                        
                        path, cost = route_through_array(cost_map, (start_y, start_x), (end_y, end_x), fully_connected=True)
                        path = np.array(path)
                        
                        # Verify path is completely obstacle-free
                        path_valid = True
                        for py, px in path:
                            if crowd_map[py, px] == 0:
                                path_valid = False
                                break
                        
                        if not path_valid:
                            st.warning("⚠️ Initial path intersected obstacles. Computing alternative route...")
                            # Use only original navigable areas without buffer
                            cost_map = np.where(crowd_map==1, 1, np.inf)
                            path, cost = route_through_array(cost_map, (start_y, start_x), (end_y, end_x), fully_connected=True)
                            path = np.array(path)

                        # Overlay route on the image
                        route_img = np.stack([crowd_map*255]*3, axis=-1).astype(np.uint8)
                        
                        # Draw start and end points
                        cv2.circle(route_img, (start_x, start_y), 5, (255, 0, 0), -1)  # Blue start
                        cv2.circle(route_img, (end_x, end_y), 5, (0, 0, 255), -1)    # Red end
                        
                        # Draw path with thicker line for better visibility
                        for y, x in path:
                            route_img[y, x] = [0, 255, 0]  # green path
                            # Add neighboring pixels for thicker path visualization
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < route_img.shape[0] and 0 <= nx < route_img.shape[1]:
                                        if crowd_map[ny, nx] == 1:  # Only draw on navigable areas
                                            route_img[ny, nx] = [0, 200, 0]  # Slightly darker green for border

                        # Final validation: ensure entire path is obstacle-free
                        obstacles_in_path = 0
                        for py, px in path:
                            if crowd_map[py, px] == 0:
                                obstacles_in_path += 1
                        
                        if obstacles_in_path > 0:
                            st.error(f"❌ Route contains {obstacles_in_path} obstacle pixels. Try different start/end points.")
                        else:
                            st.success("✅ Route is completely obstacle-free!")

                        st.image(route_img, caption="Obstacle-free route (green) through navigable white areas", use_column_width=True)
                        
                        # Route statistics
                        route_length = len(path)
                        navigable_pixels = np.sum(crowd_map)
                        total_pixels = crowd_map.shape[0] * crowd_map.shape[1]
                        navigable_density = (navigable_pixels / total_pixels) * 100
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Route Length", f"{route_length} pixels")
                        with col_stat2:
                            st.metric("Navigable Area", f"{navigable_density:.1f}%")
                        with col_stat3:
                            st.metric("Route Cost", f"{cost:.2f}")
                        
                        # Additional route quality metrics
                        col_stat4, col_stat5 = st.columns(2)
                        with col_stat4:
                            st.metric("Obstacle Pixels in Path", obstacles_in_path)
                        with col_stat5:
                            safety_score = 100 if obstacles_in_path == 0 else max(0, 100 - (obstacles_in_path/len(path))*100)
                            st.metric("Safety Score", f"{safety_score:.1f}%")
                        
                        # Store route data for GPS conversion
                        if use_gps:
                            st.session_state.route_path = path
                            st.session_state.image_shape = crowd_map.shape
                            
                    except Exception as e:
                        st.error(f"❌ Route computation failed: {e}")
                        st.info("Possible issues:")
                        st.info("• Start/end points may be in obstacle areas")
                        st.info("• No valid path exists between the points")
                        st.info("• Try selecting points in larger white areas")

with col2:
    st.subheader("📊 Route Analytics")
    
    if 'route_path' in st.session_state:
        path = st.session_state.route_path
        
        # Route analysis
        st.write("**Route Information:**")
        st.write(f"- Total waypoints: {len(path)}")
        st.write(f"- Start: ({path[0][1]}, {path[0][0]})")
        st.write(f"- End: ({path[-1][1]}, {path[-1][0]})")
        
        # Direction analysis
        if len(path) > 1:
            directions = []
            for i in range(1, len(path)):
                dy = path[i][0] - path[i-1][0]
                dx = path[i][1] - path[i-1][1]
                
                if dx > 0: directions.append("→")
                elif dx < 0: directions.append("←")
                if dy > 0: directions.append("↓")
                elif dy < 0: directions.append("↑")
            
            direction_counts = {d: directions.count(d) for d in set(directions)}
            st.write("**Movement Analysis:**")
            for direction, count in direction_counts.items():
                st.write(f"- {direction}: {count} steps")
    else:
        st.info("Upload a video and compute a route to see analytics")
    
    # GPS Integration
    if use_gps and 'route_path' in st.session_state:
        st.subheader("🌍 GPS Mapping")
        
        # Convert pixel coordinates to GPS coordinates
        path = st.session_state.route_path
        img_shape = st.session_state.image_shape
        
        # Simple linear mapping (in real application, you'd use proper coordinate transformation)
        lat_range = 0.01  # Approximate area coverage in degrees
        lon_range = 0.01
        
        gps_path = []
        for y, x in path:
            # Convert pixel to GPS coordinates
            lat = latitude + (y / img_shape[0] - 0.5) * lat_range
            lon = longitude + (x / img_shape[1] - 0.5) * lon_range
            gps_path.append([lat, lon])
        
        # Create folium map
        m = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
        
        # Add route to map
        if len(gps_path) > 1:
            folium.PolyLine(
                gps_path,
                color="green",
                weight=5,
                opacity=0.8,
                popup="Safe Route"
            ).add_to(m)
        
        # Add start and end markers
        if gps_path:
            folium.Marker(
                gps_path[0],
                popup="Start Point",
                icon=folium.Icon(color="blue", icon="play")
            ).add_to(m)
            
            folium.Marker(
                gps_path[-1],
                popup="End Point",
                icon=folium.Icon(color="red", icon="stop")
            ).add_to(m)
        
        # Display map
        st_folium(m, width=300, height=400)
        
        # GPS coordinates download
        if st.button("📥 Download GPS Route"):
            gps_data = "Latitude,Longitude\n"
            for lat, lon in gps_path:
                gps_data += f"{lat:.6f},{lon:.6f}\n"
            
            st.download_button(
                label="Download GPS Coordinates (CSV)",
                data=gps_data,
                file_name="route_gps_coordinates.csv",
                mime="text/csv"
            )

# Instructions and Help
st.markdown("---")
st.subheader("📖 How to Use")

col_help1, col_help2 = st.columns(2)

with col_help1:
    st.markdown("""
    **Step 1: Upload Video**
    - Upload an MP4, AVI, or MOV file
    - The system will process all frames
    - Automatically selects frame with most crowd detections
    
    **Step 2: Set Route Points**
    - Choose start coordinates (X, Y pixels)
    - Choose end coordinates (X, Y pixels)
    - Points should be in free (white) areas
    """)

with col_help2:
    st.markdown("""
    **Step 3: Compute Route**
    - Click "Compute Safe Route" button
    - Algorithm finds path avoiding crowded areas
    - Green line shows the optimal route
    
    **Step 4: GPS Integration (Optional)**
    - Enable GPS coordinates in sidebar
    - Set location and zoom level
    - View route on interactive map
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚦 Crowd-Aware Route Planner | Powered by YOLO & Pathfinding Algorithms</p>
    </div>
    """,
    unsafe_allow_html=True
)
