import streamlit as st
import cv2
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Vehicle & Pedestrian Detection",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Vehicle & Pedestrian Detection System")
st.markdown("Real-time object detection using HOG + SVM and Haar Cascades - Upload a video or image to detect vehicles and pedestrians")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
input_type = st.sidebar.radio("Input Type", ["Video", "Image"])
detection_method = st.sidebar.selectbox("Detection Method", ["HOG (Pedestrian)", "Haar Cascade (Vehicle)", "Both"])

# Detection classes
TARGET_CLASSES = ['person', 'car']

@st.cache_resource
def load_detectors():
    """Load HOG and Haar Cascade detectors"""
    try:
        # HOG Pedestrian Detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Haar Cascade for Cars
        car_cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
        car_cascade = cv2.CascadeClassifier(car_cascade_path)
        
        # If car cascade not found, try alternative
        if car_cascade.empty():
            # Use frontal face cascade as fallback (for demo)
            car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        return hog, car_cascade
    except Exception as e:
        st.error(f"Error loading detectors: {str(e)}")
        return None, None

def detect_pedestrians(image, hog, conf_threshold):
    """Detect pedestrians using HOG"""
    detections = []
    
    try:
        # Resize for faster processing
        scale = 1.0
        if image.shape[0] > 800:
            scale = 800.0 / image.shape[0]
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Detect people
        boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        for i, (x, y, w, h) in enumerate(boxes):
            if weights[i] > conf_threshold:
                # Scale back to original size
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                detections.append({
                    'class': 'person',
                    'confidence': float(weights[i]),
                    'bbox': (x, y, w, h)
                })
    except Exception as e:
        pass
    
    return detections

def detect_vehicles(image, cascade, conf_threshold):
    """Detect vehicles using Haar Cascade"""
    detections = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect cars
        cars = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in cars:
            # Haar cascades don't provide confidence, use fixed value
            detections.append({
                'class': 'car',
                'confidence': 0.8,
                'bbox': (x, y, w, h)
            })
    except Exception as e:
        pass
    
    return detections

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    colors = {
        'person': (0, 255, 0),  # Green
        'car': (255, 0, 0)       # Blue
    }
    
    for det in detections:
        x, y, w, h = det['bbox']
        cls = det['class']
        conf = det['confidence']
        
        color = colors.get(cls, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{cls}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - label_size[1] - 4), (x + label_size[0], y), color, -1)
        cv2.putText(image, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def create_analytics_charts(stats, frame_wise_data=None):
    """Create analytics visualizations"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for total detections
        if sum(stats.values()) > 0:
            df_stats = pd.DataFrame(list(stats.items()), columns=['Object', 'Count'])
            df_stats = df_stats[df_stats['Count'] > 0].sort_values('Count', ascending=False)
            
            fig = px.bar(df_stats, x='Object', y='Count', 
                        title='Total Detections by Object Type',
                        color='Count',
                        color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No objects detected")
    
    with col2:
        # Pie chart for distribution
        if sum(stats.values()) > 0:
            df_pie = pd.DataFrame(list(stats.items()), columns=['Object', 'Count'])
            df_pie = df_pie[df_pie['Count'] > 0]
            
            fig = px.pie(df_pie, values='Count', names='Object', 
                        title='Detection Distribution',
                        hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Frame-wise analysis for videos
    if frame_wise_data is not None and len(frame_wise_data) > 0:
        st.subheader("📈 Frame-wise Detection Analysis")
        
        df_frames = pd.DataFrame(frame_wise_data)
        
        fig = go.Figure()
        for obj_type in ['person', 'car']:
            if obj_type in df_frames.columns:
                fig.add_trace(go.Scatter(
                    x=df_frames['frame'],
                    y=df_frames[obj_type],
                    mode='lines',
                    name=obj_type.capitalize(),
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Objects Detected Over Time',
            xaxis_title='Frame Number',
            yaxis_title='Count',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_metrics(stats):
    """Display detection metrics in columns"""
    st.subheader("📊 Detection Summary")
    
    cols = st.columns(4)
    idx = 0
    
    for obj_type, count in stats.items():
        if count > 0:
            with cols[idx % 4]:
                st.metric(
                    label=obj_type.capitalize(),
                    value=count,
                    delta=None
                )
            idx += 1
    
    # Total detections
    total = sum(stats.values())
    st.markdown(f"### 🎯 Total Detections: **{total}**")

def process_image(image_path, hog, car_cascade, conf_threshold, method):
    """Process image and detect objects"""
    try:
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            st.error("Failed to read image file")
            return None, None, None
        
        detections = []
        
        # Detect based on method
        if method in ["HOG (Pedestrian)", "Both"]:
            ped_detections = detect_pedestrians(image.copy(), hog, conf_threshold)
            detections.extend(ped_detections)
        
        if method in ["Haar Cascade (Vehicle)", "Both"]:
            veh_detections = detect_vehicles(image.copy(), car_cascade, conf_threshold)
            detections.extend(veh_detections)
        
        # Draw detections
        annotated_image = draw_detections(image.copy(), detections)
        
        # Count detections
        stats = {'person': 0, 'car': 0}
        detection_details = []
        
        for det in detections:
            stats[det['class']] += 1
            x, y, w, h = det['bbox']
            detection_details.append({
                'Object': det['class'].capitalize(),
                'Confidence': f"{det['confidence']*100:.1f}%",
                'BBox': f"({x}, {y}, {w}, {h})"
            })
        
        return annotated_image, stats, detection_details
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None

def process_video(video_path, hog, car_cascade, conf_threshold, method):
    """Process video and detect objects"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Failed to open video file")
            return None, None, None, None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create temporary file for output
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.error("Failed to create output video file")
            cap.release()
            return None, None, None, None
        
        # Statistics
        stats = {'person': 0, 'car': 0}
        frame_wise_data = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_count = 0
        skip_frames = 2  # Process every 2nd frame for speed
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = []
            frame_stats = {'person': 0, 'car': 0}
            
            # Only process every nth frame
            if frame_count % skip_frames == 0:
                # Detect based on method
                if method in ["HOG (Pedestrian)", "Both"]:
                    ped_detections = detect_pedestrians(frame.copy(), hog, conf_threshold)
                    detections.extend(ped_detections)
                
                if method in ["Haar Cascade (Vehicle)", "Both"]:
                    veh_detections = detect_vehicles(frame.copy(), car_cascade, conf_threshold)
                    detections.extend(veh_detections)
                
                # Count detections
                for det in detections:
                    stats[det['class']] += 1
                    frame_stats[det['class']] += 1
            
            # Draw detections
            annotated_frame = draw_detections(frame.copy(), detections)
            
            # Store frame-wise data
            frame_data = {'frame': frame_count}
            frame_data.update(frame_stats)
            frame_wise_data.append(frame_data)
            
            out.write(annotated_frame)
            
            # Update progress
            frame_count += 1
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        # Video statistics
        duration = total_frames / fps if fps > 0 else 0
        avg_detections_per_frame = sum(stats.values()) / (total_frames / skip_frames) if total_frames > 0 else 0
        
        video_info = {
            'Total Frames': total_frames,
            'Frames Processed': total_frames // skip_frames,
            'Duration (seconds)': f"{duration:.2f}",
            'FPS': fps,
            'Avg Detections/Frame': f"{avg_detections_per_frame:.2f}"
        }
        
        return output_path, stats, frame_wise_data, video_info
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None, None, None

# Load detectors
with st.spinner("Loading detection models..."):
    hog, car_cascade = load_detectors()

if hog is None or car_cascade is None:
    st.error("Failed to load detectors. Please check OpenCV installation.")
    st.stop()

st.success("✅ Detection models loaded successfully!")

# Main app
if input_type == "Image":
    st.header("📷 Image Detection Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    with col2:
        st.info("📌 **Detection Methods**\n- **HOG**: Pedestrians\n- **Haar Cascade**: Vehicles\n- **Both**: All objects")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.close()
        
        st.success("✅ Image uploaded successfully!")
        
        # Show preview
        with st.expander("🖼️ Original Image", expanded=False):
            st.image(tfile.name, use_container_width=True)
        
        # Process button
        if st.button("🚀 Start Detection", type="primary"):
            with st.spinner("🔍 Processing image..."):
                result = process_image(tfile.name, hog, car_cascade, confidence_threshold, detection_method)
            
            if result[0] is not None:
                annotated_image, stats, detection_details = result
                
                st.success("✨ Detection complete!")
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🖼️ Detected Image")
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    display_metrics(stats)
                
                # Analytics
                st.markdown("---")
                create_analytics_charts(stats)
                
                # Detection details table
                if detection_details:
                    st.subheader("📋 Detection Details")
                    df_details = pd.DataFrame(detection_details)
                    st.dataframe(df_details, use_container_width=True, hide_index=True)
                
                # Download button
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                cv2.imwrite(output_path, annotated_image)
                
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Detected Image",
                        data=f,
                        file_name="detected_image.jpg",
                        mime="image/jpeg"
                    )
    else:
        st.info("👆 Please upload an image file to begin detection")

else:  # Video mode
    st.header("📹 Video Detection Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    with col2:
        st.info("📌 **Detection Methods**\n- **HOG**: Pedestrians\n- **Haar Cascade**: Vehicles\n- **Both**: All objects")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.close()
        
        st.success("✅ Video uploaded successfully!")
        
        # Show original video
        with st.expander("📹 Original Video", expanded=False):
            st.video(tfile.name)
        
        # Process button
        if st.button("🚀 Start Detection", type="primary"):
            st.info("🔍 Processing video... This may take a few minutes.")
            
            # Process video
            result = process_video(tfile.name, hog, car_cascade, confidence_threshold, detection_method)
            
            if result[0] is not None:
                output_path, stats, frame_wise_data, video_info = result
                
                st.success("✨ Detection complete!")
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🎬 Processed Video")
                    st.video(output_path)
                
                with col2:
                    display_metrics(stats)
                    
                    # Video info
                    st.subheader("ℹ️ Video Information")
                    for key, value in video_info.items():
                        st.text(f"{key}: {value}")
                
                # Analytics
                st.markdown("---")
                st.subheader("📊 Analytics Dashboard")
                create_analytics_charts(stats, frame_wise_data)
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f,
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )
    else:
        st.info("👆 Please upload a video file to begin detection")

# Instructions
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    ### Detection Methods:
    - **HOG (Histogram of Oriented Gradients)**: Best for pedestrian detection
    - **Haar Cascade**: Traditional method for vehicle detection
    - **Both**: Detect both pedestrians and vehicles
    
    ### For Images:
    1. Select **Image** mode in the sidebar
    2. Choose detection method
    3. Upload an image file (JPG, PNG, BMP)
    4. Click 'Start Detection'
    5. View detections, analytics, and download results
    
    ### For Videos:
    1. Select **Video** mode in the sidebar
    2. Choose detection method
    3. Upload a video file (MP4, AVI, MOV, MKV)
    4. Adjust confidence threshold (optional)
    5. Click 'Start Detection' to process
    6. View frame-by-frame analysis and download
    
    ### 💡 Tips:
    - HOG is more accurate for pedestrians but slower
    - Haar Cascade is faster but less accurate
    - Lower confidence threshold = more detections
    - Video processing skips frames for speed
    - Works without GPU (CPU only)
    
    ### ⚠️ Limitations:
    - Classical methods are less accurate than deep learning
    - Best results with clear, frontal views
    - May have false positives/negatives
    - Video processing is slower than YOLO
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, OpenCV (HOG + Haar Cascade) | 🚗 Classical Computer Vision Detection")
