import streamlit as st
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import warnings
import os
import soundfile as sf
import sounddevice as sd

# Suppress warnings
warnings.filterwarnings("ignore")

# Load YOLO model
yolo_model = YOLO('/Users/sam/Desktop/Master 02/Semester 01/AV/Project/Object Detection/yolov8m.pt')
# Set page config
st.set_page_config(page_title="Traffic Detection", page_icon="🚦", layout="wide")
# App Title and Description
st.title("AIOT Traffic Object Detection")
st.markdown("""
Welcome to **AIOT TOD**! This application allows you to:
- Detect traffic-related objects in images using our AI model.
- Customize parameters to raise alerts based on detection thresholds.
- Download detection results as CSV or Excel files.
""")

# Sidebar Parameters
st.sidebar.title("Parameters")
traffic_threshold = st.sidebar.slider(
    "Traffic Alert Threshold (Number of Objects Detected)", 
    min_value=1, 
    max_value=50, 
    value=10, 
    step=1
)

# Display object categories available to detect
object_categories = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'person', 'traffic light', 'stop sign']

# Allow user to select which objects to detect
selected_objects = st.sidebar.multiselect(
    'Select objects to detect', 
    object_categories, 
    default=object_categories
)

# Check if any objects are selected, otherwise show a warning
if not selected_objects:
    st.sidebar.warning("Please select at least one object to detect.")

bounding_box_color = st.sidebar.color_picker('Bounding Box Color', '#FF6347')  # Default is red

confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.05)


alert_sound = st.sidebar.checkbox("Play Alert Sound When Traffic is Detected", value=False)
save_processed_image = st.sidebar.checkbox("Save Processed Image", value=True)
show_text_annotations = st.sidebar.checkbox("Show Text Annotations", value=True)

# File uploader for single or multiple images
uploaded_files = st.file_uploader(
    "Upload Image(s)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    for idx, file in enumerate(uploaded_files):  # Add an index for each file
        
        # Add background color and title for each image section
        st.markdown(f"<h3>Processing Image {idx + 1}</h3></div>", unsafe_allow_html=True)
        
        # Load and process each image
        image = Image.open(file)
        image_np = np.array(image)

        # Perform object detection with the chosen confidence threshold
        results = yolo_model.predict(image_np, conf=confidence_threshold)
        detections = results[0]
        num_objects = len(detections.boxes)

        # Annotate image manually by drawing bounding boxes
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        detection_data = []
        detected_objects = []  # List to hold the detected objects for counting

        for box in detections.boxes:
            label = results[0].names[box.cls[0].item()]
            confidence = box.conf[0].item()
            coords = box.xyxy[0].tolist()

            if label in selected_objects and confidence >= confidence_threshold:
                # Draw the bounding box with the chosen color
                draw.rectangle(coords, outline=bounding_box_color, width=3)

                # Add label and confidence annotation if enabled
                if show_text_annotations:
                    text = f"{label} {confidence:.2f}"
                    draw.text((coords[0], coords[1]), text, font=font, fill="white")

                detection_data.append([label, confidence, coords])
                detected_objects.append(label)  # Track the detected objects for counting

        # Display the annotated image
        st.image(image, caption=f"Detected Objects: {len(detected_objects)}", use_container_width=True)
        # Add a colored background to separate results
        st.markdown(f"<h4>Results for Image {idx + 1}</h4></div>", unsafe_allow_html=True)
        # Check if traffic alert should be raised
        if len(detected_objects) >= traffic_threshold:
            st.warning("🚦 Traffic Detected! The number of detected objects exceeds the threshold.")
            
            if alert_sound:
                # Play sound alert (you can replace this with any sound file you prefer)
                alert_file = "/Users/sam/Desktop/Master 02/Semester 01/AV/Project/Object Detection/alert.mp3"  # Replace with actual file path
                if os.path.exists(alert_file):
                    data, fs = sf.read(alert_file)
                    sd.play(data, fs)

        # Create a dataframe for detected objects
        df = pd.DataFrame(detection_data, columns=["Object", "Confidence", "Coordinates"])

        # Display DataFrame
        st.dataframe(df)



        # Add download options for CSV and Excel in the same line using columns
        col1, col2, col3= st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"detections_{idx}.csv",  # Add index to make it unique
                mime="text/csv",
                key=f"csv_button_{idx}"  # Add index to make the key unique
            )
        
        with col2:
            excel = BytesIO()
            df.to_excel(excel, index=False)
            excel.seek(0)
            st.download_button(
                label="Download Excel",
                data=excel,
                file_name=f"detections_{idx}.xlsx",  # Add index to make it unique
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_button_{idx}"  # Add index to make the key unique
            )
        with col3: 
            # Save processed image if enabled
            if save_processed_image:
                processed_image_path = f"processed_image_{idx}.jpg"  # Add index to make it unique
                image.save(processed_image_path)
                with open(processed_image_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Image",
                        data=file,
                        file_name=f"processed_image_{idx}.jpg",  # Add index to make it unique
                        mime="image/jpeg",
                        key=f"image_button_{idx}"  # Add index to make the key unique
                    )
        # Elegant separation using a thin line and spacing
        st.markdown("""
        <hr style="border: 1px solid #ddd; margin-top: 20px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)

        
