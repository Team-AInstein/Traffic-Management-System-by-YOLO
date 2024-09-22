import cv2
import os
import torch
import pytesseract
import numpy as np

# Set path to Tesseract executable (for Windows users)
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"  # Update this to the correct path if you're on Windows

# YOLOv5 Model Loading
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Switch to 'yolov5m' for better accuracy
model.conf = 0.25  # Lower the detection confidence threshold

# Define the path to the image (replace with the direct path)
image_path = "C:/Users/naeem/VS Code Programs/Python_Projects/SIH/image3.jpg"  # Update with your file path

# Function to process the image for traffic signal control
def process_traffic_image(image_path):
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The image path '{image_path}' does not exist.")
        return

    image = cv2.imread(image_path)
    
    # Error handling if image loading fails
    if image is None:
        print("Error: Unable to load image. Please check the file path.")
        return

    print("Image loaded successfully!")
    
    # YOLOv5 inference
    print("Performing YOLOv5 inference on the image...")
    results = model(image)
    
    # Get the pandas dataframe of detected objects
    detected_objects = results.pandas().xyxy[0]
    
    # Print out the detected objects for debugging
    print("Detected objects by YOLOv5:")
    print(detected_objects[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']])
    
    # Perform emergency vehicle detection using OCR with enhanced logic
    emergency_detected = detect_emergency_vehicle_with_ocr(detected_objects, image)
    
    # Override YOLO label if emergency vehicle is detected via OCR
    detected_objects = override_yolo_label_with_ocr(detected_objects, emergency_detected)
    
    # Filter the DataFrame to count vehicles (this includes common vehicle types)
    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'emergency vehicle']  # Added 'emergency vehicle' to vehicle classes
    vehicle_count = detected_objects[detected_objects['name'].isin(vehicle_classes)].shape[0]

    print(f"Number of vehicles detected: {vehicle_count}")
    print(f"Emergency vehicle detected: {emergency_detected[0]}")
    
    # Render the results with custom bounding box for emergency vehicle
    print("Inference complete. Rendering results with custom labels...")
    render_custom_results(image, detected_objects, emergency_detected)
    
    # Control traffic signal based on detected parameters
    signal = control_traffic_signal(vehicle_count, emergency_detected[0])
    print(f"Traffic Signal: {signal}")

# Function to detect emergency vehicles using OCR with better preprocessing
def detect_emergency_vehicle_with_ocr(detected_objects, image):
    emergency_keywords = ['AMBULANCE', 'ECNALUBMA', 'POLICE', 'FIRE']  # Added reverse "AMBULANCE" for detection
    for index, row in detected_objects.iterrows():
        # Crop the bounding box area
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        vehicle_crop = image[ymin:ymax, xmin:xmax]
        
        # Preprocess the cropped image for better OCR performance
        vehicle_crop_gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # Improved preprocessing: Gaussian blur and adaptive threshold
        vehicle_crop_blur = cv2.GaussianBlur(vehicle_crop_gray, (5, 5), 0)
        vehicle_crop_thresh = cv2.adaptiveThreshold(vehicle_crop_blur, 255, 
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 11, 2)
        
        # Apply OCR to read text in the preprocessed cropped image
        ocr_text = pytesseract.image_to_string(vehicle_crop_thresh, config='--psm 6')  # PSM 6 is used for text block detection

        # Debugging OCR output
        print(f"OCR result for the vehicle: {ocr_text}")
        
        # Check if any emergency keyword or reverse keyword is found in the OCR result
        for keyword in emergency_keywords:
            if keyword in ocr_text.upper():
                print(f"Emergency vehicle detected: {keyword} found in OCR result")
                return (True, (xmin, ymin, xmax, ymax))  # Returning bounding box for later use
    
    return (False, None)  # No emergency vehicle detected

# Function to override YOLOv5 class label based on OCR detection
def override_yolo_label_with_ocr(detected_objects, emergency_detected):
    emergency_found, emergency_bbox = emergency_detected
    if emergency_found:
        xmin_em, ymin_em, xmax_em, ymax_em = emergency_bbox
        for index, row in detected_objects.iterrows():
            # Get YOLO bounding box
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Check for significant overlap using the intersection over union (IoU) method
            iou_value = calculate_iou((xmin, ymin, xmax, ymax), (xmin_em, ymin_em, xmax_em, ymax_em))
            print(f"Calculated IoU for potential overlap: {iou_value}")
            
            # If IoU is higher than 0.5, assume the bounding boxes match and update label
            if iou_value > 0.5:
                print(f"Overriding detected class with 'emergency vehicle' for bounding box at {xmin}, {ymin}, {xmax}, {ymax}")
                detected_objects.loc[index, 'name'] = 'emergency vehicle'  # Override label in the DataFrame
    
    return detected_objects

# Function to calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(bbox1, bbox2):
    # bbox1 and bbox2 are tuples: (xmin, ymin, xmax, ymax)
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Determine the coordinates of the intersection rectangle
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    # Compute the area of intersection
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # Compute the area of both bounding boxes
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Compute the union area
    union_area = bbox1_area + bbox2_area - inter_area
    
    # Compute IoU
    iou = inter_area / union_area if union_area != 0 else 0
    
    return iou

# Function to render custom results with red bounding box for emergency vehicle
def render_custom_results(image, detected_objects, emergency_detected):
    for index, row in detected_objects.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']

        # Set box color (Red for emergency vehicles, green for others)
        if label == 'emergency vehicle':
            box_color = (0, 0, 255)  # Red for emergency vehicle
        else:
            box_color = (0, 255, 0)  # Green for other vehicles

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, 2)
        
        # Put the label and confidence on the image
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    # Display the updated image
    cv2.imshow("Detected Objects with Custom Labels", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to control traffic signal based on detected vehicles and emergency status
def control_traffic_signal(vehicle_count, emergency_detected):
    # Traffic signal control based on emergency vehicle detection and traffic density
    if emergency_detected:
        return "Green for emergency vehicle"
    elif vehicle_count > 5:  # Example logic: consider high traffic density if > 5 vehicles
        return "Green for longer duration"
    else:
        return "Standard signal"

# Process the image
process_traffic_image(image_path)
