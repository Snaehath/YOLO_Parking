from flask import Flask, render_template, request,jsonify,flash
from ultralytics import YOLO
import cv2
import os
from paddleocr import PaddleOCR
import numpy as np

# Initialize PaddleOCR with English as the language
ocr = PaddleOCR(use_angle_cls=True, lang='en')

app = Flask(__name__)

# Load the YOLO model for object detection
detection_model = YOLO('./best_latest.pt')
classification_model = YOLO('./best_cc.pt')

UPLOADS_DIR = 'uploads'
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

def get_number_plate_color(image_path):
    image = cv2.imread(image_path)
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the region of interest (ROI) covering the entire image
    roi = hsv
    
    # Calculate the average color of the ROI
    avg_color = np.mean(roi, axis=(0, 1))
    
    # Check if the average color tends more towards green or white
    green_threshold = 100  # Tune this threshold based on your images
    if avg_color[1] > green_threshold:
        return "Green"
    else:
        return "White"

@app.route('/')
def index():
 
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
 

    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Save the uploaded file
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    file.save(file_path)

    # Perform object detection using YOLO
    classifcation_results = classification_model(file_path)
    prob1=classifcation_results[0].probs
    class_mapping = {0: 'Two-Wheeler', 1: 'Four-Wheeler'}
    vehicle = prob1.top5[0]
    vehicle_label = class_mapping[vehicle]

    results = detection_model(source=file_path, project='uploads', save_crop=True, exist_ok=True, name='pred')
    cropped_file_path = os.path.join('uploads', 'pred', 'crops', 'Number Plate', os.path.basename(file_path))

    color = get_number_plate_color(cropped_file_path)
    if color == "Green":
        vehicle_label = "E-" + vehicle_label
    img = cv2.imread(cropped_file_path)

    result = ocr.ocr(img, cls=True)
    # Draw detected text on the image
    text_output = ""
    for line in result:
        for word in line:
            text_output += word[1][0]

    text_output = text_output.upper().replace("-", "").replace(" ", "").replace(".", "").replace(":","").replace("$","S")
    if text_output and text_output[0].isdigit():
        text_output = text_output[1:]
    text_output = text_output.replace("IND", "").replace("INO", "")
    text_output = text_output[:2] + text_output[2:4].replace("O", "0") + text_output[4:]
    text_output = text_output[:4] + text_output[4:6].replace("8", "B") + text_output[6:]
    text_output = text_output[:4] + text_output[4:6].replace("0", "D") + text_output[6:]

    while text_output and text_output[-1].isalpha():
        text_output = text_output[:-1]
   
    text_output = text_output[:10]

    # Delete the uploaded file
    os.remove(file_path)
    os.remove(cropped_file_path)
    
    return render_template('index.html',classification_output=vehicle_label, text_output=text_output)
    # return jsonify({"VechicleType" :vehicle_label, "VehicleNo":text_output})

if __name__ == '__main__':
    app.run(debug=False,port=5050)
