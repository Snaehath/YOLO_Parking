from fastapi import FastAPI,File,UploadFile
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR
import cv2
import os

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.get("/")
async def read_root():
    return {"message":"Hello world!"}

classification_model = YOLO('./best_cc.pt')
detection_model = YOLO('./best_latest.pt')

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
    
def vechile_class(image):
 classification = classification_model(image)
 prob1=classification[0].probs
 class_mapping = {0: 'Two-Wheeler', 1: 'Four-Wheeler'}
 vehicle = prob1.top5[0]
 vehicle_label = class_mapping[vehicle]

def ocr_extract(file_path):
 img = cv2.imread(file_path)
 grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 inverted_img = cv2.bitwise_not(img)

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

@app.post("/detect/")
async def detect_objects(file: UploadFile):
 # Process the uploaded image for object detection
 image_bytes = await file.read()
 image = np.frombuffer(image_bytes, dtype=np.uint8)
 image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
 classification_result = vechile_class(image)

 results = detection_model(source=image, project='uploads', save_crop=True, exist_ok=True, name='pred')

 cropped_file_path = os.path.join('uploads', 'pred', 'crops', 'Number Plate', 'image0.jpg')
 color = get_number_plate_color(cropped_file_path)
 if color == "Green":
    classification_result = "E-" + classification_result

 ocr_result = ocr_extract(cropped_file_path)
 os.remove(cropped_file_path)
 
 return {"VehicleType": classification_result,"VehicleNo":ocr_result}


# uvicorn fast_app:app
# curl http://127.0.0.1:8000/
# curl -X POST -F "file=@IMG_20240408_110950.jpg" http://127.0.0.1:8000/detect/