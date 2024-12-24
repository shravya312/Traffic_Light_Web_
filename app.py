from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io

# Initialize Flask app
app = Flask(__name__)

# Set the path to your saved model checkpoint
MODEL_PATH = 'traffic_light_detection/models/fasterrcnn_resnet50_fpn.pth'  # Change this to your model's actual path

# Initialize the Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Initialize with no weights
model.load_state_dict(torch.load(MODEL_PATH))  # Load the saved model weights
model.eval()  # Set the model to evaluation mode

# Set the device for model inference (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transform to convert the image into a tensor
transform = transforms.Compose([transforms.ToTensor()])

# COCO class labels (You can modify this based on your custom dataset)
# 10 corresponds to "traffic light" in COCO dataset
COCO_CLASSES = {
    0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 
    5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 
    10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter", 
    14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 
    20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 
    25: "orange", 26: "apple", 27: "banana", 28: "pineapple", 29: "grapes", 
    30: "watermelon", 31: "hot dog", 32: "pizza", 33: "donut", 34: "cake", 
    35: "chair", 36: "couch", 37: "potted plant", 38: "bed", 39: "dining table", 
    40: "toilet", 41: "tv", 42: "laptop", 43: "mouse", 44: "remote", 
    45: "keyboard", 46: "cell phone", 47: "microwave", 48: "oven", 
    49: "toaster", 50: "sink", 51: "refrigerator", 52: "book", 53: "clock", 
    54: "vase", 55: "scissors", 56: "teddy bear", 57: "hair drier", 
    58: "toothbrush"
}

# Function to detect traffic light color
def detect_traffic_light_color(cropped_img):
    cropped_img = np.array(cropped_img)
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)

    # Color ranges for traffic lights
    green_range = ((35, 50, 50), (85, 255, 255))
    red_range_1 = ((0, 100, 100), (10, 255, 255))
    red_range_2 = ((170, 100, 100), (180, 255, 255))
    yellow_range = ((20, 100, 100), (30, 255, 255))

    green_mask = cv2.inRange(hsv_img, green_range[0], green_range[1])
    red_mask_1 = cv2.inRange(hsv_img, red_range_1[0], red_range_1[1])
    red_mask_2 = cv2.inRange(hsv_img, red_range_2[0], red_range_2[1])
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    yellow_mask = cv2.inRange(hsv_img, yellow_range[0], yellow_range[1])

    green_percentage = np.sum(green_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
    red_percentage = np.sum(red_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
    yellow_percentage = np.sum(yellow_mask) / (cropped_img.shape[0] * cropped_img.shape[1])

    if green_percentage > red_percentage and green_percentage > yellow_percentage:
        return "Green"
    elif red_percentage > yellow_percentage:
        return "Red"
    else:
        return "Yellow"

@app.route('/')
def serve_frontend():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_traffic_lights():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(io.BytesIO(file.read()))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Set the traffic light class label (10 in COCO dataset)
    traffic_light_class_label = 10

    # Filter out non-traffic light detections based on class label
    threshold = 0.5  # Minimum score to consider as valid detection
    traffic_light_boxes = boxes[scores > threshold]
    traffic_light_labels = labels[scores > threshold]
    traffic_light_scores = scores[scores > threshold]

    # Keep only traffic light class detections
    traffic_light_boxes = traffic_light_boxes[traffic_light_labels == traffic_light_class_label]
    traffic_light_scores = traffic_light_scores[traffic_light_labels == traffic_light_class_label]

    detection_results = []
    for box, score in zip(traffic_light_boxes, traffic_light_scores):
        cropped_img = image.crop((box[0], box[1], box[2], box[3]))  # Crop the region of interest
        light_color = detect_traffic_light_color(cropped_img)
        detection_results.append({
            "box": box.tolist(),
            "score": float(score),
            "light_color": light_color
        })

    return jsonify(detection_results)

if __name__ == '__main__':
    app.run(debug=True)