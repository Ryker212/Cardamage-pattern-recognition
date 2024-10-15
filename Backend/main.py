import random
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # ไม่ใช้ GPU


app = Flask(__name__)
CORS(app)

# โหลด YOLOv8 model
model = YOLO(r'/mnt/c/Users/Best_ryker/Desktop/Work/Year4term1/pattern-recognition/Cardamage(github)/Cardamage-pattern-recognition/modelver42.1.50/Cardamagebypart_ver42.1.50.pt')

model_level = load_model(r'/mnt/c/Users/Best_ryker/model_level/my_modelver3.100.h5')

# Load font
font_path = 'ARLRDBD.ttf'
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# Convert image from request to numpy array
def read_image(file):
    img = Image.open(file)
    return img

# Convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/test', methods=['GET'])
def test():
    print("test")
    return jsonify({"message": "Test successful!"})

@app.route('/detect', methods=['POST'])
def detect_damage():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Retrieve uploaded image files
        image_files = request.files.getlist('image')

        all_results = []  # Store all results

        # Process each file
        for image_file in image_files:
            image = read_image(image_file)
            #image_np = np.array(image)

            # Use YOLOv8 to detect damage
            results = model(image, conf=0.5, iou=0.5,imgsz=640)

            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(image)

            # Define class colors
            class_colors = {
                'Front-lamp-Damage': 'red',
                'Rear-lamp-Damage': 'blue',
                'Sidemirror-Damage': 'green',
                'Windscreen-Damage': 'yellow',
                'bonnet-damage': 'purple',
                'doorouter-damage': 'orange',
                'front-bumper-damage': 'pink',
                'rear-bumper-damage': 'cyan'
            }

            detected_objects = []
            for result in results:
                for box in result.boxes:
                    # Get box coordinates
                    box_coords = [float(coord) for coord in box.xyxy[0]]
                    class_name = model.names[int(box.cls)]
                    confidence = float(box.conf)

                    # Crop the damaged part
                    x1, y1, x2, y2 = map(int, box_coords)
                    cropped_image = image.crop((x1, y1, x2, y2)).resize((224, 224), Image.LANCZOS)

                    # Draw bounding box and label
                    color = class_colors.get(class_name, 'white')
                    draw.rectangle(box_coords, outline=color, width=3)
                    draw.text((box_coords[0], box_coords[1] - 10), f"{class_name} ({confidence:.2f})", fill=color, font=font)

                    # Add detected object details
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': box_coords,
                        'cropped_image': image_to_base64(cropped_image)  # Convert cropped image to base64
                    })

            # Severity prediction
            severity_levels = ['Minor Damage', 'Moderate Damage', 'Severe Damage']
            severity_results = []
            for obj in detected_objects:
                cropped_image_np = np.array(cropped_image) / 255.0  # Normalize pixels
                cropped_image_np = np.expand_dims(cropped_image_np, axis=0)
                severity_prediction = model_level.predict(cropped_image_np)
                severity_level = severity_levels[np.argmax(severity_prediction)]

                severity_results.append({
                    'class': obj['class'],
                    'confidence': obj['confidence'],
                    'box': obj['box'],
                    'severity': severity_level,
                    'cropped_image': obj['cropped_image']
                })

            # Resize image for output
            image = image.resize((640, 640), Image.LANCZOS)

            # Convert annotated image to base64
            img_io = io.BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Store results
            all_results.append({'detections': severity_results, 'image': img_base64})

        # Return all results
        return jsonify({'results': all_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)