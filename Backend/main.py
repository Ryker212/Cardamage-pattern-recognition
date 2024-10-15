import random
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app)
model_level = load_model(r'C:\Users\ASUS\Desktop\model levelcar\my_modelver3.100.h5')
# โหลด YOLOv8 model
model = YOLO(r'C:\Users\ASUS\Desktop\Pr-Cardamage\Cardamage-pattern-recognition\modelver40.1.50\Cardamagebypart_ver40.1.50.pt')
# กำหนดฟอนต์
font_path = r'C:\Users\ASUS\Desktop\Pr-Cardamage\Cardamage-pattern-recognition\Backend\ARLRDBD.TTF'  # ตรวจสอบให้แน่ใจว่ามีฟอนต์นี้
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# ฟังก์ชันแปลงภาพจาก request เป็น numpy array
def read_image(file):
    img = Image.open(file)
    return img
# ฟังก์ชันแปลงภาพเป็น base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/test', methods=['GET'])
def test():
    print("test")


@app.route('/detect', methods=['POST'])
def detect_damage():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # รับไฟล์รูปภาพ
        image_file = request.files['image']
        image = read_image(image_file)
        #image = image.resize((640, 640), Image.LANCZOS)
        image_np = np.array(image) # แปลงค่าให้เป็น 0-1 
        # ใช้ YOLOv8 ทำการตรวจจับ damage
        results = model(image, conf=0.5, iou=0.5)
        
        # วาดกรอบสี่เหลี่ยมบนภาพ
        draw = ImageDraw.Draw(image)

        # กำหนดสีสำหรับแต่ละ class
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
                # ดึงค่าพิกัดของกล่อง
                box_coords = [float(coord) for coord in box.xyxy[0]]
                class_name = model.names[int(box.cls)]
                confidence = float(box.conf)

                # ตัดภาพส่วนที่เสียหายออกมาก่อนวาดกรอบ
                x1, y1, x2, y2 = map(int, box_coords)
                cropped_image = image.crop((x1, y1, x2, y2))
                cropped_image = cropped_image.resize((224, 224), Image.LANCZOS)

                # กำหนดสีจาก class_name
                color = class_colors.get(class_name, 'white')

                # วาดกรอบสี่เหลี่ยมบนภาพ
                draw.rectangle(box_coords, outline=color, width=3)

                # วาด label บนกล่อง
                draw.text((box_coords[0], box_coords[1] - 10), f"{class_name} ({confidence:.2f})", fill=color, font=font)

                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': box_coords,
                    'cropped_image': image_to_base64(cropped_image)  # เพิ่มภาพที่ถูกตัดออกมา
                })
        #โมเดลระดบความเสียหาย
        # severity_results = []
        # for obj in detected_objects:
        #     # ตัดภาพส่วนที่เสียหายออกมา
        #     x1, y1, x2, y2 = map(int, obj['box'])
        #     cropped_image = image.crop((x1, y1, x2, y2))
        #     cropped_image_np = np.array(cropped_image)

        #     # ใช้โมเดลที่สองในการประเมินระดับความเสียหาย
        #     severity_result = severity_model(cropped_image_np)#

        #     severity_results.append({
        #         'class': obj['class'],
        #         'confidence': obj['confidence'],
        #         'box': obj['box'],
        #         'severity': severity_level
        #     })
        # Fake
        severity_levels = ['Minor Dam', 'Moderate Dam', 'Severe Dam']
        severity_results = []
        for obj in detected_objects:
            # ตัดภาพส่วนที่เสียหายออกมา
            x1, y1, x2, y2 = map(int, obj['box'])
            cropped_image = image.crop((x1, y1, x2, y2))
        
            # ปรับขนาดภาพให้ตรงตามที่โมเดลคาดหวัง
            cropped_image = cropped_image.resize((224, 224), Image.LANCZOS)
            cropped_image_np = np.array(cropped_image) / 255.0  # ปรับขนาดค่าพิกเซลให้เป็น 0-1

            cropped_image_np = np.expand_dims(cropped_image_np, axis=0)
            severity_prediction = model_level.predict(cropped_image_np)
            severity_level = severity_levels[np.argmax(severity_prediction)]  # แปลงผลลัพธ์เป็นระดับความเสียหาย
            #severity_level = random.choice(severity_levels)  # สุ่มคลาสความเสียหาย
            #x1, y1, x2, y2 = map(int, obj['box'])
            #cropped_image = image.crop((x1, y1, x2, y2))
            
            cropped_image = cropped_image.resize((224, 224), Image.LANCZOS)
            severity_results.append({
                'class': obj['class'],
                'confidence': obj['confidence'],
                'box': obj['box'],
                'severity': severity_level,
                'cropped_image': obj['cropped_image']  # ใช้ภาพที่ถูกตัดออกมาโดยตรง
            })
            


        # ปรับขนาดภาพ
        image = image.resize((640, 640), Image.LANCZOS)

        # แปลงภาพที่มีกรอบเป็น base64
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)

        # แปลงเป็น base64 string
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        #return jsonify({'detections': detected_objects, 'image': img_base64})
        return jsonify({'detections': severity_results, 'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
