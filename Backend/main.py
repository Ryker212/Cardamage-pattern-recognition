from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app)

# โหลด YOLOv8 model
model = YOLO('C:/Users/ASUS/Desktop/Pr-Cardamage/Cardamage-pattern-recognition/modelver36.1/Cardamagebypart_ver36.1.pt')

# กำหนดฟอนต์
font_path = 'ARLRDBD.ttf'  # ตรวจสอบให้แน่ใจว่ามีฟอนต์นี้
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# ฟังก์ชันแปลงภาพจาก request เป็น numpy array
def read_image(file):
    img = Image.open(file)
    return img

@app.route('/detect', methods=['POST'])
def detect_damage():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # รับไฟล์รูปภาพ
        image_file = request.files['image']
        image = read_image(image_file)
        image_np = np.array(image)

        # ใช้ YOLOv8 ทำการตรวจจับ damage
        results = model(image_np)
        
        # วาดกรอบสี่เหลี่ยมบนภาพ
        draw = ImageDraw.Draw(image)

        # กำหนดสีสำหรับแต่ละ class
        class_colors = {
            'Front-lamp-Damage': 'red',
            'Rear-lamp-Damage': 'blue',
            'Sidemirror-Damage': 'green',
            'Windscreen-Damage': 'yellow',
            'Bonnet-Damage': 'purple',
            'Doorouter-Damage': 'orange',
            'Front-Bumper-Damage': 'pink',
            'Rear-Bumper-Damage': 'cyan'
        }
        
        detected_objects = []
        for result in results:
            for box in result.boxes:
                # ดึงค่าพิกัดของกล่อง
                box_coords = [float(coord) for coord in box.xyxy[0]]
                class_name = model.names[int(box.cls)]
                confidence = float(box.conf)

                # กำหนดสีจาก class_name
                color = class_colors.get(class_name, 'white')

                # วาดกรอบสี่เหลี่ยมบนภาพ
                draw.rectangle(box_coords, outline=color, width=3)

                # วาด label บนกล่อง
                draw.text((box_coords[0], box_coords[1] - 10), f"{class_name} ({confidence:.2f})", fill=color, font=font)

                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': box_coords
                })

        # ปรับขนาดภาพ
        image = image.resize((640, 640), Image.LANCZOS)

        # แปลงภาพที่มีกรอบเป็น base64
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)

        # แปลงเป็น base64 string
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        return jsonify({'detections': detected_objects, 'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
