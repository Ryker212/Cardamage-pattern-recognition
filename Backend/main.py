from flask import Flask,request,jsonify,send_file,make_response
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, create_access_token, get_jwt
from datetime import timedelta
import os
from flask_cors import CORS
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from datetime import datetime

app = Flask(__name__)
CORS(app)  # เปิดใช้งาน CORS



######################################################################
#API base
@app.route('/')
def index():
    return ""
######################################################################

# API detection
# ส่งรูปจากกล้องมา
@app.route('/request_pic', methods=['POST'])
@jwt_required()
def handle_request_video():
    # Connect to the database
    mydb = mysql.connector.connect(host=host, user=user, password=password, database=database)
    mycursor = mydb.cursor(dictionary=True)
        
    # Get current user from JWT
    current_user = get_jwt_identity()
        
    # Check if the user exists
    sql = "SELECT * FROM user WHERE name = %s"
    val = (current_user,)
    mycursor.execute(sql, val)
    user_result = mycursor.fetchone()
        
    # If user does not exist, return an error response
    if not user_result:
        return make_response(jsonify({"msg": "Token is bad"}), 404)
    # แปลง base64 กลับมาเป็นภาพ
    data = request.get_json()
    datas = data['imageData']
    header, encoded = datas.split(",", 1)
    img_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(img_data))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if frame is not None:
        frames, num,filename = camera.get_pic(frame)
        print(num)
        sum = num[0]+num[1]+num[2]+num[3]+num[4]#+num[5]
        false = num[1]
        if(sum != 0):
            
            percent = (false/sum) *100
            x = float("{:.2f}".format(percent))
            # print(x)
        else:
            x = 0
        response_data = {
        "num": num,      # ข้อมูลตัวเลข
        "filename": filename,  # ชื่อไฟล์
        "percent": x
        
        }
        if frames is not None: 
            print(num)
            return jsonify(response_data)
        else:   
            return "Error: Failed to grab frame from camera"
    else :
        print("ssss")

# ลบรูปที่ถ่าย
@app.route('/delete_capture', methods=['POST'])
@jwt_required()
def delete_capture():
    # Connect to the database
    mydb = mysql.connector.connect(host=host, user=user, password=password, database=database)
    mycursor = mydb.cursor(dictionary=True)
        
    # Get current user from JWT
    current_user = get_jwt_identity()
        
    # Check if the user exists
    sql = "SELECT * FROM user WHERE name = %s"
    val = (current_user,)
    mycursor.execute(sql, val)
    user_result = mycursor.fetchone()
        
    # If user does not exist, return an error response
    if not user_result:
        return make_response(jsonify({"msg": "Token is bad"}), 404)
    data = request.get_json()
    filename = data['filename']
    if os.path.exists(filename):
        os.remove(filename)
        return jsonify({'status': 'Delete', 'data_received': data}), 200
    else:
        return jsonify({'error': 'No data provided'}), 400


# ดึงรูปที่ถ่าย
@app.route('/image', methods=['GET'])
def image():
    filename = request.args.get('filename')
    print("aaaaa=" + filename)
    file_path = os.path.join(app.root_path, filename) 
    if os.path.exists(file_path):  # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
        return send_file(file_path, mimetype='image/jpeg')
    else:
        return "File not found", 404  # ส่งโค้ด 404 หากไม่พบไฟล์
######################################################################



######################################################################
# Main
if __name__ == '__main__':
    app.run()
######################################################################