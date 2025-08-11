import os
import json
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO

app = Flask(__name__)

# Usamos /tmp para evitar conflictos en Render
RECEIVED_FOLDER = '/tmp/received_images'
RESULT_FOLDER = '/tmp/result_images'

os.makedirs(RECEIVED_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('yolov5s.pt')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró el archivo 'image'"}), 400

    file = request.files['image']
    filename = file.filename or 'imagen.jpg'
    save_path = os.path.join(RECEIVED_FOLDER, filename)
    file.save(save_path)

    try:
        results = model(save_path)
        result_img_path = os.path.join(RESULT_FOLDER, filename)
        results[0].save(result_img_path)

        detections = []
        for box in results[0].boxes:
            bbox = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())
            detections.append({
                'bbox': bbox,
                'confidence': confidence,
                'class': cls
            })

        json_path = os.path.join(RESULT_FOLDER, os.path.splitext(filename)[0] + '.json')
        with open(json_path, 'w') as f:
            json.dump(detections, f, indent=4)

        return jsonify({
            "message": "Imagen procesada correctamente",
            "detections": detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
