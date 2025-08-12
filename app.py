from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import numpy as np
import base64
import cv2

app = Flask(__name__)

# Cargar modelo YOLOv8n (liviano)
model = YOLO('yolov8n.pt')

HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>YOLOv8n Smart Parking</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; }
h1 { color: #2c3e50; }
form { margin-top: 20px; }
input[type=file] { margin-bottom: 10px; }
.resultado { margin-top: 20px; padding: 10px; background: #f1f1f1; border-radius: 8px; }
</style>
</head>
<body>
<h1>Subir imagen para detección</h1>
<form id="uploadForm" enctype="multipart/form-data" method="post" action="/upload">
<input type="file" name="image" accept="image/*" required><br>
<button type="submit">Enviar</button>
</form>
<div class="resultado" id="resultado">
{% if image_data %}
<img src="data:image/jpeg;base64,{{ image_data }}" style="max-width:400px;margin-top:10px;" />
<pre>{{ detections }}</pre>
{% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_PAGE)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró el archivo 'image'"}), 400
    file = request.files['image']

    # Abrir imagen y reducir resolución a 320x320 máximo
    img = Image.open(file.stream).convert("RGB")
    img.thumbnail((320, 320), Image.LANCZOS)

    # Convertir a numpy array para YOLO
    img_array = np.array(img)

    # Detección
    results = model(img_array)

    # Imagen con detecciones dibujadas
    annotated = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated)
    image_data = base64.b64encode(buffer).decode('utf-8')

    # Extraer detecciones
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': int(box.cls),
            'confidence': float(box.conf),
            'bbox': box.xyxy[0].tolist()
        })

    return render_template_string(HTML_PAGE, image_data=image_data, detections=detections)

@app.route('/upload', methods=['GET'])
def upload_get():
    # Redirige GET a raíz para evitar error Method Not Allowed
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
