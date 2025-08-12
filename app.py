from flask import Flask, render_template_string, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
import base64

# Inicializar Flask
app = Flask(__name__)

# Cargar modelo YOLOv8n (más liviano)
model = YOLO("yolov8n.pt")

# Página HTML de prueba
HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Detección YOLOv8n</title>
</head>
<body>
    <h1>Prueba de detección con YOLOv8n</h1>
    <form action="/detect" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Detectar</button>
    </form>
    {% if image_data %}
        <h2>Resultado:</h2>
        <img src="data:image/jpeg;base64,{{ image_data }}" alt="Resultado">
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No se envió imagen"}), 400

    file = request.files["image"]

    # Leer y reducir resolución automáticamente
    img = Image.open(file.stream).convert("RGB")
    max_size = (640, 640)  # tamaño máximo
    img.thumbnail(max_size, Image.LANCZOS)

    # Convertir a array
    img_array = np.array(img)

    # Detección con YOLOv8n
    results = model.predict(img_array)

    # Dibujar resultados
    annotated_img = results[0].plot()

    # Convertir a JPEG y luego a base64 para mostrar en HTML
    _, buffer = cv2.imencode(".jpg", annotated_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return render_template_string(HTML_PAGE, image_data=img_base64)

@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Endpoint API para detección que devuelve resultados en JSON"""
    if "image" not in request.files:
        return jsonify({"error": "No se envió imagen"}), 400

    file = request.files["image"]

    # Leer y reducir resolución
    img = Image.open(file.stream).convert("RGB")
    max_size = (640, 640)
    img.thumbnail(max_size, Image.LANCZOS)
    img_array = np.array(img)

    # Detección
    results = model.predict(img_array)
    detections = []
    for box in results[0].boxes:
        detections.append({
            "cls": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
