import os
import json
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# Carpetas temporales en Render
RECEIVED_FOLDER = '/tmp/received_images'
RESULT_FOLDER = '/tmp/result_images'

os.makedirs(RECEIVED_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Cargar modelo YOLO
model = YOLO('yolov5s.pt')

# Página web para prueba desde navegador
@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Prueba YOLO Smart Parking</title>
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
        <form id="uploadForm">
            <input type="file" name="image" accept="image/*" required><br>
            <button type="submit">Enviar</button>
        </form>
        <div class="resultado" id="resultado"></div>

        <script>
        document.getElementById("uploadForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            document.getElementById("resultado").innerHTML = "Procesando...";
            try {
                const res = await fetch("/upload", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                let html = "<pre>" + JSON.stringify(data, null, 2) + "</pre>";
                if (data.image_url) {
                    html += `<img src="${data.image_url}" style="max-width:400px;margin-top:10px;">`;
                }
                document.getElementById("resultado").innerHTML = html;
            } catch (err) {
                document.getElementById("resultado").innerHTML = "Error: " + err;
            }
        });
        </script>
    </body>
    </html>
    """)

# Servir imágenes procesadas
@app.route('/result_images/<filename>')
def get_result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# Endpoint API para detección
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
            cls = int(box.cls[0].item())
            if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                detections.append({
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': box.conf[0].item(),
                    'class': cls
                })

        ocupado = 1 if len(detections) > 0 else 0

        return jsonify({
            "ocupado": ocupado,
            "image_url": f"/result_images/{filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
