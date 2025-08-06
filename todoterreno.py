import os
import json
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch


app = Flask(__name__)


RECEIVED_FOLDER = 'received_images'
RESULT_FOLDER = 'result_images'


os.makedirs(RECEIVED_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No se encontró el archivo 'image' en la petición.", 400


    file = request.files['image']
    filename = file.filename or 'imagen_recibida.jpg'
    save_path = os.path.join(RECEIVED_FOLDER, filename)
    file.save(save_path)
    print(f"Imagen recibida y guardada en: {save_path}")


    try:
        img = Image.open(save_path)


        results = model(img)
        results.render()


        if hasattr(results, 'imgs'):
            img_array = results.imgs[0]
        elif hasattr(results, 'ims'):
            img_array = results.ims[0]
        elif hasattr(results, 'rendered'):
            img_array = results.rendered[0]
        else:
            raise AttributeError("No se pudo encontrar la imagen renderizada en 'results'")


        result_img = Image.fromarray(img_array)


        result_img_path = os.path.join(RESULT_FOLDER, filename)
        result_img.save(result_img_path)
        print(f"Imagen con detección guardada en: {result_img_path}")


        json_path = os.path.join(RESULT_FOLDER, os.path.splitext(filename)[0] + '.json')
        detections = results.pandas().xyxy[0].to_dict(orient='records')
        with open(json_path, 'w') as f:
            json.dump(detections, f, indent=4)
        print(f"JSON con detecciones guardado en: {json_path}")


        return "Imagen y detección guardadas correctamente", 200


    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return f"Error interno: {e}", 500


# --- ENDPOINTS GET PARA CONSULTAS ---


# Listar archivos JSON disponibles
@app.route('/results/json', methods=['GET'])
def listar_json():
    archivos = [f for f in os.listdir(RESULT_FOLDER) if f.endswith('.json')]
    return jsonify(archivos)


# Obtener contenido de JSON específico
@app.route('/results/json/<filename>', methods=['GET'])
def obtener_json(filename):
    ruta = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(ruta):
        return "Archivo JSON no encontrado", 404
    with open(ruta, 'r') as f:
        data = json.load(f)
    return jsonify(data)


# Listar imágenes procesadas disponibles
@app.route('/results/images', methods=['GET'])
def listar_imagenes():
    archivos = [f for f in os.listdir(RESULT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return jsonify(archivos)


# Obtener imagen procesada específica
@app.route('/results/images/<filename>', methods=['GET'])
def obtener_imagen(filename):
    ruta = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(ruta):
        return "Imagen no encontrada", 404
    return send_file(ruta, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


