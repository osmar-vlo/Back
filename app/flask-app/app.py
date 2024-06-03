import glob
import json
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import euclidean_distances
import os
import cv2
import numpy as np

from Imagen import Imagen
from Rostro import Rostro
from KNN import KNN
from GIFGenerator import GIFGenerator

app = Flask(__name__)
CORS(app) # Habilita CORS para toda la aplicación

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocesamiento', methods=['POST'])
def procesar_imagen():
    try:
        img_bytes = request.files['imagen'].read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Crear una instancia de la clase Imagen
        imagen = Imagen(npimg)

        # Realizar el preprocesamiento y obtener las imágenes procesadas
        imagen_lan, imagen_gris = imagen.preprocesamiento()

        # Realizar recorte
        landmarks_gris_roi, landmarks_color_roi = imagen.deteccion_facial(imagen_lan, imagen_gris)

        # Crear una instancia de la clase Rostro con los ROIs obtenidos
        rostro = Rostro(landmarks_gris_roi, landmarks_color_roi)

        # Llamar al método get_landmarks_image para procesar los ROIs y obtener landmarks
        img, landmarks, distancia_36_33, distancia_39_33, distancia_42_33, distancia_45_33, distancia_48_33, distancia_54_33 = rostro.get_landmarks_vector()
        
        response_data_list = []
        for i in range(len(landmarks)):
            response_data = {
                "img": img[i],
                "landmarks": landmarks[i],
                "distancia_36_33": distancia_36_33[i],
                "distancia_39_33": distancia_39_33[i],
                "distancia_42_33": distancia_42_33[i],
                "distancia_45_33": distancia_45_33[i],
                "distancia_48_33": distancia_48_33[i],
                "distancia_54_33": distancia_54_33[i]
            }
            response_data_list.append(response_data)

        return jsonify(response_data_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Ruta para recibir los datos seleccionados del formulario
@app.route('/knn', methods=['POST'])
def procesar_knn():
    try:
        # Obtener los datos en formato JSON
        data = request.json
        etnia = data.get('etnia')
        genero = data.get('genero')
        edad = data.get('edad')
        distancia_36_33 = data.get('distancia_36_33')
        distancia_39_33 = data.get('distancia_39_33')
        distancia_42_33 = data.get('distancia_42_33')
        distancia_45_33 = data.get('distancia_45_33')
        distancia_48_33 = data.get('distancia_48_33')
        distancia_54_33 = data.get('distancia_54_33')

        knn = KNN(etnia, genero, edad, distancia_36_33, distancia_39_33, distancia_42_33, distancia_45_33, distancia_48_33, distancia_54_33)

        # Realizar recorte
        imagenes_base64, imagenes_landmarks, gifs_base64 = knn.clasificar()

        # Suponiendo que retornes un resultado como ejemplo
        resultado = {
            "status": "success",
            "message": "Datos procesados correctamente",
            "imagenes_base64": imagenes_base64,
            "imagenes_landmarks":imagenes_landmarks,
            "gifs_base64": gifs_base64
        }
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)