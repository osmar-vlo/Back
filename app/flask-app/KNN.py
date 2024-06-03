import glob
import json
import base64
from sklearn.metrics.pairwise import euclidean_distances
import os
import numpy as np

class KNN:

    def __init__(self, etnia, genero, edad, distancia_36_33, distancia_39_33, distancia_42_33, distancia_45_33, distancia_48_33, distancia_54_33):
        self.etnia = etnia
        self.genero = genero
        self.edad = edad
        self.distancia_36_33 = distancia_36_33
        self.distancia_39_33 = distancia_39_33
        self.distancia_42_33 = distancia_42_33
        self.distancia_45_33 = distancia_45_33
        self.distancia_48_33 = distancia_48_33
        self.distancia_54_33 = distancia_54_33
        self.imagenes_base64 = []
        self.imagenes_landmarks = []
        self.gifs_base64 = []

    def clasificar(self):
        try:
            # Formar Etiqueta
            etiqueta = self.genero + self.edad

            # Definir el diccionario de rutas por etnia
            etnia_rutas = {
                "Afro": "/home/mxn/Documents/2024-A022/DatasetAgeGif/Afro/datos.json",
                "Arab": "/home/mxn/Documents/2024-A022/DatasetAgeGif/Árabes/datos.json",
                "Asian": "/home/mxn/Documents/2024-A022/DatasetAgeGif/Asia/datos.json",
                "Blanco": "/home/mxn/Documents/2024-A022/DatasetAgeGif/Europa/datos.json",
                "Latin": "/home/mxn/Documents/2024-A022/DatasetAgeGif/Hispano/datos.json"
            }

            # Cargar los datos solo de la etnia específica
            json_file_path = etnia_rutas[self.etnia]
            with open(json_file_path, 'r') as json_file:
                data_list = json.load(json_file)
                print(f"Cargados {len(data_list)} registros de {self.etnia}")

            # Filtrar las instancias que tengan la etiqueta específica
            filtered_data = [instance for instance in data_list if isinstance(instance, dict) and instance.get("etiqueta") == etiqueta]

            # Extraer características
            X = []

            for instance in filtered_data:
                # Verificar que todas las claves están presentes en el vector
                vector = instance.get("vector", {})
                required_keys = ["distancia_36_33", "distancia_39_33", "distancia_42_33", "distancia_45_33", "distancia_48_33", "distancia_54_33"]
                if all(key in vector for key in required_keys):
                    # Usar las distancias como características
                    distancia_36_33 = vector["distancia_36_33"][0]
                    distancia_39_33 = vector["distancia_39_33"][0]
                    distancia_42_33 = vector["distancia_42_33"][0]
                    distancia_45_33 = vector["distancia_45_33"][0]
                    distancia_48_33 = vector["distancia_48_33"][0]
                    distancia_54_33 = vector["distancia_54_33"][0]

                    X.append([distancia_36_33, distancia_39_33, distancia_42_33, distancia_45_33, distancia_48_33, distancia_54_33])
                else:
                    print(f"Instancia ignorada por falta de claves: {instance}")

            # Convertir a array de NumPy
            X = np.array(X)

            # Crear un array con las distancias de la imagen procesada
            Y = np.array([self.distancia_36_33, self.distancia_39_33, self.distancia_42_33, self.distancia_45_33, self.distancia_48_33, self.distancia_54_33]).reshape(1, -1)

            # Calcular la distancia euclidiana entre la imagen procesada y cada instancia del dataset combinado
            distancias_euclidianas = euclidean_distances(Y, X)

            # Ordenar las distancias euclidianas de menor a mayor
            indices_ordenados = np.argsort(distancias_euclidianas, axis=1)

            # Obtener los índices de los 5 vecinos más cercanos (con las menores distancias euclidianas)
            indices_menor_distancia = indices_ordenados[0][:5]

            # Obtener las rutas de las imágenes correspondientes a los 5 vecinos más cercanos
            rutas_vecinos_cercanos = [filtered_data[i]["ruta_nueva_imagen"] for i in indices_menor_distancia]
            rutas_vecinos_landamrks = [filtered_data[i]["ruta_nuevo_landmark"] for i in indices_menor_distancia]

            print("Rutas de las imágenes de los 5 vecinos más cercanos:")
            print(rutas_vecinos_cercanos)

            # Convertir imágenes de rutas_vecinos_cercanos a base64
            for ruta in rutas_vecinos_cercanos:
                try:
                    with open(ruta, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        self.imagenes_base64.append(encoded_string)

                    # Recortar la ruta para buscar el GIF
                    carpeta_recortada = os.path.dirname(os.path.dirname(ruta))
                    print(f"Ruta completa: {ruta}")
                    print(f"Ruta recortada: {carpeta_recortada}")
                    
                    gifs_encontrados = glob.glob(os.path.join(carpeta_recortada, "*.gif"))
                    if gifs_encontrados:
                        with open(gifs_encontrados[0], "rb") as gif_file:
                            encoded_gif = base64.b64encode(gif_file.read()).decode('utf-8')
                            self.gifs_base64.append(encoded_gif)
                    else:
                        self.gifs_base64.append(None)
                        print(f"No se encontró ningún GIF en la carpeta: {carpeta_recortada}")
                except Exception as e:
                    print(f"Error en el clasificador KNN - Imagen GIF: {e}")

            # Convertir imágenes de rutas_vecinos_land a base64
            for ruta in rutas_vecinos_landamrks:
                try:
                    with open(ruta, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        self.imagenes_landmarks.append(encoded_string)
                except Exception as e:
                    print(f"Error en el clasificador KNN - Landmarks: {e}")

            return self.imagenes_base64, self.imagenes_landmarks, self.gifs_base64
        
        except Exception as e:
            # Manejo de errores
            print(f"Error en el clasificador KNN: {e}")