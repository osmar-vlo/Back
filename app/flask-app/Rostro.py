import cv2
import dlib
import base64
import math

class Rostro:

    def __init__(self, gris_cuts, color_cuts):
            self.gris_cuts = gris_cuts
            self.color_cuts = color_cuts
            self.landmarks = []
            self.img = []
            self.distancia_36_33 = []
            self.distancia_39_33 = []
            self.distancia_42_33 = []
            self.distancia_45_33 = []
            self.distancia_48_33 = []
            self.distancia_54_33 = []

    def get_landmarks_vector(self):
        try:
            # Cargar el detector de caras de dlib (HOG-based)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("./resources/shape_predictor_68_face_landmarks.dat")        
            # Iterar sobre las imágenes en escala de grises y en color
            
            for imagen_gris, imagen_color in zip(self.gris_cuts, self.color_cuts):
                caras = detector(imagen_gris)
                for cara in caras:
                    landmarks = predictor(imagen_gris, cara)
                    # Crear una copia de la imagen a color para dibujar las landmarks
                    imagen_color_con_landmarks = imagen_color.copy()

                    # Definir puntos de interés
                    puntos_interes = {
                        "p36": landmarks.part(36),
                        "p39": landmarks.part(39),
                        "p42": landmarks.part(42),
                        "p45": landmarks.part(45),
                        "p48": landmarks.part(48),
                        "p54": landmarks.part(54),
                        "punto_referencia": landmarks.part(33),
                    }

                    # Dibujar líneas
                    for punto in puntos_interes:
                        cv2.line(imagen_color_con_landmarks, (puntos_interes[punto].x, puntos_interes[punto].y), (puntos_interes["punto_referencia"].x, puntos_interes["punto_referencia"].y), (255, 0, 0), 1)

                    # Calcular y guardar las distancias calculadas
                    self.distancia_36_33.append(math.sqrt((landmarks.part(36).x - landmarks.part(33).x)**2 + (landmarks.part(36).y - landmarks.part(33).y)**2))
                    self.distancia_39_33.append(math.sqrt((landmarks.part(39).x - landmarks.part(33).x)**2 + (landmarks.part(39).y - landmarks.part(33).y)**2))
                    self.distancia_42_33.append(math.sqrt((landmarks.part(42).x - landmarks.part(33).x)**2 + (landmarks.part(42).y - landmarks.part(33).y)**2))
                    self.distancia_45_33.append(math.sqrt((landmarks.part(45).x - landmarks.part(33).x)**2 + (landmarks.part(45).y - landmarks.part(33).y)**2))
                    self.distancia_48_33.append(math.sqrt((landmarks.part(48).x - landmarks.part(33).x)**2 + (landmarks.part(48).y - landmarks.part(33).y)**2))
                    self.distancia_54_33.append(math.sqrt((landmarks.part(54).x - landmarks.part(33).x)**2 + (landmarks.part(54).y - landmarks.part(33).y)**2))

                    # Dibujar puntos y líneas para diferentes regiones de landmarks
                    for n in range(68):
                        # Definir el color y el tipo de landmark
                        if n < 17:
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            color = (0, 255, 0)  # Verde para puntos del contorno facial
                        elif n < 22:
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            color = (0, 0, 100)  # Rojo para cejas
                        elif n < 27:
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            color = (0, 0, 255)  # Rojo para cejas
                        elif n < 36:
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            color = (150, 150, 0)  # Cyan para la nariz
                        elif n < 42:
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            color = (0, 255, 255)  # Amarillo para ojos
                        elif n < 48:
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            color = (0, 255, 100)  # Amarillo para ojos
                        else:
                            color = (150, 150, 0)  # Azul menta para boca y mandíbula
                        
                        # Dibujar el punto
                        cv2.circle(imagen_color_con_landmarks, (x, y), 2, color, -1)
                        
                        # Mostrar el número de landmark
                        cv2.putText(imagen_color_con_landmarks, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                    # Obtener las coordenadas de los puntos de los landmarks
                    puntos_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]  # Suponiendo 68 landmarks
                    
                    # Calcular los límites (rectángulo delimitador) de la región de landmarks
                    x_min = min(puntos_landmarks, key=lambda x: x[0])[0]
                    x_max = max(puntos_landmarks, key=lambda x: x[0])[0]
                    y_min = min(puntos_landmarks, key=lambda x: x[1])[1]
                    y_max = max(puntos_landmarks, key=lambda x: x[1])[1]
                    
                    # Recortar la región de landmarks de la imagen original
                    img_recortados = imagen_color[y_min:y_max, x_min:x_max]
                    landmarks_recortados = imagen_color_con_landmarks[y_min:y_max, x_min:x_max]

                    # Redimensionar la imagen recortada a 300x300 con interpolación Lanczos
                    img_redimensionados = cv2.resize(img_recortados, (300, 300), interpolation=cv2.INTER_LANCZOS4)
                    landmarks_redimensionados = cv2.resize(landmarks_recortados, (300, 300), interpolation=cv2.INTER_LANCZOS4)

                    # Guardar y convertir la imagen recortada a formato base64
                    _, rostro_encoded = cv2.imencode('.png', img_redimensionados)
                    img_base64 = base64.b64encode(rostro_encoded).decode('utf-8')

                    _, rostro_encoded = cv2.imencode('.png', landmarks_redimensionados)
                    land_base64 = base64.b64encode(rostro_encoded).decode('utf-8')
                    
                    # Agregar rostros recortados a la lista
                    self.img.append(img_base64)
                    self.landmarks.append(land_base64)

            return self.img, self.landmarks, self.distancia_36_33, self.distancia_39_33, self.distancia_42_33, self.distancia_45_33, self.distancia_48_33, self.distancia_54_33
        
        except Exception as e:
            # Manejo de errores
            print(f"Error en la detección de Landmarks: {e}")