import cv2

class Imagen:

    def __init__(self, img):
        #self.imagen = cv2.imread(img, cv2.IMREAD_COLOR)
        self.imagen = cv2.imdecode(img, cv2.IMREAD_COLOR)
        self.gris_roi = []
        self.color_roi = []
        
    def preprocesamiento(self):
        try:
            # Redimensionar la imagen con interpolación Lanczos
            imagen_lan = cv2.resize(self.imagen, (600, 600), interpolation=cv2.INTER_LANCZOS4)

            # Convertir la imagen a escala de grises
            imagen_gris = cv2.cvtColor(imagen_lan, cv2.COLOR_BGR2GRAY)

            # Devolver la imagen en escala de grises
            return imagen_lan, imagen_gris
        
        except Exception as e:
            # Manejo de errores
            print(f"Error en el preprocesamiento: {e}")

    def deteccion_facial(self, imagen_lan, imagen_gris):
        try:
            cascada_rostros = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')

            rostros = cascada_rostros.detectMultiScale(imagen_gris, scaleFactor=1.3, minNeighbors=5)

            margen = 23
            for x, y, w, h in rostros:
                nuevo_x = max(0, x - margen)
                nuevo_y = max(0, y - margen)
                nuevo_w = min(imagen_gris.shape[1] - nuevo_x, w + 2 * margen)
                nuevo_h = min(imagen_gris.shape[0] - nuevo_y, h + 2 * margen)
                
                roi_gris = imagen_gris[nuevo_y:nuevo_y+nuevo_h, nuevo_x:nuevo_x+nuevo_w]
                roi_color = imagen_lan[nuevo_y:nuevo_y+nuevo_h, nuevo_x:nuevo_x+nuevo_w]
                
                self.gris_roi.append(cv2.resize(roi_gris, (300, 300), interpolation=cv2.INTER_LANCZOS4))
                self.color_roi.append(cv2.resize(roi_color, (300, 300), interpolation=cv2.INTER_LANCZOS4))
            
            return self.gris_roi, self.color_roi

        except Exception as e:
            print(f"Error en la detección facial: {e}")