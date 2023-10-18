import os

import cv2


# Función para detectar caras en una imagen
def detect_faces(image_path):
    # Cargar el clasificador de cascadas Haar para la detección de caras
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Cargar la imagen desde el archivo
    image = cv2.imread(image_path)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Realizar la detección de caras
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return len(faces)


# Prueba de detección de caras en una imagen
def test_face_detection():
    image_path = '/home/flora/PycharmProjects/pythonOpenCV/images/one-face.jpg'  # Reemplaza con la ruta de tu imagen
    expected_faces = 1  # Reemplaza con la cantidad esperada de caras en la imagen
    detected_faces = detect_faces(image_path)
    # file_name = os.path.join(os.path.dirname(__file__), 'one-face.jpg')
    # assert os.path.exists(file_name)

    # img = cv2.imread(file_name, -1)
    assert detected_faces == expected_faces, f"Se esperaban {expected_faces} caras, pero se detectaron { detected_faces}."


if __name__ == "__main":
    test_face_detection()
