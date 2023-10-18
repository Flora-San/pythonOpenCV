import cv2
import pytest

from imageDetector.classifier import detect_faces


# Función para crear una imagen con una cara y guardarla en un archivo
def create_image_with_face(image_path):
    # Crear una imagen en blanco
    image = cv2.imread('/home/flora/PycharmProjects/pythonOpenCV/images/one-face.jpg')

    # Dibujar una cara en la imagen
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Guardar la imagen con la cara en el archivo
    cv2.imwrite(image_path, image)


# Prueba 1: Detección de una cara en una imagen con una cara
def test_face_detection_with_single_face():
    image_path = '/home/flora/PycharmProjects/pythonOpenCV/images/one-face.jpeg'
    create_image_with_face(image_path)

    detected_faces = detect_faces(image_path)
    expected_faces = 1

    assert detected_faces == expected_faces, f"Se esperaba una cara, pero se detectaron {detected_faces}."


if __name__ == "__main":
    test_face_detection_with_single_face()

