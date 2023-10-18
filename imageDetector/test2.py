import numpy as np
import cv2

from imageDetector.classifier import detect_faces


# Función para crear una imagen en blanco y guardarla en un archivo
def create_blank_image(image_path, width=640, height=480):
    # Crear una matriz NumPy en blanco con el tamaño especificado
    image = np.zeros((height, width, 3), np.uint8)

    # Guardar la imagen en el archivo especificado
    cv2.imwrite(image_path, image)


# Prueba 2: Detección de caras en una imagen sin caras
def test_face_detection_with_no_faces():
    image_path = '../image_with_no_faces.jpg'
    create_blank_image(image_path)

    detected_faces = detect_faces(image_path)
    expected_faces = 0

    assert detected_faces == expected_faces, f"No se esperaban caras, pero se detectaron {detected_faces}."


if __name__ == "__main":
    test_face_detection_with_no_faces()
