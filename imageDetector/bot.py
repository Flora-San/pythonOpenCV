import cv2

# Cargar una imagen desde el archivo
image = cv2.imread('imagen.jpg')

# Verificar si la imagen se cargó correctamente
if image is not None:
    # Realizar análisis de la imagen aquí

    # Mostrar la imagen en una ventana
    cv2.imshow('Imagen', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('No se pudo cargar la imagen.')

# Finalizar el programa
