import cv2
import numpy as np

# Capturar video desde la cámara
video = cv2.VideoCapture("./video_prueba_1.mp4")

# Verificar si la cámara se abrió correctamente
if not video.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Configurar la resolución de la cámara
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Cámara inicializada correctamente")

# Crear ventana para los trackbars
cv2.namedWindow('Controles HSV')

# Valores iniciales para HSV - ROJO específico para la pelota
# Valores ajustados para detectar mejor la pelota roja en la escena
h_min, s_min, v_min = 160, 118, 127
h_max, s_max, v_max = 179, 255, 255

# Crear trackbars para ajustar los valores HSV
def nothing(x):
    pass

cv2.createTrackbar('H Min', 'Controles HSV', h_min, 179, nothing)
cv2.createTrackbar('S Min', 'Controles HSV', s_min, 255, nothing)
cv2.createTrackbar('V Min', 'Controles HSV', v_min, 255, nothing)
cv2.createTrackbar('H Max', 'Controles HSV', h_max, 179, nothing)
cv2.createTrackbar('S Max', 'Controles HSV', s_max, 255, nothing)
cv2.createTrackbar('V Max', 'Controles HSV', v_max, 255, nothing)
# Ajustar área mínima para ser más pequeña (para detectar la pelota pequeña)
cv2.createTrackbar('Área mínima', 'Controles HSV', 20, 2000, nothing)
# Aumentar circularidad para asegurar que sea más circular
cv2.createTrackbar('Circularidad', 'Controles HSV', 60, 100, nothing)
# Activar por defecto el rango doble para rojo
cv2.createTrackbar('Tipo Color', 'Controles HSV', 1, 1, nothing)
# Parámetros de HoughCircles
cv2.createTrackbar('MinRadius', 'Controles HSV', 3, 50, nothing)
cv2.createTrackbar('MaxRadius', 'Controles HSV', 15, 100, nothing)
cv2.createTrackbar('Param1', 'Controles HSV', 30, 300, nothing)
cv2.createTrackbar('Param2', 'Controles HSV', 20, 100, nothing)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        # Reiniciar el video cuando llegue al final
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Redimensionar el frame a 320x240
    frame = cv2.resize(frame, (320, 240))

    # Convertir a HSV para detectar el color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Obtener valores actuales de los trackbars
    h_min = cv2.getTrackbarPos('H Min', 'Controles HSV')
    s_min = cv2.getTrackbarPos('S Min', 'Controles HSV')
    v_min = cv2.getTrackbarPos('V Min', 'Controles HSV')
    h_max = cv2.getTrackbarPos('H Max', 'Controles HSV')
    s_max = cv2.getTrackbarPos('S Max', 'Controles HSV')
    v_max = cv2.getTrackbarPos('V Max', 'Controles HSV')
    area_min = cv2.getTrackbarPos('Área mínima', 'Controles HSV')
    circ_min = cv2.getTrackbarPos('Circularidad', 'Controles HSV') / 100.0
    tipo_color = cv2.getTrackbarPos('Tipo Color', 'Controles HSV')
    min_radius = cv2.getTrackbarPos('MinRadius', 'Controles HSV')
    max_radius = cv2.getTrackbarPos('MaxRadius', 'Controles HSV')
    param1 = cv2.getTrackbarPos('Param1', 'Controles HSV')
    param2 = cv2.getTrackbarPos('Param2', 'Controles HSV')
    
    # Definir el rango de colores en HSV usando los trackbars
    color_bajo = np.array([h_min, s_min, v_min])
    color_alto = np.array([h_max, s_max, v_max])
    
    # Si tipo_color es 1, usamos el segundo rango del rojo (160-179)
    if tipo_color == 1:
        color_bajo1 = np.array([0, s_min, v_min])
        color_alto1 = np.array([10, s_max, v_max])
        color_bajo2 = np.array([160, s_min, v_min])
        color_alto2 = np.array([179, s_max, v_max])
        mask1 = cv2.inRange(hsv, color_bajo1, color_alto1)
        mask2 = cv2.inRange(hsv, color_bajo2, color_alto2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # Crear máscara para detectar el color
        mask = cv2.inRange(hsv, color_bajo, color_alto)
    
    # Aplicar operaciones morfológicas para mejorar la máscara
    kernel = np.ones((3,3), np.uint8)  # Reducir tamaño del kernel para preservar detalles pequeños
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Aplicar GaussianBlur para suavizar bordes
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Crear copias del frame para dibujar
    frame_contornos = frame.copy()
    
    # Crear una imagen en negro para mostrar solo las detecciones
    resultado = np.zeros_like(frame)
    
    # Método 1: Usar HoughCircles para detectar círculos directamente en la máscara
    # Este método es más específico para formas circulares
    circles = None
    
    # Solo intentar encontrar círculos si hay suficientes píxeles blancos en la máscara
    if cv2.countNonZero(mask) > 10:
        mask_gray = mask.copy()  # La máscara ya está en escala de grises
        
        # Asegurarse de que param1 y param2 no sean cero
        param1 = max(1, param1)
        param2 = max(1, param2)
        
        try:
            # Aplicar la transformada de Hough para detectar círculos
            circles = cv2.HoughCircles(
                mask_gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=20, 
                param1=param1, 
                param2=param2,
                minRadius=min_radius, 
                maxRadius=max_radius
            )
        except Exception as e:
            print(f"Error en HoughCircles: {e}")
    
    # Método 2: Usar findContours y analizar la circularidad
    # Encontrar contornos en la máscara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar para contar objetos detectados
    objetos_detectados = 0
    
    # Dibujar círculos detectados por HoughCircles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Dibujar el círculo exterior
            cv2.circle(frame_contornos, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Dibujar el centro del círculo
            cv2.circle(frame_contornos, (i[0], i[1]), 2, (0, 0, 255), 3)
            
            # Extraer la región del círculo para mostrarla en resultado
            mask_circulo = np.zeros_like(mask)
            cv2.circle(mask_circulo, (i[0], i[1]), i[2], 255, -1)
            roi = cv2.bitwise_and(frame, frame, mask=mask_circulo)
            resultado = cv2.add(resultado, roi)
            
            # Mostrar información del círculo
            radio = i[2]
            area = np.pi * (radio ** 2)
            cv2.putText(resultado, f"R:{radio} A:{int(area)}", (i[0]-40, i[1]), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            objetos_detectados += 1

    # Procesar contornos si no se detectaron círculos con HoughCircles
    if objetos_detectados == 0:
        for cnt in contornos:
            # Obtener el área y el perímetro del contorno
            area = cv2.contourArea(cnt)
            
            # Filtrar por área mínima
            if area < area_min:
                continue

            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:  # Evitar divisiones por cero
                continue

            # Calcular la circularidad: (4π * Área) / (Perímetro^2)
            circularidad = (4 * np.pi * area) / (perimetro ** 2)

            # Dibujar todos los contornos en verde en la ventana de contornos
            cv2.drawContours(frame_contornos, [cnt], -1, (0, 255, 0), 2)
            
            # Si cumple con los criterios de circularidad, dibujar en rojo y mostrar en resultado
            if circularidad > circ_min:
                # Obtener el círculo mínimo que engloba al contorno
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Dibujar el círculo en frame_contornos
                cv2.circle(frame_contornos, center, radius, (0, 0, 255), 2)
                
                # Crear máscara para este círculo
                mask_circulo = np.zeros_like(mask)
                cv2.circle(mask_circulo, center, radius, 255, -1)
                
                # Copiar pixels de la imagen original usando la máscara del círculo
                roi = cv2.bitwise_and(frame, frame, mask=mask_circulo)
                resultado = cv2.add(resultado, roi)
                
                # Agregar texto con información sobre el objeto
                cv2.putText(resultado, f"R:{radius} C:{circularidad:.2f}", (center[0]-40, center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                objetos_detectados += 1

    # Mostrar las máscaras y resultados
    cv2.imshow('Frame Original', frame)
    cv2.imshow('Máscara HSV', mask)
    cv2.imshow('Contornos', frame_contornos)
    cv2.imshow('Detección de Círculos', resultado)

    # Mostrar los valores HSV actuales
    font = cv2.FONT_HERSHEY_SIMPLEX
    texto = f"HSV Min: [{h_min}, {s_min}, {v_min}], Max: [{h_max}, {s_max}, {v_max}]"
    cv2.putText(frame, texto, (10, 20), font, 0.5, (255, 255, 255), 1)
    
    # Mostrar mensaje sobre el tipo de color y objetos detectados
    if tipo_color == 0:
        color_msg = "Rango simple (H_min a H_max)"
    else:
        color_msg = "Rango doble para rojo (0-10 y 160-179)"
    cv2.putText(frame, color_msg, (10, 35), font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Objetos: {objetos_detectados}", (10, 50), font, 0.5, (255, 255, 255), 1)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video.release()
cv2.destroyAllWindows()
