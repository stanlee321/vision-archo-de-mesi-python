import cv2
import numpy as np

# Variables globales para el dibujo del polígono
roi_points = []
roi_mask = None
drawing_roi = True
roi_closed = False
roi_image = None

GLOBAL_RESOLUTION = (640, 480)
VIDEO_SOURCE = "./video_prueba_1.mp4"

# Función para manejar eventos del mouse
def draw_roi(event, x, y, flags, param):
    global roi_points, roi_mask, drawing_roi, roi_closed, roi_image
    
    # Crear una copia de la imagen para dibujar
    img_copy = roi_image.copy()
    
    if drawing_roi and not roi_closed:
        # Clic izquierdo para añadir un punto
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            
        # Mover el mouse para ver la línea a dibujar
        elif event == cv2.EVENT_MOUSEMOVE and len(roi_points) > 0:
            # Crear una copia temporal para dibujar la línea de vista previa
            temp_img = img_copy.copy()
            cv2.line(temp_img, roi_points[-1], (x, y), (0, 255, 0), 1)
            
            # Dibujar los puntos y líneas existentes
            if len(roi_points) > 0:
                # Dibujar los puntos
                for point in roi_points:
                    cv2.circle(temp_img, point, 3, (0, 0, 255), -1)
                
                # Dibujar las líneas que conectan los puntos
                for i in range(len(roi_points) - 1):
                    cv2.line(temp_img, roi_points[i], roi_points[i + 1], (0, 255, 0), 2)
            
            # Mostrar las instrucciones en la parte superior
            show_instructions(temp_img)
            
            # Mostrar la imagen con la línea de vista previa
            cv2.imshow('Definir Region de Interes (ROI)', temp_img)
            return
    
    # Dibujar los puntos y líneas del polígono
    if len(roi_points) > 0:
        # Dibujar los puntos
        for point in roi_points:
            cv2.circle(img_copy, point, 3, (0, 0, 255), -1)
        
        # Dibujar las líneas que conectan los puntos
        for i in range(len(roi_points) - 1):
            cv2.line(img_copy, roi_points[i], roi_points[i + 1], (0, 255, 0), 2)
        
        # Si el polígono está cerrado, dibujar la línea de cierre
        if roi_closed and len(roi_points) > 2:
            cv2.line(img_copy, roi_points[-1], roi_points[0], (0, 255, 0), 2)
    
    # Si el polígono está cerrado, crear la máscara
    if roi_closed and len(roi_points) > 2:
        # Crear máscara en negro
        mask = np.zeros(roi_image.shape[:2], dtype=np.uint8)
        # Crear una copia de los puntos con el punto inicial al final para cerrar el polígono
        points_for_mask = roi_points.copy()
        if points_for_mask[0] != points_for_mask[-1]:
            points_for_mask.append(points_for_mask[0])
        # Dibujar el polígono relleno en blanco
        points_array = np.array([points_for_mask], dtype=np.int32)
        cv2.fillPoly(mask, points_array, 255)
        roi_mask = mask
        
        # Mostrar la máscara aplicada a la imagen
        masked_img = cv2.bitwise_and(roi_image, roi_image, mask=mask)
        alpha = 0.5
        cv2.addWeighted(masked_img, alpha, img_copy, 1 - alpha, 0, img_copy)
        
        # Mostrar texto
        cv2.putText(img_copy, "ROI definida. Presione 'c' para continuar o 'r' para reiniciar.", (10, img_copy.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Mostrar las instrucciones en la parte superior
    show_instructions(img_copy)
    
    # Mostrar la imagen con el ROI
    cv2.imshow('Definir Region de Interes (ROI)', img_copy)

# Función para mostrar las instrucciones en la imagen
def show_instructions(img):
    # Añadir un fondo semitransparente para hacer el texto más legible
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (400, 150), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Añadir las instrucciones
    cv2.putText(img, "Instrucciones:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "- Clic izquierdo: Agregar punto", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "- 't': Cerrar poligono actual", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "- 'r': Reiniciar poligono", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "- 'c': Continuar al procesamiento", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Función para definir la región de interés (ROI)
def define_roi(video_source):
    global roi_points, roi_mask, drawing_roi, roi_closed, roi_image
    
    # Inicializar variables
    roi_points = []
    roi_closed = False
    drawing_roi = True
    
    # Capturar un frame del video para dibujar el ROI
    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo leer el video.")
        return None
    
    # Redimensionar para mantener consistencia
    frame = cv2.resize(frame, GLOBAL_RESOLUTION)
    roi_image = frame.copy()
    
    # Configurar la ventana y el callback del mouse
    cv2.namedWindow('Definir Region de Interes (ROI)')
    cv2.setMouseCallback('Definir Region de Interes (ROI)', draw_roi)
    
    # Mostrar el frame inicial con instrucciones
    img_with_instructions = roi_image.copy()
    show_instructions(img_with_instructions)
    cv2.imshow('Definir Region de Interes (ROI)', img_with_instructions)
    
    # Esperar a que el usuario defina el ROI o salga
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 't' para cerrar el polígono actual
        if key == ord('t') and len(roi_points) > 2 and not roi_closed:
            roi_closed = True
            # Forzar redibujado
            draw_roi(None, 0, 0, None, None)
        
        # 'r' para reiniciar
        elif key == ord('r'):
            roi_points = []
            roi_closed = False
            img_with_instructions = roi_image.copy()
            show_instructions(img_with_instructions)
            cv2.imshow('Definir Region de Interes (ROI)', img_with_instructions)
        
        # 'c' para continuar si el ROI está definido
        elif key == ord('c') and roi_closed and roi_mask is not None:
            print("Continuando con el procesamiento...")
            cv2.destroyWindow('Definir Region de Interes (ROI)')
            break
        
        # 'q' para salir
        elif key == ord('q'):
            print("Saliendo sin definir ROI...")
            cv2.destroyWindow('Definir Region de Interes (ROI)')
            cap.release()
            return None
    
    cap.release()
    return roi_mask


# Función principal
def main():
    # Capturar video desde la cámara o archivo
    
    # Definir la región de interés
    roi_mask = define_roi(VIDEO_SOURCE)
    
    if roi_mask is None:
        print("No se definió una región de interés. Saliendo.")
        return
    
    # Continuar con el procesamiento del video
    video = cv2.VideoCapture(VIDEO_SOURCE)
    
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
    
        # Redimensionar el frame a GLOBAL_RESOLUTION
        frame = cv2.resize(frame, GLOBAL_RESOLUTION)
        
        # Crear una copia para mostrar el ROI
        frame_with_roi = frame.copy()
        
        # Dibujar el contorno del ROI en la imagen
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_with_roi, contours, -1, (0, 255, 0), 2)
        
        # Aplicar máscara ROI al frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    
        # Convertir a HSV para detectar el color
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        
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
        cv2.imshow('ROI Frame', frame_with_roi)
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


# Ejecutar el programa
if __name__ == "__main__":
    main()
