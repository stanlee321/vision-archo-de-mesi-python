import cv2
import numpy as np

# Variables globales para el dibujo del polígono ROI principal
roi_points = []
roi_mask = None
drawing_roi = True
roi_closed = False
roi_image = None

# Variables globales para las zonas de interés (IZQUIERDA, CENTRO, DERECHA)
zones = {
    "IZQUIERDA": {"points": [], "mask": None, "color": (255, 0, 0)},   # Azul
    "CENTRO": {"points": [], "mask": None, "color": (0, 255, 0)},      # Verde
    "DERECHA": {"points": [], "mask": None, "color": (0, 0, 255)}      # Rojo
}
current_zone = None
zone_drawing = False
zone_closed = False
zone_image = None

GLOBAL_RESOLUTION = (640, 480)
VIDEO_SOURCE = "./video_prueba_1.mp4"

# Función para manejar eventos del mouse para el ROI principal
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
            show_instructions(temp_img, "roi")
            
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
    show_instructions(img_copy, "roi")
    
    # Mostrar la imagen con el ROI
    cv2.imshow('Definir Region de Interes (ROI)', img_copy)

# Función para manejar eventos del mouse para las zonas
def draw_zone(event, x, y, flags, param):
    global zones, current_zone, zone_drawing, zone_closed, zone_image
    
    if current_zone is None:
        return
    
    # Crear una copia de la imagen para dibujar
    img_copy = zone_image.copy()
    
    # Dibujar todas las zonas ya definidas
    for zone_name, zone_info in zones.items():
        if zone_name != current_zone and len(zone_info["points"]) > 0:
            # Dibujar puntos y líneas de las zonas ya definidas
            for point in zone_info["points"]:
                cv2.circle(img_copy, point, 3, zone_info["color"], -1)
            
            for i in range(len(zone_info["points"]) - 1):
                cv2.line(img_copy, zone_info["points"][i], zone_info["points"][i + 1], zone_info["color"], 2)
            
            # Si la zona está cerrada, dibujar línea de cierre
            if len(zone_info["points"]) > 2 and zone_info["mask"] is not None:
                cv2.line(img_copy, zone_info["points"][-1], zone_info["points"][0], zone_info["color"], 2)
                
                # Mostrar el nombre de la zona
                center_x = sum(p[0] for p in zone_info["points"]) // len(zone_info["points"])
                center_y = sum(p[1] for p in zone_info["points"]) // len(zone_info["points"])
                cv2.putText(img_copy, zone_name, (center_x - 40, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_info["color"], 2)
    
    # Manejar eventos para la zona actual
    if zone_drawing and not zone_closed:
        current_points = zones[current_zone]["points"]
        current_color = zones[current_zone]["color"]
        
        # Clic izquierdo para añadir un punto
        if event == cv2.EVENT_LBUTTONDOWN:
            zones[current_zone]["points"].append((x, y))
            
        # Mover el mouse para ver la línea a dibujar
        elif event == cv2.EVENT_MOUSEMOVE and len(current_points) > 0:
            # Crear una copia temporal para dibujar la línea de vista previa
            temp_img = img_copy.copy()
            cv2.line(temp_img, current_points[-1], (x, y), current_color, 1)
            
            # Dibujar los puntos y líneas existentes
            for point in current_points:
                cv2.circle(temp_img, point, 3, current_color, -1)
            
            for i in range(len(current_points) - 1):
                cv2.line(temp_img, current_points[i], current_points[i + 1], current_color, 2)
            
            # Mostrar las instrucciones en la parte superior
            show_instructions(temp_img, "zone", current_zone)
            
            # Mostrar la imagen con la línea de vista previa
            cv2.imshow('Definir Zonas de Interés', temp_img)
            return
    
    # Dibujar los puntos y líneas de la zona actual
    current_points = zones[current_zone]["points"]
    current_color = zones[current_zone]["color"]
    
    if len(current_points) > 0:
        # Dibujar los puntos
        for point in current_points:
            cv2.circle(img_copy, point, 3, current_color, -1)
        
        # Dibujar las líneas que conectan los puntos
        for i in range(len(current_points) - 1):
            cv2.line(img_copy, current_points[i], current_points[i + 1], current_color, 2)
        
        # Si la zona está cerrada, dibujar la línea de cierre
        if zone_closed and len(current_points) > 2:
            cv2.line(img_copy, current_points[-1], current_points[0], current_color, 2)
            
            # Mostrar el nombre de la zona
            center_x = sum(p[0] for p in current_points) // len(current_points)
            center_y = sum(p[1] for p in current_points) // len(current_points)
            cv2.putText(img_copy, current_zone, (center_x - 40, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
    
    # Si la zona está cerrada, crear la máscara
    if zone_closed and len(current_points) > 2:
        # Crear máscara en negro
        mask = np.zeros(zone_image.shape[:2], dtype=np.uint8)
        # Crear una copia de los puntos con el punto inicial al final para cerrar el polígono
        points_for_mask = current_points.copy()
        if points_for_mask[0] != points_for_mask[-1]:
            points_for_mask.append(points_for_mask[0])
        # Dibujar el polígono relleno en blanco
        points_array = np.array([points_for_mask], dtype=np.int32)
        cv2.fillPoly(mask, points_array, 255)
        zones[current_zone]["mask"] = mask
        
        # Mostrar texto
        cv2.putText(img_copy, f"Zona {current_zone} definida. Presione 'c' para continuar, 'r' para reiniciar.", 
                   (10, img_copy.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Mostrar las instrucciones en la parte superior
    show_instructions(img_copy, "zone", current_zone)
    
    # Mostrar la imagen con la zona
    cv2.imshow('Definir Zonas de Interés', img_copy)

# Función para mostrar las instrucciones en la imagen
def show_instructions(img, mode="roi", zone_name=None):
    # Añadir un fondo semitransparente para hacer el texto más legible
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (400, 150), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Añadir las instrucciones según el modo
    if mode == "roi":
        cv2.putText(img, "Instrucciones ROI Principal:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "- Clic izquierdo: Agregar punto", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "- 't': Cerrar poligono actual", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "- 'r': Reiniciar poligono", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "- 'c': Continuar al procesamiento", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    elif mode == "zone":
        color = zones[zone_name]["color"] if zone_name else (255, 255, 255)
        cv2.putText(img, f"Definiendo Zona {zone_name}:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, "- Clic izquierdo: Agregar punto", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "- 't': Cerrar poligono actual", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "- 'r': Reiniciar poligono", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "- 'c': Continuar a siguiente zona/procesamiento", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
    show_instructions(img_with_instructions, "roi")
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
            show_instructions(img_with_instructions, "roi")
            cv2.imshow('Definir Region de Interes (ROI)', img_with_instructions)
        
        # 'c' para continuar si el ROI está definido
        elif key == ord('c') and roi_closed and roi_mask is not None:
            print("Continuando con la definición de zonas...")
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

# Función para definir las zonas de interés (IZQUIERDA, CENTRO, DERECHA)
def define_zones(video_source):
    global zones, current_zone, zone_drawing, zone_closed, zone_image
    
    # Capturar un frame del video para dibujar las zonas
    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo leer el video.")
        return None
    
    # Redimensionar para mantener consistencia
    frame = cv2.resize(frame, GLOBAL_RESOLUTION)
    zone_image = frame.copy()
    
    # Configurar la ventana y el callback del mouse
    cv2.namedWindow('Definir Zonas de Interés')
    cv2.setMouseCallback('Definir Zonas de Interés', draw_zone)
    
    # Definir cada zona en secuencia
    zone_order = ["IZQUIERDA", "CENTRO", "DERECHA"]
    
    for zone_name in zone_order:
        # Inicializar variables para la zona actual
        current_zone = zone_name
        zone_drawing = True
        zone_closed = False
        zones[zone_name]["points"] = []
        zones[zone_name]["mask"] = None
        
        # Mostrar el frame inicial con instrucciones para esta zona
        img_with_instructions = zone_image.copy()
        show_instructions(img_with_instructions, "zone", zone_name)
        cv2.imshow('Definir Zonas de Interés', img_with_instructions)
        
        print(f"Definiendo zona {zone_name}...")
        
        # Esperar a que el usuario defina la zona o salga
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # 't' para cerrar el polígono actual
            if key == ord('t') and len(zones[zone_name]["points"]) > 2 and not zone_closed:
                zone_closed = True
                # Forzar redibujado
                draw_zone(None, 0, 0, None, None)
            
            # 'r' para reiniciar la zona actual
            elif key == ord('r'):
                zones[zone_name]["points"] = []
                zone_closed = False
                img_with_instructions = zone_image.copy()
                show_instructions(img_with_instructions, "zone", zone_name)
                draw_zone(None, 0, 0, None, None)
            
            # 'c' para continuar a la siguiente zona o finalizar
            elif key == ord('c') and zone_closed and zones[zone_name]["mask"] is not None:
                print(f"Zona {zone_name} definida, continuando...")
                break
            
            # 'q' para salir
            elif key == ord('q'):
                print("Saliendo sin definir todas las zonas...")
                cv2.destroyWindow('Definir Zonas de Interés')
                cap.release()
                return False
    
    print("Todas las zonas han sido definidas.")
    cv2.destroyWindow('Definir Zonas de Interés')
    cap.release()
    return True

# Función para determinar en qué zona se encuentra un punto
def get_zone_location(point):
    global zones
    for zone_name, zone_info in zones.items():
        if zone_info["mask"] is not None and zone_info["mask"][point[1], point[0]] > 0:
            return zone_name
    return None

# Función principal
def main():
    # Definir la región de interés principal
    roi_mask = define_roi(VIDEO_SOURCE)
    
    if roi_mask is None:
        print("No se definió una región de interés principal. Saliendo.")
        return
    
    # Definir las zonas de interés (IZQUIERDA, CENTRO, DERECHA)
    if not define_zones(VIDEO_SOURCE):
        print("No se definieron todas las zonas de interés. Saliendo.")
        return
    
    # Continuar con el procesamiento del video
    video = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Verificar si la cámara se abrió correctamente
    if not video.isOpened():
        print("Error: No se pudo abrir la cámara o el video.")
        exit()
    
    # Configurar la resolución del video
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Video inicializado correctamente")
    
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
    
    # Variable para trackear la última zona donde estuvo la pelota
    last_zone = None
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            # Reiniciar el video cuando llegue al final
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
    
        # Redimensionar el frame a GLOBAL_RESOLUTION
        frame = cv2.resize(frame, GLOBAL_RESOLUTION)
        
        # Crear una copia para mostrar el ROI y las zonas
        frame_with_overlays = frame.copy()
        
        # Dibujar el contorno del ROI principal
        roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_with_overlays, roi_contours, -1, (255, 255, 255), 2)
        
        # Dibujar contornos de las zonas con sus nombres
        for zone_name, zone_info in zones.items():
            if zone_info["mask"] is not None:
                zone_contours, _ = cv2.findContours(zone_info["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_with_overlays, zone_contours, -1, zone_info["color"], 2)
                
                # Mostrar el nombre de la zona
                M = cv2.moments(zone_contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(frame_with_overlays, zone_name, (cx - 40, cy), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_info["color"], 2)
        
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
        current_zone = None
        
        # Dibujar círculos detectados por HoughCircles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Obtener las coordenadas del centro y el radio
                center = (i[0], i[1])
                radio = i[2]
                
                # Dibujar el círculo exterior
                cv2.circle(frame_contornos, center, radio, (0, 255, 0), 2)
                # Dibujar el centro del círculo
                cv2.circle(frame_contornos, center, 2, (0, 0, 255), 3)
                
                # Extraer la región del círculo para mostrarla en resultado
                mask_circulo = np.zeros_like(mask)
                cv2.circle(mask_circulo, center, radio, 255, -1)
                roi = cv2.bitwise_and(frame, frame, mask=mask_circulo)
                resultado = cv2.add(resultado, roi)
                
                # Mostrar información del círculo
                area = np.pi * (radio ** 2)
                cv2.putText(resultado, f"R:{radio} A:{int(area)}", (center[0]-40, center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Determinar en qué zona se encuentra el centro del círculo
                current_zone = get_zone_location(center)
                
                # Si está en una zona, dibujar información adicional
                if current_zone is not None:
                    zone_color = zones[current_zone]["color"]
                    cv2.putText(frame_with_overlays, f"ZONA: {current_zone}", (center[0]-60, center[1]-20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                    
                    # Verificar si cambió de zona
                    if current_zone != last_zone:
                        print(f"Pelota entró en zona: {current_zone}")
                        last_zone = current_zone
                        # Aquí implementa tu lógica condicional
                        if current_zone == "IZQUIERDA":
                            # Acciones para la zona IZQUIERDA
                            sent_to_arduino("IZQUIERDA")
                        elif current_zone == "CENTRO":
                            # Acciones para la zona CENTRO
                            sent_to_arduino("CENTRO")
                        elif current_zone == "DERECHA":
                            # Acciones para la zona DERECHA
                            sent_to_arduino("DERECHA")
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
                    
                    # Determinar en qué zona se encuentra el centro del círculo
                    current_zone = get_zone_location(center)
                    
                    # Si está en una zona, dibujar información adicional
                    if current_zone is not None:
                        zone_color = zones[current_zone]["color"]
                        cv2.putText(frame_with_overlays, f"ZONA: {current_zone}", (center[0]-60, center[1]-20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                        
                        # Verificar si cambió de zona
                        if current_zone != last_zone:
                            print(f"Pelota entró en zona: {current_zone}")
                            last_zone = current_zone
                        
                            # Aquí implementa tu lógica condicional
                            if current_zone == "IZQUIERDA":
                                # Acciones para la zona IZQUIERDA
                                sent_to_arduino("IZQUIERDA")
                            elif current_zone == "CENTRO":
                                # Acciones para la zona CENTRO
                                sent_to_arduino("CENTRO")
                            elif current_zone == "DERECHA":
                                # Acciones para la zona DERECHA
                                sent_to_arduino("DERECHA")
                    objetos_detectados += 1
    
        # Mostrar las máscaras y resultados
        cv2.imshow('Zonas y ROI', frame_with_overlays)
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
        
        # Mostrar la zona actual de la pelota si existe
        if current_zone:
            zone_color = zones[current_zone]["color"]
            cv2.putText(frame, f"Zona: {current_zone}", (10, 70), font, 0.6, zone_color, 2)
    
        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    video.release()
    cv2.destroyAllWindows()


def sent_to_arduino(zone):
    print(f"Enviando zona {zone} al Arduino")
    # Aquí puedes agregar la lógica para enviar la zona al Arduino
    # RESERVADO PARA ARDUINO
    
    
    
# Ejecutar el programa
if __name__ == "__main__":
    main()
