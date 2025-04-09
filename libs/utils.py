import cv2
import numpy as np
import pickle
import torch
import os

# Variables globales para el dibujo del polígono ROI principal
roi_points = []
roi_mask = None
drawing_roi = True
roi_closed = False
roi_image = None


# Variables globales para las zonas de interés (IZQUIERDA, CENTRO, DERECHA)
current_zone = None
zone_drawing = False
zone_closed = False
zone_image = None



zones = {
    "IZQUIERDA": {"points": [], "mask": None, "color": (255, 0, 0)},   # Azul
    "CENTRO": {"points": [], "mask": None, "color": (0, 255, 0)},      # Verde
    "DERECHA": {"points": [], "mask": None, "color": (0, 0, 255)}      # Rojo
}

ROI_SAVE_FILE = "roi_config.pkl"

def save_roi_data(roi_mask_to_save, zones_to_save):
    """Saves the ROI mask and zones data to a file using pickle."""
    try:
        data_to_save = {
            "roi_mask": roi_mask_to_save,
            "zones": zones_to_save
        }
        with open(ROI_SAVE_FILE, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Configuración de ROI y zonas guardada en {ROI_SAVE_FILE}")
    except Exception as e:
        print(f"Error al guardar la configuración: {e}")

def load_roi_data():
    """Loads the ROI mask and zones data from a file if it exists."""
    global zones, roi_mask # Declare upfront that we might modify globals
    
    if os.path.exists(ROI_SAVE_FILE):
        try:
            with open(ROI_SAVE_FILE, 'rb') as f:
                data = pickle.load(f)
            print(f"Configuración de ROI y zonas cargada desde {ROI_SAVE_FILE}")
            # Update global zones with loaded points/masks
            # global zones, roi_mask # Remove redundant declaration
            loaded_roi_mask = data.get("roi_mask") # Load into temporary var first
            loaded_zones_data = data.get("zones", {}) 
            
            # Check if loaded data is valid before assigning to globals
            if loaded_roi_mask is None or not isinstance(loaded_zones_data, dict):
                 print("Error: Archivo de configuración inválido.")
                 raise ValueError("Invalid config file format") # Force jump to except block
                 
            # Assign to globals only after validation
            roi_mask = loaded_roi_mask 

            # Update points and masks for existing zones using the global zones dict
            for zone_name, loaded_info in loaded_zones_data.items():
                if zone_name in zones:
                     # Ensure loaded_info is a dictionary before accessing keys
                     if isinstance(loaded_info, dict):
                         zones[zone_name]["points"] = loaded_info.get("points", [])
                         zones[zone_name]["mask"] = loaded_info.get("mask") # Allow loading None mask
                     else:
                         print(f"Advertencia: Datos inválidos para la zona '{zone_name}' en el archivo guardado.")
            return roi_mask, zones # Return the (potentially updated) global vars
        except Exception as e:
            print(f"Error al cargar o procesar la configuración: {e}")
            # In case of error, reset globals to defaults
            # global zones, roi_mask # Remove redundant declaration
            # Reset to initial empty/None state (matching definitions at top of file)
            roi_mask = None 
            zones = {
                "IZQUIERDA": {"points": [], "mask": None, "color": (255, 0, 0)},
                "CENTRO": {"points": [], "mask": None, "color": (0, 255, 0)},
                "DERECHA": {"points": [], "mask": None, "color": (0, 0, 255)}
            }
            return None, None
    else:
        print(f"Archivo de configuración {ROI_SAVE_FILE} no encontrado.")
        return None, None

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
def define_roi(video_source, global_resolution=(640, 480)):
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
    frame = cv2.resize(frame, global_resolution)
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
    # No guardar aquí, guardar después de definir zonas
    # save_roi_data(roi_mask, zones) 
    return roi_mask

# Función para definir las zonas de interés (IZQUIERDA, CENTRO, DERECHA)
def define_zones(video_source, global_resolution=(640, 480)):
    global zones, current_zone, zone_drawing, zone_closed, zone_image
    
    # Capturar un frame del video para dibujar las zonas
    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo leer el video.")
        return None
    
    # Redimensionar para mantener consistencia
    frame = cv2.resize(frame, global_resolution)
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
    # Guardar ROI y Zonas juntas después de definir ambas
    save_roi_data(roi_mask, zones) 
    return True

# Función para determinar en qué zona se encuentra un punto
def get_zone_location(point):
    global zones
    for zone_name, zone_info in zones.items():
        if zone_info["mask"] is not None and zone_info["mask"][point[1], point[0]] > 0:
            return zone_name
    return None


def draw_roi_and_zones(frame, roi_mask, zones):
    roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, roi_contours, -1, (255, 255, 255), 2) # White ROI outline
    
    # Dibujar contornos de las zonas con sus nombres en annotated_frame
    for zone_name, zone_info in zones.items():
        if zone_info["mask"] is not None:
            zone_contours, _ = cv2.findContours(zone_info["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, zone_contours, -1, zone_info["color"], 2) # Zone color outline
            
            # Mostrar el nombre de la zona
            M = cv2.moments(zone_contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, zone_name, (cx - 40, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_info["color"], 2)
            # ---------------------------------------------
    return frame


def get_gpu_device():
    """
    Get the GPU device.
    
    if NVIDIA GPU is available,  return "gpu"
    if MPS (Apple Silicon GPU) is available, return "mps"
    otherwise, return "cpu"
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        return "cuda"
    else:
        return "cpu"