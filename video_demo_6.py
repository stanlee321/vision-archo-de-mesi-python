import cv2

from libs.utils import (define_roi, 
                        define_zones, 
                        get_zone_location, 
                        zones, 
                        load_roi_data, 
                        draw_roi_and_zones,
                        get_gpu_device)# Added load_roi_data


from arduino.sensor import ArduinoController
from libs.core import CoreDetector


PUERTO = 'COM3'

print(f"DEVICE: {get_gpu_device()}")
GLOBAL_RESOLUTION = (640, 480)
THRESHOLD = 0.3
DEVICE = get_gpu_device()
# VIDEO_SOURCE = "./video_prueba_1.mp4"
VIDEO_SOURCE = "./train/data/videos_v1/video_oficial.mp4"

# Modelo de detección
# best_11n.pt es el modelo de detección  mas pequeño (n de nano)
# best_11s.pt es el modelo de detección mas grande (s de small)

detector = CoreDetector(
    model_path="./train/best_ball_yolo11n.pt"
)

controlador = ArduinoController(port=PUERTO, baud_rate=9600, timeout=1)

if not controlador.connect():
    print("No se pudo establecer la conexión. Saliendo. :X")
    exit()


# Función principal
def main():
    # --- Ask user to load or define new ROI/Zones --- 
    loaded_roi_mask = None
    loaded_zones_data = None
    
    load_choice = input("¿Desea cargar la configuración de ROI/Zonas guardada? (s/n): ").lower()
    if load_choice == 's':
        loaded_roi_mask, loaded_zones_data = load_roi_data()
        if loaded_roi_mask is None or loaded_zones_data is None:
            print("No se pudo cargar la configuración. Procediendo a definir una nueva.")
        else:
             # Successfully loaded, update the global variables used by get_zone_location etc.
             # Note: define_roi and define_zones also modify globals, so this might be redundant
             # but ensures consistency if loading is successful.
             global zones # We need to modify the global zones dict from libs.utils
             zones = loaded_zones_data 
             roi_mask = loaded_roi_mask # Assign loaded mask
             print("Configuración cargada exitosamente.")
             
    # If not loaded or user chose 'n', define new ROI and Zones
    if loaded_roi_mask is None or loaded_zones_data is None:
        print("\n--- Definiendo nueva configuración de ROI y Zonas ---")
        # Definir la región de interés principal
        roi_mask = define_roi(VIDEO_SOURCE, global_resolution=GLOBAL_RESOLUTION)
        
        if roi_mask is None:
            print("No se definió una región de interés principal. Saliendo.")
            return # Exit if ROI definition cancelled
        
        # Definir las zonas de interés (IZQUIERDA, CENTRO, DERECHA)
        # Pass the already defined roi_mask to define_zones if needed for context?
        # Currently define_zones doesn't use it, but could be modified.
        if not define_zones(VIDEO_SOURCE, global_resolution=GLOBAL_RESOLUTION):
            print("No se definieron todas las zonas de interés. Saliendo.")
            return # Exit if zone definition cancelled
        print("Nueva configuración definida y guardada.")
        # The define_zones function already saves the new configuration
    # ------------------------------------------------------

    # Continuar con el procesamiento del video
    video = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Verificar si la cámara se abrió correctamente
    if not video.isOpened():
        print("Error: No se pudo abrir la cámara o el video.")
        exit()
    
    # Configurar la resolución del video
    video.set(cv2.CAP_PROP_FRAME_WIDTH, GLOBAL_RESOLUTION[0])
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, GLOBAL_RESOLUTION[1])
    
    print("Video inicializado correctamente")
   
    # Variable para trackear la última zona donde estuvo la pelota
    last_zone = None
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            # Reiniciar el video cuando llegue al final
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        print(f"DEVICE: {get_gpu_device()}")

        # Redimensionar el frame a GLOBAL_RESOLUTION
        frame = cv2.resize(frame, GLOBAL_RESOLUTION)
        
        # Aplicar máscara ROI al frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

        # Process frame with confidence threshold
        annotated_frame, centroids = detector.process_frame(masked_frame, confidence_threshold=THRESHOLD, device=DEVICE)
        
        # Reset current_zone for each frame before checking centroids
        current_zone = None 

        for centroid in centroids:
            center = (int(centroid[0]), int(centroid[1]))
            
            # Determinar en qué zona se encuentra el centro del círculo
            current_zone = get_zone_location(center)
            
            # Si está en una zona, dibujar información adicional
            if current_zone is not None:
                zone_color = zones[current_zone]["color"]
                cv2.putText(annotated_frame, f"ZONA: {current_zone}", (center[0]-60, center[1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                
                # Verificar si cambió de zona
                if current_zone != last_zone:
                    print(f"Pelota entró en zona: {current_zone}")
                    last_zone = current_zone
                    # Aquí implementa tu lógica condicional
                    if current_zone == "IZQUIERDA":
                        # Acciones para la zona IZQUIERDA
                        sent_to_arduino(current_zone)
                    elif current_zone == "CENTRO":
                        # Acciones para la zona CENTRO
                        sent_to_arduino(current_zone)
                    elif current_zone == "DERECHA":
                        # Acciones para la zona DERECHA
                        sent_to_arduino(current_zone)
        
        # ---- Draw ROI and Zones on Annotated Frame ----
        # Dibujar el contorno del ROI principal en annotated_frame
        annotated_frame = draw_roi_and_zones(annotated_frame, roi_mask, zones)

        # Mostrar el resultado final
        # cv2.imshow('Zonas y ROI', frame_with_overlays) # No longer needed
        cv2.imshow('Deteccion YOLO con Zonas', annotated_frame)

        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    video.release()
    cv2.destroyAllWindows()
    
    # Desconectar el Arduino
    controlador.disconnect() # Commented out as controlador is commented

def sent_to_arduino(zone: str,):
    if zone == "IZQUIERDA":
        controlador.move_to_sensor(1)
        controlador.move_to_sensor(2)
        controlador.move_to_sensor(3)
        print("Enviando zona IZQUIERDA al Arduino")
    elif zone == "CENTRO":
        controlador.move_to_sensor(4)
        print("Enviando zona CENTRO al Arduino")
    elif zone == "DERECHA":
        controlador.move_to_sensor(5)
        controlador.move_to_sensor(6)
        controlador.move_to_sensor(7)
        print("Enviando zona DERECHA al Arduino")
    print(f"Enviando zona {zone} al Arduino")
    # Aquí puedes agregar la lógica para enviar la zona al Arduino
    # RESERVADO PARA ARDUINO
    

# Ejecutar el programa
if __name__ == "__main__":
    main()
