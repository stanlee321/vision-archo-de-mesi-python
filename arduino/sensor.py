import serial
import time
import threading
from typing import Optional

class ArduinoController:
    """
    Clase para controlar un Arduino con el sketch de sensores y modos,
    considerando la disposición física de los sensores S1-S7.
    """
    # --- Constantes descriptivas para los sensores (basadas en el diagrama) ---
    SENSOR_EXTREMO_IZQUIERDO = 1 # S1
    SENSOR_INTERMEDIO_IZQ_1 = 2  # S2
    SENSOR_INTERMEDIO_IZQ_2 = 3  # S3
    SENSOR_CENTRO = 4            # S4
    SENSOR_INTERMEDIO_DER_1 = 5  # S5
    SENSOR_INTERMEDIO_DER_2 = 6  # S6
    SENSOR_EXTREMO_DERECHO = 7   # S7

    def __init__(self, port: str, baud_rate: int = 9600, timeout: int = 1):
        """
        Inicializa la conexión serial y el hilo de lectura.
        (Sin cambios respecto a la versión anterior)
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        """Intenta conectar con el Arduino. (Sin cambios)"""
        if self.is_connected:
            print("Ya está conectado.")
            return True
    
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            print(f"Intentando conectar a {self.port}...")
            time.sleep(2)
            
            if self.ser.is_open:
                self.is_connected = True
                self._stop_event.clear()
                self._reader_thread = threading.Thread(target=self._read_from_arduino, daemon=True)
                self._reader_thread.start()
                print(f"✅ Conectado al Arduino en {self.port}")
                return True
            else:
                print(f"❌ No se pudo abrir el puerto {self.port}")
                self.ser = None
                return False
        except serial.SerialException as e:
            print(f"❌ Error al conectar con el Arduino: {e}")
            self.ser = None
            self.is_connected = False
            return False
        except Exception as e:
            print(f"❌ Error inesperado durante la conexión: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Cierra la conexión serial y detiene el hilo de lectura. (Sin cambios)"""
        if not self.is_connected or not self.ser:
            #print("No hay conexión activa.") # Comentado para reducir ruido si ya está desco
            return

        print("Desconectando del Arduino...")
        self._stop_event.set()
        if self._reader_thread and self._reader_thread.is_alive():
             self._reader_thread.join(timeout=2)

        try:
            if self.ser and self.ser.is_open: # Añadida verificación extra
                self.ser.close()
            print("🔌 Conexión cerrada.")
        except Exception as e:
            print(f"⚠️ Error al cerrar el puerto: {e}")
        finally:
            self.ser = None
            self.is_connected = False
            self._reader_thread = None


    def _send_command(self, command: str):
        """Método privado para enviar comandos al Arduino. (Sin cambios)"""
        if not self.is_connected or not self.ser:
            print("Error: No conectado al Arduino.")
            return
        try:
            full_command = (command.strip() + "\n").encode('utf-8')
            self.ser.write(full_command)
            # print(f"Enviado: {command}") # Descomentar para depuración
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"Error al enviar comando '{command}': {e}")
            self.disconnect() # Intentar desconectar si falla el envío
        except Exception as e:
             print(f"Error inesperado al enviar '{command}': {e}")

    def _read_from_arduino(self):
        """Método privado que corre en un hilo para leer datos del Arduino. (Sin cambios)"""
        print("Hilo de lectura iniciado.")
        while not self._stop_event.is_set():
            if not self.ser or not self.ser.is_open: # Salir si la conexión se pierde
                print("Hilo de lectura: Conexión perdida.")
                break
            try:
                if self.ser.in_waiting > 0:
                    # Leer línea por línea para procesar mensajes completos
                    try:
                        linea = self.ser.readline().decode('utf-8', errors='replace').strip()
                        if linea:
                            print(f"Arduino: {linea}")
                    except UnicodeDecodeError as ude:
                         print(f"Error de decodificación: {ude}. Datos crudos: {self.ser.read(self.ser.in_waiting)}") # Leer lo que quedó
            except serial.SerialException as e:
                 print(f"Error de lectura serial: {e}. Deteniendo hilo.")
                 self._stop_event.set()
                 # No llamar a disconnect aquí para evitar recursión si el error es en close()
                 self.is_connected = False
                 break # Salir del bucle
            except Exception as e:
                print(f"Error inesperado en hilo de lectura: {e}")
                time.sleep(0.1) # Pausa antes de reintentar

            # Pequeña pausa para no consumir CPU innecesariamente si no hay datos
            # Evita el busy-waiting extremo
            if self.ser.in_waiting == 0:
                time.sleep(0.05)

        print("Hilo de lectura detenido.")


    # --- Métodos para los Comandos del Arduino ---

    def set_mode_manual(self):
        """Cambia el Arduino al modo 1 (Manual S1-S7, RESET)."""
        self._send_command("MODE1")

    def set_mode_intermittent(self):
        """Cambia el Arduino al modo 2 (Intermitente S1-S7)."""
        self._send_command("MODE2")

    def stop_system(self):
        """Detiene cualquier operación y pone el Arduino en modo IDLE."""
        self._send_command("STOP")

    def pause_intermittent(self):
        """Pausa el movimiento en el modo 2 (Intermitente)."""
        self._send_command("PAUSE")

    def resume_intermittent(self):
        """Reanuda el movimiento en el modo 2 (Intermitente)."""
        self._send_command("RESUME")

    def move_to_sensor(self, sensor_number: int):
        """
        Envía el comando para mover el sistema al sensor físico especificado (S1-S7).
        El Arduino se encarga de la lógica de movimiento (dirección, parada)
        hasta alcanzar el sensor indicado por su número.
        Válido principalmente en Modo 1.

        Args:
            sensor_number (int): Número del sensor físico (1 a 7).
                                 Según el diagrama proporcionado:
                                 1: SENSOR_EXTREMO_IZQUIERDO
                                 2: SENSOR_INTERMEDIO_IZQ_1
                                 3: SENSOR_INTERMEDIO_IZQ_2
                                 4: SENSOR_CENTRO
                                 5: SENSOR_INTERMEDIO_DER_1
                                 6: SENSOR_INTERMEDIO_DER_2
                                 7: SENSOR_EXTREMO_DERECHO
        """
        # Puedes usar las constantes si quieres más claridad al llamar:
        # if sensor_number == self.SENSOR_EXTREMO_IZQUIERDO: ...
        # O simplemente validar el rango numérico:
        if self.SENSOR_EXTREMO_IZQUIERDO <= sensor_number <= self.SENSOR_EXTREMO_DERECHO:
            command = f"S{sensor_number}"
            print(f"Solicitando movimiento al sensor {sensor_number} (Comando: {command})")
            self._send_command(command)
        else:
            print(f"Error: Número de sensor inválido ({sensor_number}). Debe ser entre {self.SENSOR_EXTREMO_IZQUIERDO} y {self.SENSOR_EXTREMO_DERECHO}.")

    def reset_position(self):
        """Mueve el sistema a la posición central (Sensor S4). Válido en Modo 1."""
        print(f"Solicitando reseteo a posición central (Sensor {self.SENSOR_CENTRO})")
        # El comando "RESET" en el Arduino está programado para ir a S4.
        self._send_command("RESET")

# --- Parte Principal (Ejemplo de cómo usar la clase) ---
# La función main puede permanecer igual, ya que llama a move_to_sensor
# con el número correcto (1-7) que se mapea internamente al comando S1-S7.
# Podrías, si quisieras, modificar la entrada del usuario en main para usar
# las constantes, ej: input("Introduce comando (s1, ..., s7, reset, centro, extremo_izq): ")
# y luego mapearlo al número o método adecuado, pero no es estrictamente necesario.



def main():
    PUERTO = 'COM3' # <<<--- CAMBIA ESTO A TU PUERTO SERIAL CORRECTO
    # Linux: '/dev/ttyACM0', '/dev/ttyUSB0', etc.
    # macOS: '/dev/cu.usbmodemXXXX', '/dev/cu.usbserial-XXXX', etc.

    controlador = ArduinoController(port=PUERTO)
    
    if not controlador.connect():
        print("No se pudo establecer la conexión. Saliendo.")
        return

    try:
        while True:
            print("\n--- Menú de Control Arduino ---")
            print("Modos: mode1, mode2")
            print("Movimiento (en Modo 1): s1, s2, s3, s4 (centro), s5, s6, s7, reset")
            # Ejemplo usando constantes (opcional):
            # print(f"Movimiento: s{controlador.SENSOR_EXTREMO_IZQUIERDO} (izq), ..., s{controlador.SENSOR_CENTRO} (cen), ..., s{controlador.SENSOR_EXTREMO_DERECHO} (der), reset")
            print("Control (en Modo 2): pause, resume")
            print("General: stop, salir")
            
            comando = input("Introduce comando: ").strip().lower()

            if comando == "salir":
                break
            elif comando == "mode1":
                controlador.set_mode_manual()
            elif comando == "mode2":
                controlador.set_mode_intermittent()
            elif comando == "stop":
                controlador.stop_system()
            elif comando == "pause":
                controlador.pause_intermittent()
            elif comando == "resume":
                controlador.resume_intermittent()
            elif comando == "reset":
                controlador.reset_position() # s1
            elif comando.startswith("s") and len(comando) == 2 and comando[1].isdigit():
                try:
                    num_sensor = int(comando[1])
                    # La validación ahora está dentro del método move_to_sensor
                    controlador.move_to_sensor(num_sensor)
                except ValueError:
                    # Esto no debería ocurrir por la comprobación isdigit, pero por si acaso
                    print("Entrada de sensor inválida.")
            # Podrías añadir alias si usaras las constantes en el input:
            # elif comando == "centro":
            #     controlador.move_to_sensor(controlador.SENSOR_CENTRO)
            else:
                print(f"Comando '{comando}' no reconocido. Intenta de nuevo.")

    except KeyboardInterrupt:
        print("\nInterrupción por teclado detectada.")
    finally:
        print("Finalizando programa...")
        controlador.disconnect() # Asegura que la desconexión se intente

if __name__ == '__main__':
    main()