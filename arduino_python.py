import serial, time, threading

# Configuración del puerto serial y baud rate
PUERTO = 'COM3'  # Ejemplo para Windows; en Linux podría ser '/dev/ttyACM0'
BAUD_RATE = 9600

def leer_desde_arduino(ser):
    while True:
        try:
            if ser.in_waiting:
                linea = ser.readline().decode('utf-8', errors='replace').strip()
                if linea:
                    print("Arduino:", linea)
        except Exception as e:
            print("Error al leer:", e)
            break

def main():
    try:
        arduino = serial.Serial(PUERTO, BAUD_RATE, timeout=1)
        print("Conectado al Arduino en", PUERTO)
        time.sleep(2)  # Permite que el Arduino se reinicie
    except Exception as e:
        print("Error al conectar con el Arduino:", e)
        return

    # Iniciar un hilo para leer mensajes del Arduino
    hilo_lectura = threading.Thread(target=leer_desde_arduino, args=(arduino,), daemon=True)
    hilo_lectura.start()

    try:
        while True:
            print("\nIntroduce un comando para enviar:")
            print("Opciones: S1, S2, S3, S4, S5, S6, S7, RESET o SALIR para terminar")
            comando = input("Comando: ").strip().upper()
            if comando == "SALIR":
                print("Terminando el programa...")
                break
            if comando in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "RESET"]:
                arduino.write((comando + "\n").encode())
            else:
                print("Comando no reconocido. Vuelve a intentarlo.")
            time.sleep(0.1)  # Evita saturar el puerto serial
    except KeyboardInterrupt:
        print("Interrupción manual. Saliendo...")
    finally:
        arduino.close()

if __name__ == '__main__':
    main()