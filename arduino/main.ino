// Definición de pines
#define RUN_FORWARD 6   // IN1 del módulo de relés (RUN FORWARD)
#define RUN_REVERSE 7   // IN2 del módulo de relés (RUN REVERSE)

// Sensores inductivos (parte superior)
#define SENSOR_S1 2     // Extremo izquierdo
#define SENSOR_S2 3     // Intermedio izquierdo
#define SENSOR_S3 4     // Intermedio izquierdo
#define SENSOR_S4 5     // Centro
#define SENSOR_S5 8     // Intermedio derecho
#define SENSOR_S6 9     // Intermedio derecho
#define SENSOR_S7 10    // Extremo derecho

// Estados del sistema
enum SystemMode {
  MODE_IDLE,          // Esperando comando
  MODE_1,             // Modo manual original (tus comandos S1-S7, RESET)
  MODE_2_INTERMITTENT // Modo intermitente S1-S7
};

// Variables globales
SystemMode currentMode = MODE_IDLE;
bool systemRunning = false;
unsigned long lastMoveTime = 0;
const unsigned long INTERMITTENT_DELAY = 100; // 1 segundo entre movimientos

void setup() {
  Serial.begin(9600);
  
  // Configurar pines de motor
  pinMode(RUN_FORWARD, OUTPUT);
  pinMode(RUN_REVERSE, OUTPUT);
  
  // Configurar sensores con pull-up interno
  pinMode(SENSOR_S1, INPUT_PULLUP);
  pinMode(SENSOR_S2, INPUT_PULLUP);
  pinMode(SENSOR_S3, INPUT_PULLUP);
  pinMode(SENSOR_S4, INPUT_PULLUP);
  pinMode(SENSOR_S5, INPUT_PULLUP);
  pinMode(SENSOR_S6, INPUT_PULLUP);
  pinMode(SENSOR_S7, INPUT_PULLUP);

  detenerMotor();
  Serial.println("🔄 Posicionando en S4...");
  moverA(SENSOR_S4, "✅ Sistema inicializado en posición central (S4)");
  mostrarMenuInicial();
}

void loop() {
  manejarEntradaSerial();
  
  // Máquina de estados principal
  switch(currentMode) {
    case MODE_IDLE:
      // Esperando comando del usuario
      break;
      
    case MODE_1:
      // El modo 1 se maneja por comandos directos (tu código original)
      break;
      
    case MODE_2_INTERMITTENT:
      manejarModoIntermitente();
      break;
  }
}

void manejarEntradaSerial() {
  if (Serial.available() > 0) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();
    comando.toUpperCase();
    
    if (comando == "MODE1") {
      cambiarModo(MODE_1);
      Serial.println("🔧 Modo 1 activado: Control manual original");
      Serial.println("   Comandos disponibles: S1-S7, RESET, STOP");
    } 
    else if (comando == "MODE2") {
      cambiarModo(MODE_2_INTERMITTENT);
      Serial.println("🔄 Modo 2 activado: Movimiento intermitente S1-S7");
      Serial.println("   Comandos disponibles: STOP, PAUSE, RESUME");
    }
    else if (comando == "STOP") {
      detenerMotor();
      currentMode = MODE_IDLE;
      Serial.println("🛑 Sistema detenido. En modo IDLE");
      mostrarMenuInicial();
    }
    else if (comando == "PAUSE" && currentMode == MODE_2_INTERMITTENT) {
      detenerMotor();
      systemRunning = false;
      Serial.println("⏸️ Modo intermitente pausado. Use 'RESUME' para continuar");
    }
    else if (comando == "RESUME" && currentMode == MODE_2_INTERMITTENT) {
      systemRunning = true;
      Serial.println("▶️ Reanudando modo intermitente");
    }
    else if (currentMode == MODE_1) {
      // MANEJO DEL MODO 1 ORIGINAL (TU CÓDIGO)
      if (comando == "S1") {
        moverA(SENSOR_S1, "📍 Posición extrema izquierda alcanzada (S1)");
      } 
      else if (comando == "S2") {
        moverA(SENSOR_S2, "📍 Posición intermedia izquierda alcanzada (S2)");
      } 
      else if (comando == "S3") {
        moverA(SENSOR_S3, "📍 Posición intermedia izquierda alcanzada (S3)");
      }
      else if (comando == "S4") {
        moverA(SENSOR_S4, "📍 Posición central alcanzada (S4)");
      }
      else if (comando == "S5") {
        moverA(SENSOR_S5, "📍 Posición intermedia derecha alcanzada (S5)");
      }
      else if (comando == "S6") {
        moverA(SENSOR_S6, "📍 Posición intermedia derecha alcanzada (S6)");
      }
      else if (comando == "S7") {
        moverA(SENSOR_S7, "📍 Posición extrema derecha alcanzada (S7)");
      }
      else if (comando == "RESET") {
        Serial.println("🔄 Reiniciando sistema...");
        moverA(SENSOR_S4, "✅ Listo en posición central (S4)");
      }
      else {
        Serial.println("⚠️ Comando no reconocido. Use S1-S7, RESET o STOP");
      }
    }
    else {
      Serial.println("⚠️ Comando no reconocido o no disponible en este modo");
      mostrarAyuda();
    }
  }
}

void manejarModoIntermitente() {
  if (!systemRunning) return;
  
  if (millis() - lastMoveTime > INTERMITTENT_DELAY) {
    int sensorActual = obtenerSensorActivo();
    
    if (sensorActual == SENSOR_S1) {
      moverA(SENSOR_S7, "➡️ Moviendo a posición extrema derecha (S7)");
    } 
    else if (sensorActual == SENSOR_S7) {
      moverA(SENSOR_S1, "⬅️ Moviendo a posición extrema izquierda (S1)");
    }
    else {
      // Si por alguna razón no está en S1 o S7, ir a S1 primero
      moverA(SENSOR_S1, "⬅️ Posicionando en extremo izquierdo (S1)");
    }
    
    lastMoveTime = millis();
  }
}

void cambiarModo(SystemMode newMode) {
  detenerMotor();
  currentMode = newMode;
  systemRunning = (newMode == MODE_2_INTERMITTENT);
  lastMoveTime = millis();
}

// FUNCIONES ORIGINALES (TAL CUAL LAS TENÍAS)
void moverA(int sensorObjetivo, String mensaje) {
  Serial.println("🏃 Motor en movimiento...");
  int sensorActual = obtenerSensorActivo();

  if (sensorActual < sensorObjetivo) {
    digitalWrite(RUN_FORWARD, LOW);
    digitalWrite(RUN_REVERSE, HIGH);
  } 
  else if (sensorActual > sensorObjetivo) {
    digitalWrite(RUN_REVERSE, LOW);
    digitalWrite(RUN_FORWARD, HIGH);
  }

  while (digitalRead(sensorObjetivo) != LOW && currentMode != MODE_IDLE) {
    delay(10);
  }
  
  detenerMotor();
  Serial.println(mensaje);
}

void detenerMotor() {
  digitalWrite(RUN_FORWARD, HIGH);
  digitalWrite(RUN_REVERSE, HIGH);
}

int obtenerSensorActivo() {
  if (digitalRead(SENSOR_S1) == LOW) return SENSOR_S1;
  if (digitalRead(SENSOR_S2) == LOW) return SENSOR_S2;
  if (digitalRead(SENSOR_S3) == LOW) return SENSOR_S3;
  if (digitalRead(SENSOR_S4) == LOW) return SENSOR_S4;
  if (digitalRead(SENSOR_S5) == LOW) return SENSOR_S5;
  if (digitalRead(SENSOR_S6) == LOW) return SENSOR_S6;
  if (digitalRead(SENSOR_S7) == LOW) return SENSOR_S7;
  return -1;
}

void mostrarMenuInicial() {
  Serial.println("\n🛠️  COMANDOS DISPONIBLES:");
  Serial.println("------------------------");
  Serial.println("MODE1 - Activar modo manual original (S1-S7, RESET)");
  Serial.println("MODE2 - Activar modo intermitente (S1-S7 automático)");
  Serial.println("STOP  - Detener sistema y volver a modo IDLE");
  Serial.println("------------------------");
}

void mostrarAyuda() {
  switch(currentMode) {
    case MODE_IDLE:
      mostrarMenuInicial();
      break;
    case MODE_1:
      Serial.println("📌 Modo 1 - Comandos originales:");
      Serial.println("S1, S2, S3, S4, S5, S6, S7 - Mover a posición específica");
      Serial.println("RESET - Volver a posición central (S4)");
      Serial.println("STOP - Volver a modo IDLE");
      break;
    case MODE_2_INTERMITTENT:
      Serial.println("🔁 Modo 2 - Comandos intermitentes:");
      Serial.println("PAUSE - Pausar movimiento");
      Serial.println("RESUME - Reanudar movimiento");
      Serial.println("STOP - Volver a modo IDLE");
      break;
  }
}