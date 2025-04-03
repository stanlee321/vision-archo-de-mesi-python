# Sistema de Detección y Seguimiento de Objetos con Zonas Inteligentes

Este proyecto implementa un sistema avanzado de visión artificial capaz de detectar objetos específicos (como una pelota roja) y determinar su ubicación en zonas personalizadas definidas por el usuario. El sistema puede aplicarse a una variedad de escenarios como seguimiento deportivo, automatización industrial, seguridad y más.

## Características Principales

- **Detección de Objetos por Color**: Enfocado en la detección de objetos por su color (predeterminado para color rojo) con rangos HSV ajustables.
- **Definición de ROI (Región de Interés)**: Permite al usuario definir manualmente una región poligonal para limitar el área de procesamiento.
- **Zonas Personalizables**: Capacidad para definir tres zonas distintas (IZQUIERDA, CENTRO, DERECHA) donde se monitoreará la presencia del objeto.
- **Detección de Cambios de Zona**: El sistema identifica cuando el objeto se mueve de una zona a otra y ejecuta acciones personalizadas.
- **Detección Robusta de Círculos**: Implementa dos métodos complementarios para la detección de formas circulares:
  - Transformada de Hough para círculos
  - Análisis de contornos con métricas de circularidad
- **Interfaz Visual Interactiva**: Muestra diferentes visualizaciones del procesamiento (imagen original, máscaras, contornos, detecciones).
- **Controles Ajustables en Tiempo Real**: Trackbars para ajustar parámetros de detección (HSV, tamaño mínimo, circularidad, etc.)
- **Comunicación Externa**: Posibilidad de integración con sistemas externos (como Arduino) en respuesta a la detección de zonas.

## Aplicaciones

- **Deportes**: Seguimiento de pelotas u objetos durante eventos deportivos.
- **Robótica**: Guía para robots que necesitan localizar y seguir objetos específicos.
- **Automatización Industrial**: Control de calidad y seguimiento de productos en líneas de producción.
- **Sistemas de Seguridad**: Monitoreo de áreas específicas y detección de intrusiones.
- **Análisis de Movimiento**: Estudio de patrones de movimiento en diversas aplicaciones.

## Funcionamiento

El sistema opera en varias fases secuenciales:

### 1. Definición de ROI Principal

El usuario dibuja un polígono para delimitar el área principal de interés. Esto ayuda a reducir el ruido y enfocarse solo en la región relevante.

### 2. Definición de Zonas

Después de la ROI principal, el usuario define tres zonas distintas (codificadas por colores):
- **IZQUIERDA** (Azul)
- **CENTRO** (Verde)
- **DERECHA** (Rojo)

Estas zonas se utilizan para determinar la ubicación espacial del objeto detectado.

### 3. Procesamiento y Detección

Una vez configuradas las zonas, el sistema:
1. Captura cada frame del video
2. Aplica la máscara de la ROI principal
3. Convierte la imagen a espacio de color HSV
4. Crea una máscara basada en los rangos de color configurados
5. Detecta objetos circulares usando dos métodos complementarios
6. Determina en qué zona se encuentra cada objeto detectado
7. Ejecuta acciones específicas cuando un objeto cambia de zona

### 4. Visualización y Feedback

El sistema muestra múltiples ventanas para facilitar la supervisión y ajuste:
- Imagen original
- Máscara HSV
- Contornos detectados
- Objetos finales detectados
- Vista de zonas y ROI

## Tecnologías Utilizadas

- **OpenCV**: Biblioteca principal para procesamiento de imágenes y visión artificial
- **NumPy**: Procesamiento eficiente de arrays para manipulación de imágenes
- **Python**: Lenguaje de programación base
- **Transformada de Hough**: Algoritmo para detección de formas geométricas
- **Detección de Contornos**: Técnicas de análisis morfológico

## Instalación y Requisitos

### Requisitos previos
- Python 3.6+
- OpenCV 4.x
- NumPy

### Instalación
```bash
pip install opencv-python numpy
```

## Uso

1. Ejecute el script principal:
```bash
python video_demo_4.py
```

2. Siga las instrucciones en pantalla para:
   - Dibujar la ROI principal (región de interés)
   - Definir las tres zonas (IZQUIERDA, CENTRO, DERECHA)

3. Ajuste los parámetros de detección usando los controles deslizantes para optimizar la detección según las condiciones específicas.

4. Para salir del programa, presione 'q'.

## Personalización

### Ajuste de Parámetros HSV

El sistema permite ajustar en tiempo real los rangos HSV para adaptar la detección a diferentes colores:
- Para objetos rojos: Use H=0-10 y 160-179 (el rojo está en ambos extremos del espectro H)
- Para objetos azules: Aproximadamente H=90-130
- Para objetos verdes: Aproximadamente H=40-80

### Integración con Sistemas Externos

El sistema incluye una función base `sent_to_arduino()` que puede expandirse para comunicarse con hardware externo:

```python
def sent_to_arduino(zone):
    print(f"Enviando zona {zone} al Arduino")
    # Implementar comunicación con Arduino u otro dispositivo
```

## Ampliaciones Futuras

- Seguimiento de múltiples objetos simultaneamente
- Comunicación bidireccional con sistemas de control
- Implementación de algoritmos de aprendizaje automático para mejorar la detección
- Interfaz gráfica de usuario completa
- Grabación y análisis de métricas temporales

## Contribución

Este proyecto es de código abierto y las contribuciones son bienvenidas. Las áreas de mejora incluyen:
- Optimización de rendimiento
- Soporte para detección de formas adicionales
- Mejoras en la interfaz de usuario
- Expansión a más plataformas

## Licencia

Este proyecto está disponible bajo la licencia MIT.

## Autor

Sistema desarrollado por [Tu Nombre] 