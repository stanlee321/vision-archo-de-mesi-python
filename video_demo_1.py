# Demo for video processing

import cv2
import numpy as np
import time

# Load the video
video = cv2.VideoCapture(0)

# Check if the video was opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Set the video to 60 FPS
video.set(cv2.CAP_PROP_FPS, 60)

while True:
    
    start_time = time.time()
    
    ret, image = video.read()
    print("ORIGINAL SHAPE")
    print(image.shape)
    if not ret:
        break
    
    # Resize the frame
    frame = cv2.resize(image, (320, 240)) # Tuple of width and height
    print("RESIZED SHAPE")
    print(frame.shape)
    # Convert the image to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # do blur to reduce noise
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    # Define range for red color detection
    # Red has hue around 0 or 180 in HSV, so we need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red detection
    mask1 = cv2.inRange(blur, lower_red1, upper_red1)
    mask2 = cv2.inRange(blur, lower_red2, upper_red2)
    
    # Combine the masks
    red_mask = mask1 + mask2
    
    # Apply the mask to the resized frame instead of original image
    red_only = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Show the original frame and the red detection
    cv2.imshow('Original', frame)
    cv2.imshow('Red Detection', red_only)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    # In FPS mode
    fps = 1 / (end_time - start_time)
    
    print(f"FPS: {fps}")
    
    if fps > 60:
        # DO SOMETHING WITH ARDUINO
        pass
        # my_function_que_activa_el_motor()
    else:
        # DO SOMETHING WITH ARDUINO
        pass
        # my_function_que_desactiva_el_motor()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
