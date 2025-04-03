# Import necessary libraries
import cv2

# Load the image
image = cv2.imread('test.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
# blur = cv2.GaussianBlur(gray, (5, 5), 100)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Display the original and edge-detected images
cv2.imshow('Original Image', image)
cv2.imshow('Edge-Detected Image', edges)

# exit the program
cv2.waitKey(0)
cv2.destroyAllWindows()
