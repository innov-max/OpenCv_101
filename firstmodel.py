import cv2

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('/home/max/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('/home/max/Desktop/OpenCv/test.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with the detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
