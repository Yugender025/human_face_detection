import cv2

# Load the pre-trained Haar Cascade Classifier for human detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
# Load the image
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('Original image shape:', img.shape)
    print('Grayscale image shape:', gray.shape)

# Detect humans in the image
    face = face_cascade.detectMultiScale(gray, 1.1, 4)

# Print the number of humans detected
    print('Number of humans detected:', len(face))

# Draw a rectangle around the detected humans
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
