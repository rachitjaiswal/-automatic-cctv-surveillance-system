import cv2

# Set up the video capture
video_capture = cv2.VideoCapture(0)

# Set up the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Define the video writer object
video_writer = cv2.VideoWriter("video.avi", cv2.VideoWriter_fourcc(*"MJPG"), 24, (640, 480))

while True:
    # Capture a frame from the video
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        video_writer.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Check for user input and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()


