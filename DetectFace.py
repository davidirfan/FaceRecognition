# alhamdulillah, worked, tapi run di terminal

# import the necessary packages
from picamera2 import Picamera2
import face_recognition
import pickle
import time
import cv2

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"

# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize Picamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (848, 480), "format": "BGR888"})
picam2.configure(preview_config)
# Enable autofocus
picam2.set_controls({"AfMode": 2})  # Continuous autofocus mode
picam2.start()

time.sleep(2.0)  # Allow time for camera warm-up

# Initialize FPS counter
fps_counter = 0
start_time = time.time()

# Loop over frames from the camera
while True:
    # Grab a frame from the camera
    frame = picam2.capture_array()

    # Convert the frame from BGR (OpenCV format) to RGB (dlib/face_recognition format)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    boxes = face_recognition.face_locations(rgb)

    # Compute the facial embeddings for each detected face
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop over each face encoding
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # Check if there’s a match
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Identify the face with the highest count
            name = max(counts, key=counts.get)

            # If a recognized face is identified, print the name
            if currentname != name:
                currentname = name
                print(currentname)

        # Add the name to the list of recognized faces
        names.append(name)

    # Draw bounding boxes and names around detected faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

	# FPS Calculation
    fps_counter += 1
    elapsed_time = time.time() - start_time
    fps = fps_counter / elapsed_time

    # Display the FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Facial Recognition", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
