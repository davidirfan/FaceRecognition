# take photos using picamera2 and opencv

from picamera2 import Picamera2
import cv2
import time

# Initialize the camera
picam2 = Picamera2()

# Configure the camera for preview
preview_config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "BGR888"})
picam2.configure(preview_config)

# Enable autofocus
picam2.set_controls({"AfMode": 2})  # Continuous autofocus mode

# Start the camera
picam2.start()

# Allow the camera to warm up
time.sleep(1)

photo_count = 0  # Counter for saved photos

# Capture frames continuously and display them
while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()
    
    # Rotate the image by 180 degrees
    # frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Convert the image from RGB to BGR (to fix the color issue)
    # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the frame using OpenCV
    cv2.imshow("Camera Preview", frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # If 's' is pressed, save the current frame as a photo
    if key == ord('s'):
        photo_filename = f"David_{photo_count}.jpg"
        cv2.imwrite(photo_filename, frame)
        print(f"Saved {photo_filename}")
        photo_count += 1

    # Break the loop on 'q' key press
    if key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
