# alhamdulillah, worked tapi run di terminal

# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# define the dataset folder
dataset_folder = "faceDataset"

# check if the dataset folder exists
if not os.path.exists(dataset_folder):
    print(f"[ERROR] Dataset folder {dataset_folder} not found!")
    exit()

# start processing the faces
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images(dataset_folder))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    
    # Extract person name from folder name
    name = imagePath.split(os.path.sep)[-2]
    
    # load the input image and convert it from BGR (OpenCV default) to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect the face locations using HOG model
    boxes = face_recognition.face_locations(rgb, model="hog")
    
    # compute the facial embeddings
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    # loop over the encodings
    for encoding in encodings:
        # add the encoding and the corresponding name to the lists
        knownEncodings.append(encoding)
        knownNames.append(name)

# save the encodings and names to a pickle file
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] encoding complete and saved to encodings.pickle")
