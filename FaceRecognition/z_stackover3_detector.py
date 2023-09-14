from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw

import numpy as np # MY ATTEMPTED FIX

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue" # NEW ENTRY
TEXT_COLOR = "white" # NEW ENTRY

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

# BLOCK NEW ENTRY: FUNCTION '_display_face()'
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )
########################  END OF BLOCK NEW ENTRY: FUNCTION '_display_face()'

def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
        ) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

encode_known_faces()

def recognize_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
        ) -> None:
    
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    open_image = Image.open(image_location) # MY ATTEMPTED FIX
    image_array = np.asarray(open_image) # MY ATTEMPTED FIX
    pillow_image = Image.fromarray(image_array) # NEW LINE IN VERSION 2 AND AMENDED BY ME IN VERSION 3
    draw = ImageDraw.Draw(pillow_image) # NEW LINE
    
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        #print(name, bounding_box) REMOVED
        _display_face(draw, bounding_box, name) # NEW LINE

    del draw # NEW LINE
    pillow_image.show() # NEW LINE

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    
recognize_faces(input('Enter image file name and its directory : '))