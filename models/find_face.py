import cv2
import face_recognition


def circle_faces(image):
    face_locations = get_face_locations(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        image = cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    return image

def get_face_locations(image):
    return face_recognition.face_locations(image)
