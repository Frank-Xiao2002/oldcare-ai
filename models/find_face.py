import cv2
import face_recognition


def mark_faces(image, process=True, face_locations=None):
    """
    Draw a rectangle around the face in the image
    :param face_locations:
    :param process:
    :param image: the image frame
    :return: the processed image
    """
    if process:
        face_locations = get_face_locations(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        image = cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    return image


def get_face_locations(image, model='hog'):
    """
    Get the face locations in the image.

    :param image: frame to be processed
    :return: A list of face locations, in the format of (top, right, bottom, left)
    """
    return face_recognition.face_locations(image, model=model)


def show_faces(source):
    cap = cv2.VideoCapture(source)
    face_locations = []
    counter1 = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print('Source open fail!')
            break
        if counter1 == 0:
            # do the process once every 5 frames
            face_locations = get_face_locations(img)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            img = cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        counter1 = (counter1 + 1) % 5
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


def show_original(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


def mark_faces_video(source):
    cap = cv2.VideoCapture(source)
    face_locations = []
    counter1 = 0

    # Initialize the video writer
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        if counter1 == 0:
            # do the process once every 5 frames
            face_locations = get_face_locations(img)
        img = mark_faces(img, process=False, face_locations=face_locations)
        counter1 = (counter1 + 1) % 5

        # Write the processed frame to the video
        out.write(img)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
