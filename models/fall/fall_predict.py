from ultralytics import YOLO
from PIL import Image

model = None


def load_fall_model():
    """
    Load the fall detection model
    :return:
    """
    global model
    if model is None:
        model = YOLO('runs/detect/train3/weights/best.pt')
    return model


def fall_predict(image_path):
    """
    Predict if there is a fall in the image
    :param image_path: path to the image
    :return: true if there is a fall, false otherwise
    """
    model = load_fall_model()
    im1 = Image.open(image_path)
    results = model.predict(source=im1, save=True)
    return len(results[0].boxes.cls.tolist()) > 0
