import cv2
import numpy as np
import torch
from PIL import Image
from skimage import io
from skimage.transform import resize

import models.expression.transforms as transforms
from models.expression.VGG import VGG


def detect_and_mark_faces(raw_img):
    img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)

    # Load expression classifier
    net = VGG('VGG19')
    checkpoint = torch.load('../params/PrivateTest_model.t7',
                            map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    transform_test = transforms.Compose([
        transforms.TenCrop(44),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    for (x, y, w, h) in faces:
        # Preprocess the face
        face = raw_img[y:y + h, x:x + w]
        gray = np.dot(face[..., :3], [0.299, 0.587, 0.114])
        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
        face_img = gray[:, :, np.newaxis]
        face_img = np.concatenate((face_img, face_img, face_img), axis=2)
        face_img = Image.fromarray(face_img)
        inputs = transform_test(face_img)

        # Classify expression
        inputs = inputs.view(-1, 3, 44, 44).cuda()
        outputs = net(inputs)
        outputs_avg = outputs.view(-1, len(class_names)).mean(0)
        _, predicted = torch.max(outputs_avg.data, 0)
        expression = class_names[int(predicted.cpu().numpy())]

        # Mark the expression on the image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Convert back to RGB and display/save the image
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    io.imsave('../images/result.jpg', result_img)
    return result_img


def mark_expressions_from_live(source):
    global result
    counter = 0
    vc = cv2.VideoCapture(source)
    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            print("no live source:(")
            break
        if counter == 0:
            result = detect_and_mark_faces(frame)
        counter = (counter + 1) % 10
        cv2.imshow('live', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


mark_expressions_from_live('rtmp://play.live.frankxxj.top/oldcare/source1')