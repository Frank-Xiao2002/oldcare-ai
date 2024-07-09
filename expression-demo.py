import os
import numpy as np
import torch
from PIL import Image
from skimage import io
from skimage.transform import resize
from torch.autograd import Variable
import torch.nn.functional as F
import models.expression.transforms as transforms
from models.expression.VGG import VGG
import cv2

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def mark_face_expressions(img):
    raw_img = io.imread(img)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('params', 'PrivateTest_model.t7'),
                            map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()
    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)
    print('predicted = ', predicted)
    ind = 0.1 + 0.6 * np.arange(len(class_names))  # the x locations for the groups
    width = 0.4  # the width of the bars: can also be len(x) sequence
    print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))


# mark_face_expressions('detected_faces/face_0.jpg')


def detect_and_mark_faces(image_path, expression_model_path='params/PrivateTest_model.t7'):
    # Load the image
    raw_img = io.imread(image_path)
    img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)
    # print(faces)

    # Load expression classifier
    net = VGG('VGG19')
    checkpoint = torch.load(expression_model_path,
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
        print('predicted = ', predicted)
        expression = class_names[int(predicted.cpu().numpy())]
        print("The Expression is %s" % expression)

        # Mark the expression on the image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Convert back to RGB and display/save the image
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    io.imsave('marked_faces.jpg', result_img)


# Example usage
detect_and_mark_faces('images/smile.jpg')
