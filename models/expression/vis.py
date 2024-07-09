import os

import numpy as np
import torch
from PIL import Image
from skimage import io
from skimage.transform import resize

import transforms as transforms
from torch.autograd import Variable

from models.expression.VGG import *

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


raw_img = io.imread(os.path.join('../../', 'images/1.jpg'))
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
inputs = inputs.cpu()
inputs = Variable(inputs, volatile=True)
outputs = net(inputs)

outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

score = F.softmax(outputs_avg)
_, predicted = torch.max(outputs_avg.data, 0)

print('predicted = ', predicted)

ind = 0.1 + 0.6 * np.arange(len(class_names))  # the x locations for the groups
width = 0.4  # the width of the bars: can also be len(x) sequence
color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']

emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))

print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))
