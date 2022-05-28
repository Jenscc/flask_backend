import json
import uuid

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as T
from tensorflow import keras
from keras.applications.xception import preprocess_input

from service import utils
from service.model import UNet

localizationWeights = r'F:\demo\flask_backend\checkpoints\LUSC\model_epoch=15.pth.tar'
classifyModel = r'F:\demo\flask_backend\checkpoints\LUSC\my_model.h5'


def detect(srcImgPath):
    model = UNet(2).to(utils.device())
    model.load_state_dict(torch.load(localizationWeights, map_location=utils.device())['state_dict'])
    model.eval()

    image = utils.read_image(srcImgPath)
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        x = x.to(utils.device())
        pred = utils.ensure_array(model(x)).squeeze(0)
        points = utils.extract_points_from_direction_field_map(pred, lambda1=0.701, step=10)

    # save image and points
    image_array = np.array(image)
    plt.figure(dpi=500)
    plt.imshow(image_array)
    plt.axis('off')
    points_array = np.array(points)
    plt.plot(points_array[:, 1], points_array[:, 0], marker='o', markerfacecolor='#f9f738', markeredgecolor='none',
             markersize=2,
             linestyle='none')
    uid = str(uuid.uuid4())
    fileName = ''.join(uid.split('-')) + ".png"
    savePath = 'F:/demo/res/img/' + fileName
    plt.savefig(savePath, bbox_inches='tight', pad_inches=0)
    plt.close()
    torch.cuda.empty_cache()
    return fileName, image, points


def getPatch(image, point):
    xMax, yMax = image.size
    x, y = point
    l = (x - 30) if ((x - 30) > 0) else 0
    r = (x + 30) if ((x + 30) < xMax) else xMax
    h = (y - 30) if ((y - 30) > 0) else 0
    b = (y + 30) if ((y + 30) < yMax) else yMax
    box = (l, h, r, b)

    patch = image.crop(box)
    return patch


def classify(model, image):
    x = np.array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predict = model.predict(x)
    # print(predict)

    if round(predict[0, 0]) == 0:
        result = 'ying'
    else:
        result = 'yang'

    return result


def analysis(**kwargs):
    srcImgPath = kwargs.get('path')
    # 获取细胞位置
    fileName, image, points = detect(srcImgPath)
    # print(len(points))

    model = keras.models.load_model(classifyModel)
    yang = 0
    ying = 0
    for point in points:
        patch = getPatch(image, point)
        patch = tf.image.resize(patch, [80, 80])
        patch = tf.cast(patch, tf.float32)
        patch = patch / 255

        result = classify(model, patch)
        if result == 'yang':
            yang += 1
        else:
            ying += 1
    with open('./static/record.json', 'w') as recordF:
        dic = {'fileName': fileName, 'yang': yang, 'ying': ying}
        json.dump(dic, recordF)
