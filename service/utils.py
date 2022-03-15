import os
import math
import time

import skimage
import skimage.measure as measure
import numpy as np
from PIL import Image
import torch


def cur_time_str():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())


def ensure_array(array):
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, torch.Tensor):
        return array.cpu().detach().numpy()
    return np.array(array)


def ensure_dir_exist(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def list_dir(path):
    return [os.path.join(path, filename) for filename in os.listdir(path)]


def device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def read_image(path):
    return Image.open(path).convert('RGB')


def lerp(a, b, coord):
    ratio = coord - math.floor(coord)
    return round(a * (1.0 - ratio) + b * ratio)


def clamp_number(num, a, b):
  return max(min(num, max(a, b)), min(a, b))


def bilinear(array, y, x):
    y1, x1 = math.floor(y), math.floor(x)
    y2, x2 = y1 + 1, x1 + 1
    if y2 >= array.shape[0]:
        y2 = y2 - 1
    if x2 >= array.shape[1]:
        x2 = x2 - 1
    left = lerp(array[y1, x1], array[y1, x2], x)
    right = lerp(array[y2, x1], array[y2, x2], x)
    return lerp(left, right, y)


def convert_mask_to_points(mask, retry=5):
    # 0 indicates background, thus remove 0
    labels = np.unique(mask)[1:]
    points = []
    for label in labels:
        instance_mask = (mask == label).astype('uint8')
        ys, xs = np.where(instance_mask)
        y_mean = ys.mean().round().astype('int')
        x_mean = xs.mean().round().astype('int')
        if instance_mask[y_mean, x_mean] != 0:
            points.append([y_mean, x_mean])
            continue
        y_center = y_mean
        x_center = x_mean
        found = False
        for i in range(retry):
            y = np.random.randint(-5, 6)
            x = np.random.randint(-5, 6)
            if instance_mask[y, x] != 0:
                points.append([y, x])
                found = True
                break
        if not found:
            indices = np.random.permutation(len(ys))
            y = ys[indices[0]].astype('int')
            x = xs[indices[0]].astype('int')
            points.append([y, x])
    return points


def extract_points_from_direction_field_map(direction_field_map, lambda1=0.6, step=1):
    binary_map = convert_direction_field_map_to_mask(direction_field_map, lambda1=lambda1, step=step)
    return convert_mask_to_points(measure.label(binary_map))


def create_magnitude_map(direction_field_map):
    return np.sqrt(np.square(direction_field_map[0]) + np.square(direction_field_map[1])).astype('float32')


def convert_direction_field_map_to_mask(direction_field_map, lambda1=0.6, step=1):
    magnitude_map = create_magnitude_map(direction_field_map)
    y_direction_map, x_direction_map = direction_field_map[0], direction_field_map[1]
    binary_map = np.zeros(direction_field_map[0].shape, dtype=int)
    ys, xs = np.where(magnitude_map > lambda1)
    for i in range(len(ys)):
        y = ys[i]
        x = xs[i]
        converged_point = [y, x]
        for s in range(step):
            tmp_y = converged_point[0] + y_direction_map[converged_point[0], converged_point[1]]
            tmp_x = converged_point[1] + x_direction_map[converged_point[0], converged_point[1]]
            if (tmp_y < 0 or tmp_y >= y_direction_map.shape[0]) or \
                    (tmp_x < 0 or tmp_x >= x_direction_map.shape[1]):
                break
            converged_point = [math.floor(tmp_y), math.floor(tmp_x)]
        binary_map[converged_point[0], converged_point[1]] = 1
    return binary_map


def extract_points_from_direction_field_map(direction_field_map, lambda1=0.7, step=10):
    binary_map = convert_direction_field_map_to_mask(direction_field_map, lambda1=lambda1, step=step)
    return convert_mask_to_points(measure.label(binary_map))


def convert_direction_field_map_to_mask(direction_field_map, lambda1=0.7, step=10):
    magnitude_map = create_magnitude_map(direction_field_map)
    y_direction_map, x_direction_map = direction_field_map[0], direction_field_map[1]

    y_direction_map[magnitude_map <= lambda1] = 0.0
    x_direction_map[magnitude_map <= lambda1] = 0.0

    binary_map = np.zeros(direction_field_map[0].shape, dtype=int)
    ys, xs = np.where(magnitude_map > lambda1)
    for i in range(len(ys)):
        y = ys[i]
        x = xs[i]
        converged_point = [y, x]
        for s in range(step):
            converged_point[0] = clamp_number(converged_point[0], 0, y_direction_map.shape[0] - 1)
            converged_point[1] = clamp_number(converged_point[1], 0, x_direction_map.shape[1] - 1)
            delta_y = bilinear(y_direction_map, converged_point[0], converged_point[1])
            delta_x = bilinear(x_direction_map, converged_point[0], converged_point[1])
            tmp_y = converged_point[0] + delta_y
            tmp_x = converged_point[1] + delta_x
            if (tmp_y < 0 or tmp_y >= y_direction_map.shape[0]) or \
                    (tmp_x < 0 or tmp_x >= x_direction_map.shape[1]):
                break
            converged_point = [tmp_y, tmp_x]
        y, x = math.floor(converged_point[0]), math.floor(converged_point[1])
        y = clamp_number(y, 0, y_direction_map.shape[0] - 1)
        x = clamp_number(x, 0, x_direction_map.shape[1] - 1)
        binary_map[y, x] = 1
    return binary_map
