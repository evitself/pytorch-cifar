import os
import cv2
import numpy as np
import openvino.runtime as ov
from tqdm import tqdm

class OpenVinoInferenceModel():
    def __init__(self, model_file) -> None:
        self.core = ov.Core()
        self.model = self.core.compile_model(model_file)
    
    def infer(self, input_ndarray):
        inference = self.model.create_infer_request()
        ov_ret = inference.infer(input_ndarray)
        return [v for _, v in ov_ret.items()]


im_mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
im_std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


def read_one_image(im_file, convert_to_rgb=False):
    im = cv2.imread(im_file, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32)
    if convert_to_rgb:
        im = im[::-1]
    im = (im / 255. - im_mean) / im_std
    im = np.transpose(im, (2, 0, 1))
    return im


def test_batch(model, data, label):
    ret = model.infer(np.array(data))
    pred = np.argmax(ret, -1)
    correct = np.sum(pred == np.array(label))
    return correct, len(data)


def test_model(model_file, data_dir, labels, batch_size=9):
    model = OpenVinoInferenceModel(model_file)
    dataset = []
    for idx, label in enumerate(labels):
        for im_file in os.listdir(os.path.join(data_dir, label)):
            if not im_file.endswith('.png'):
                continue
            dataset.append((os.path.join(data_dir, label, im_file), idx))
    
    batch_data = []
    batch_label = []
    total_correct, total_count = 0, 0
    for im_file, label in tqdm(dataset, bar_format='{l_bar}{bar:32}{r_bar}{bar:-10b}'):
        data = read_one_image(im_file)
        batch_data.append(data)
        batch_label.append(label)
        if len(batch_data) == batch_size:
            correct, item_count = test_batch(model, batch_data, batch_label)
            total_correct += correct
            total_count += item_count
            batch_data = []
            batch_label = []
    if len(batch_data) > 0:
        correct, item_count = test_batch(model, batch_data, batch_label)
        total_correct += correct
        total_count += item_count
        batch_data = []
        batch_label = []
    print('Acc: %.3f%% (%d/%d)' % (100.*total_correct/total_count, total_correct, total_count))


if __name__ == '__main__':
    test_model('./resources/output2.xml', './data/cars_9_data', ['is_cars_9', 'not_cars_9'])
