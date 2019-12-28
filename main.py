from ann.sequential_model import Sequential
from ann.hidden_layer import HiddenLayer
from ann.activation import Linear
import json
from ann.image_operations import *
import numpy as np


# global variables
window_shape = (3, 3)
row_margin = window_shape[0] // 2
col_margin = window_shape[1] // 2

x_data = []
y_data = []
with open('../dataset/forest_photos_info.json', 'r') as fo:
    dataset = json.loads(fo.read())
    for i, image_key in enumerate(dataset):
        print(f"Processing Img: {i}")
        img = get_image_from_url(dataset[image_key]['url'])
        # img.show()
        # input()
        generate_data_set_from_image(img, x_data, y_data, window_shape)
        if i == 10:
            print(f"Generated data set from {i+1} images")
            break


def predict(model):
    # test image
    test_item = dataset.popitem()
    img = get_image_from_url(test_item[1]['url'])
    np_img = np.asarray(img)
    img.show()

    # gray scale
    gray_img = 0.21 * np_img[:, :, 0] + 0.72 * np_img[:, :, 1] + 0.07 * np_img[:, :, 2]
    gray_img = Image.fromarray(gray_img.astype(np.uint8))
    gray_img.show()
    # generate test data set
    x_test, y_test = [], []
    generate_data_set_from_image(img, x_test, y_test, window_shape)

    out = model.predict(x_test, y_test)
    np_out = np.zeros(np_img.shape - np.array([2, 2, 0]))
    k = 0
    for i in range(np_out.shape[0]):
        for j in range(np_out.shape[1]):
            np_out[i, j, ] = out[k] * 255
            k += 1
    out_img = Image.fromarray(np_out.astype(np.uint8))
    out_img.show()


model = Sequential()

model.add(HiddenLayer(units=18, prev_layer_dim=9))
model.add(HiddenLayer(units=9, prev_layer_dim=18))  # +1: bias node
model.add(HiddenLayer(units=3, prev_layer_dim=9, activation=Linear, is_output=True))

model.fit(x_data, y_data, rate=0.001, epoch=3)

predict(model)