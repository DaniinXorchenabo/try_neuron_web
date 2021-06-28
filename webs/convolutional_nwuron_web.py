import os
from typing import Type, Any, Union
import importlib
from random import shuffle
from time import sleep, time
from PIL import Image, ImageDraw, ImageFont
from os.path import join, split
from numpy import array
import numpy as np


class BaseNeuron:

    def __init__(self,
                 limit=10,
                 directory="weights1",
                 size=(64, 64),
                 k: float = 0.3, **kwargs):
        self.__dict__ = kwargs
        self.size = size
        self.k = k
        self.limit = limit
        # self.path_of_weight = f"{directory}.{target}_weights.py"
        # self.path_of_weight = join(directory, f"{target}_weights.py")
        try:
            pass
            # self.weights = importlib.import_module(f"{directory}.{target}_weights").weights  #
            # print("читаем из файла")
        except ModuleNotFoundError as e:

            self.weights = np.zeros(size, dtype=np.float32)  # has to be unsigned bytes
            self.weights[:] = 0
            print('не читаем из файла', e)


class Layer:
    def __init__(self, neuron_class: Type[BaseNeuron],
                 size: tuple[int, int],
                 count: Union[int]):
        self.neurons = [neuron_class(size=size) for i in range(count)]


class Web:

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def learn(self):
        pass


# web = Web([Layer(BaseNeuron, (64, 64), 64**2),
#            Layer(BaseNeuron, (64, 64), 64**2),
#            Layer(BaseNeuron, (64, 64), 64**2)])
# web.learn()

class WeightsMatrix:
    counter = 0
    file_weights_content = "import numpy as np\n\n\n"

    def __init__(self, synapses_size: list[int]):
        WeightsMatrix.counter += 1
        self.number = WeightsMatrix.counter
        self.size = synapses_size
        try:

            self.weights: np.array = getattr(importlib.import_module(f"weights2.weights"), f"matrix_{self.number}")  #
            self.basis = getattr(importlib.import_module(f"webs.weights2.weights"), f"matrix_basis_{self.number}")
            # print("читаем из файла")
        except (ModuleNotFoundError, AttributeError, SyntaxError) as e:

            # self.weights = np.zeros(synapses_size, dtype=np.float32)  # has to be unsigned bytes
            # self.weights[:] = np.random.random()
            self.weights: np.array = np.random.rand(*synapses_size).astype(np.float16) - 0.5
            self.basis = 0

            print('не читаем из файла', e)
            # print(self.weights)

    @classmethod
    def save_file_weights(cls):
        with open("webs/weights2/weights.py", "w", encoding="utf-8") as f:
            print(cls.file_weights_content, file=f)

    def save_weights(self):
        WeightsMatrix.file_weights_content += f"matrix_{self.number} = np.asarray(["
        shape = self.weights.shape
        WeightsMatrix.file_weights_content += ', '.join([str(i) for i in self.weights.ravel()])
        WeightsMatrix.file_weights_content += f"], dtype=np.float16).reshape({', '.join([str(i) for i in shape])})\n"
        WeightsMatrix.file_weights_content += f"matrix_basis_{self.number} = {self.basis}\n\n"


class Neuron:

    def __init__(self, weights: WeightsMatrix):
        self.weights = weights.weights
        self.size = weights.size
        self.basis = weights.basis

    def test_img(self, testing_img: np.array):
        result = np.sum(self.weights * testing_img) + self.basis
        return result


class ConvolutionalNeuron:
    counter = 0

    def __init__(self, weights: WeightsMatrix, index: tuple):
        self.weights = weights.weights
        self.size = weights.size
        self.index = index
        self.basis = weights.basis

    def test_img(self, testing_img: np.array):
        setup_x = self.size[0] // 2
        setup_y = self.size[1] // 2
        on_y = np.zeros((setup_y, *testing_img.shape[1:]))
        on_x = np.zeros((testing_img.shape[0] + setup_y * 2, setup_x, *testing_img.shape[2:]))
        testing_img = np.concatenate((on_y, testing_img, on_y), 0)
        testing_img = np.concatenate((on_x, testing_img, on_x), 1)
        result = np.sum(self.weights * testing_img[self.index[0]:self.index[0] + self.size[0],
                                       self.index[1]:self.index[1] + self.size[1]]) + self.basis
        return result


def layer_iteration(layer, data):
    return np.array([[[j.test_img(data) for j in i] for i in m] for m in layer], dtype=np.float16).reshape(
        *data.shape[:2], -1)


def pool_iteration(data: np.array):
    # data = data[:, :, 0]
    # shape = data.shape

    return np.transpose(np.array(
        [
            [
                [
                    np.max(data[i:i + 2, j:j + 2, m])
                    for j in range(0, data.shape[1], 2)
                ]
                for i in range(0, data.shape[0], 2)
            ]
            for m in range(data.shape[-1])
        ],
        dtype=np.float16), (1, 2, 0))  # .reshape(data.shape[0]//2, data.shape[1]//2, *data.shape[2:])


def full_layer_iterator(layer, data):
    return np.array([m.test_img(data) for m in layer], dtype=np.float16)


def get_layer(filters, data_size: tuple[int, int]):
    return [[[ConvolutionalNeuron(matrix, (i, j, *matrix.size[2:])) for j in range(data_size[1])]
             for i in range(data_size[0])] for matrix in filters]


def get_full_layer(filters):
    return [Neuron(matrix) for matrix in filters]


# =======! DATA !=======
arr = array([[1, 2, 3, 4, 5, 6],
             [9, 8, 7, 6, 5, 4],
             [2, 3, 4, 5, 6, 7],
             [3, 4, 5, 6, 7, 8],
             [8, 7, 6, 5, 4, 3],
             [1, 2, 3, 6, 7, 8]]) * 0.1

print(np.array([False, True, False]).astype(np.int32))

#
# arr = np.transpose(np.stack((arr, arr.copy()[::-1], arr.copy()[::, ::-1])), (1, 2, 0))
# data = [join(root, file) for root, _, files in os.walk(join(os.getcwd(), "data", "img32")) for file in files if
#         file.endswith(".jpeg")]
# # print(*data, sep='\n')
#
# data = [np.array(Image.open(i)) / 255 for i in data]
# result = data[0]
#
# # =======! LAYERS !=======
# filters = [[WeightsMatrix([3, 3, 3]) for _ in range(8)],
#            # [WeightsMatrix([3, 3, 16]) for _ in range(16)],
#            [WeightsMatrix([3, 3, 8]) for _ in range(16)],
#            # [WeightsMatrix([3, 3, 32]) for _ in range(32)],
#            [WeightsMatrix([3, 3, 16]) for _ in range(32)],
#            # [WeightsMatrix([3, 3, 64]) for _ in range(64)],
#            [WeightsMatrix([3, 3, 32]) for _ in range(64)],
#            # [WeightsMatrix([3, 3, 128]) for _ in range(128)],
#            [WeightsMatrix([4, 4, 64]) for _ in range(1000)],
#            [WeightsMatrix([1000]) for _ in range(1000)],
#            [WeightsMatrix([1000]) for _ in range(32)],
#            ]
#
# first_layer = get_layer(filters[0], (32, 32))
# # second_layer = get_layer(filters[1], (32, 32))
# third_layer = get_layer(filters[1], (16, 16))
# # fourth_layer = get_layer(filters[3], (16, 16))
# fifth_layer = get_layer(filters[2], (8, 8))
# # sixth_layer = get_layer(filters[5], (8, 8))
# seventh_layer = get_layer(filters[3], (4, 4))
# # eleventh_layer = get_layer(filters[7], (4, 4))
# ninth_layer = get_full_layer(filters[4])
# tenth_layer = get_full_layer(filters[5])
# eleventh_layer = get_full_layer(filters[6])
# # =======! Go neuron web !=======
#
# start_time = time()
# result = layer_iteration(first_layer, result)
# result = np.maximum(result, 0)  # ReLU
# t_01 = time()
# # result = layer_iteration(second_layer, result)
# # result = np.maximum(result, 0)
# result = pool_iteration(result)
# t_1 = time()
#
# result = layer_iteration(third_layer, result)
# result = np.maximum(result, 0)  # ReLU
# # result = layer_iteration(fourth_layer, result)
# # result = np.maximum(result, 0)
# t_12 = time()
# result = pool_iteration(result)
# t_2 = time()
#
# result = layer_iteration(fifth_layer, result)
# result = np.maximum(result, 0)  # ReLU
# # result = layer_iteration(sixth_layer, result)
# # result = np.maximum(result, 0)
# t_22 = time()
# result = pool_iteration(result)
# t_3 = time()
#
# result = layer_iteration(seventh_layer, result)
# result = np.maximum(result, 0)  # ReLU
# t_4 = time()
# result = full_layer_iterator(ninth_layer, result)
# result = full_layer_iterator(tenth_layer, result)
# result = full_layer_iterator(eleventh_layer, result)
# result = np.exp(resultdddd)
# # result = layer_iteration(eleventh_layer, result)
# # result = np.maximum(result, 0)
# # result = pool_iteration(result)
# end_time = time()
# print(result)
# times = [end_time - start_time, t_01 - start_time, t_1 - t_01, t_12 - t_1, t_2 - t_12, t_22 - t_2, t_3 - t_22,
#          t_4 - t_3]
# print(*[round(i, 3) for i in times])
# # =======! Save weights !=======
# [i.save_weights() for layer_filters in filters for i in layer_filters]
# WeightsMatrix.save_file_weights()
