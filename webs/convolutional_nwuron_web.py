import os
from typing import Type, Any, Union
import importlib
from random import shuffle
from time import sleep
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

            self.weights = getattr(importlib.import_module(f"weights2.weights"), f"matrix_{self.number}")  #
            self.basis = getattr(importlib.import_module(f"weights2.weights"), f"matrix_basis_{self.number}")
            print("читаем из файла")
        except (ModuleNotFoundError, AttributeError) as e:

            # self.weights = np.zeros(synapses_size, dtype=np.float32)  # has to be unsigned bytes
            # self.weights[:] = np.random.random()
            self.weights = np.random.rand(*synapses_size).astype(np.float16) - 0.5
            self.basis = 0

            print('не читаем из файла', e)
            # print(self.weights)

    @classmethod
    def save_file_weights(cls):
        with open("weights2/weights.py", "w", encoding="utf-8") as f:
            print(cls.file_weights_content, file=f)

    def save_weights(self):

        WeightsMatrix.file_weights_content += f"matrix_{self.number} = np.asarray([" + ', \n'.join(
                ['[' + ", ".join([str(j) for j in i]) + ']' for i in self.weights]) + "], dtype=np.float16)\n"
        WeightsMatrix.file_weights_content += f"matrix_basis_{self.number} = {self.basis}\n\n"



class ConvolutionalNeuron:
    counter = 0

    def __init__(self, weights: WeightsMatrix, index: tuple):
        self.weights = weights.weights
        self.size = weights.size
        self.index = index

    def test_img(self, testing_img: np.array):
        setup_x = self.size[0] // 2
        setup_y = self.size[1] // 2
        on_y = np.zeros((setup_y, *testing_img.shape[1:]))
        on_x = np.zeros((testing_img.shape[0] + setup_y * 2, setup_x, *testing_img.shape[2:]))
        testing_img = np.concatenate((on_y, testing_img, on_y), 0)
        testing_img = np.concatenate((on_x, testing_img, on_x), 1)
        result = np.sum(self.weights * testing_img[self.index[0]:self.index[0] + self.size[0],
                                       self.index[1]:self.index[1] + self.size[1]])
        return result


filters = [WeightsMatrix([3, 3]) for i in range(16)]
one_layer = [[[ConvolutionalNeuron(filters[m], (i, j)) for j in range(6)]
             for i in range(6)] for m in range(16)]

arr = array([[1, 2, 3, 4, 5, 6],
             [9, 8, 7, 6, 5, 4],
             [2, 3, 4, 5, 6, 7],
             [3, 4, 5, 6, 7, 8],
             [8, 7, 6, 5, 4, 3],
             [1, 2, 3, 6, 7, 8]]) * 0.1

result = np.array([[[j.test_img(arr) for j in i] for i in m] for m in one_layer], dtype=np.float16)

[i.save_weights() for i in filters]
WeightsMatrix.save_file_weights()