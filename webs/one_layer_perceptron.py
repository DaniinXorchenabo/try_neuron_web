"""
Однослойный перцептрон для распознования картинок (буквы)
Максимальная точность: 98%
Веса хранятся в питонячьем файле
"""

import os
import importlib
from random import shuffle
from time import sleep
from PIL import Image, ImageDraw, ImageFont
from os.path import join, split
from numpy import array
import numpy as np


class NaturalNeuron:

    def __init__(self, target: str, limit=10,
                 directory="weights1",
                 size=(30, 30),
                 k=0.3, **kwargs):
        self.__dict__ = kwargs
        self.target = target
        self.size = size
        self.k = k
        self.limit = limit
        self.path_of_weight = f"{directory}.{target}_weights.py"
        self.path_of_weight = join(directory, f"{target}_weights.py")
        try:
            self.weights = importlib.import_module(f"{directory}.{target}_weights").weights  #
            # print("читаем из файла")
        except ModuleNotFoundError as e:

            self.weights = np.zeros(size, dtype=np.float32)  # has to be unsigned bytes
            self.weights[:] = 0
            print('не читаем из файла', e)

    def save_weights(self):
        with open(self.path_of_weight, "w", encoding="utf-8") as f:
            print("import numpy as np\n\n\nweights = np.asarray([", ', \n'.join(
                ['[' + ", ".join([str(j) for j in i]) + ']' for i in self.weights]),
                  end="], dtype=np.float32)\n", sep='\n', file=f)

    def test_img(self, testing_img: np.array):
        result = np.sum(self.weights * testing_img)
        return 1 if result >= self.limit else 0

    def change_weights(self, testing_img, answer, real_result):
        result = 0
        if answer != real_result and not answer:
            self.weights += testing_img
        elif answer != real_result and answer:
            self.weights -= testing_img

        return result

    def load_img(self, path):
        return (
            split(path)[1].split('.')[0],
            np.asarray(np.array(Image.open(path).convert('1').getdata(), dtype=np.uint8) > 100,
                       dtype=np.uint8).reshape(self.size)
        )

    @classmethod
    def learn(cls, web, real_ans, img):
        ans = max([(n, n.test_img(img)) for n in web], key=lambda i: i[1])
        # if ans[0].target != real_ans:
        if real_ans != ans[0].target:
            [i.change_weights(img, False, True) for i in web if i.target == real_ans]
            ans[0].change_weights(img, True, real_ans == ans[0].target)

    @classmethod
    def test_web(cls, web, real_ans, img):
        arr = [(n, n.test_img(img)) for n in web]
        ans = max(arr, key=lambda i: i[1])
        # print([i[0].target + str(i[1]) for i in arr if real_ans == i[0].target][0] + '-' + ans[0].target + str(ans[1]), end=', ')
        return int(ans[0].target == real_ans)

    @staticmethod
    def learning_web():
        import os
        neuron_web = [NaturalNeuron(target=chr(i)) for i in range(ord("а"), ord("а") + 34)]
        data = [join(root, file) for root, _, files in os.walk(join(os.getcwd(), "img")) for file in files if
                file.endswith(".jpeg")]
        # print(*data, sep='\n')

        data = [neuron_web[0].load_img(i) for i in data]

        for iteration in range(100):
            shuffle(data)
            for i in range(50):
                name, img = data[i]
                NaturalNeuron.learn(neuron_web, name, img)
                # print(name, end=', ')
            # print()
            shuffle(data)
            print(iteration, "---- result is:", NaturalNeuron.testing(neuron_web, data[:25]))

        [i.save_weights() for i in neuron_web]
        print(NaturalNeuron.testing(neuron_web, data))

    @staticmethod
    def testing(web, imgs):
        return sum([NaturalNeuron.test_web(web, name, img) for name, img in imgs]) / len(imgs)


NaturalNeuron.learning_web()