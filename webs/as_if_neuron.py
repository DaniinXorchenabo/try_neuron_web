"""
Квази-нейронная сеть.
Задача: разпознавать букву, которую сеть увидела на картинке
Веса хранятся в виде фотографий (т.е. вес не может быть больше нуля)
Максимальная точность - 20%
"""

import os
from random import shuffle
from time import sleep
from os.path import join, split

from PIL import Image, ImageDraw, ImageFont
from numpy import array
import numpy as np


class Neuron:

    def __init__(self, target: str, directory="weights", extension="jpeg", size=(30, 30), k=0.3, **kwargs):
        self.__dict__ = kwargs
        self.target = target
        self.size = size
        self.k = k
        self.path_of_weight = join(directory, f"{target}.{extension}")
        try:
            self.weights = np.array(Image.open(self.path_of_weight).convert('1').getdata(), dtype=np.uint8).reshape(
                (size))
        except FileNotFoundError as e:
            self.weights = np.zeros(size, dtype=np.uint8)  # has to be unsigned bytes
            self.weights[:] = 255

    def save_weights(self):
        Image.fromarray(np.array([(j, j, j) for i in self.weights for j in i],
                                 dtype=np.uint8).reshape((*self.size, 3))).convert("RGB").save(self.path_of_weight)

    def test_img(self, testing_img):
        result = 0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (testing_img[i, j] < 250):
                    if (abs(self.weights[i, j] - testing_img[i, j]) < 120):
                        result += 1
                    # result += abs(self.weights[i, j] - testing_img[i, j]) ** 2

        return result

    def change_weights(self, testing_img, answer, real_result):
        change_f = lambda w, img: max(min(w + round(((img - w) if answer == real_result else (w - img)) * self.k), 255),
                                      0)
        my_round = lambda i: min(max(i, 0), 255)
        result = 0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (abs(self.weights[i, j] - testing_img[i, j]) < 120):
                    if (testing_img[i, j] < 250):
                        result += 1
                if self.weights[i, j] != 0 or (testing_img[i, j] != 0):
                    if (testing_img[i, j] < 250):
                        self.weights[i, j] = round(
                            (self.weights[i, j] + (self.weights[i, j] + testing_img[i, j]) / 2) / 2)
        return result

        # if real_result == answer and self.weights[i, j] > testing_img[i, j]:  # угадал
        #     self.weights[i, j] = my_round(self.weights[i, j] - (testing_img[i, j] - self.weights[i, j]) * self.k)
        # elif real_result == answer and self.weights[i, j] <= testing_img[i, j]:  # угадал
        #     self.weights[i, j] = my_round(self.weights[i, j] + (testing_img[i, j] - self.weights[i, j]) * self.k)
        # elif self.weights[i, j] > testing_img[i, j]:
        #     self.weights[i, j] = my_round(self.weights[i, j] - (255 - (testing_img[i, j] - self.weights[i, j])) * self.k)
        # else:
        #     self.weights[i, j] = my_round(self.weights[i, j] + (255 - (testing_img[i, j] - self.weights[i, j])) * self.k)
        # print(self.weights[i, j], end=' ')
        # print()

    @classmethod
    def learn(cls, web, real_ans, img):
        ans = max([(n, n.test_img(img)) for n in web], key=lambda i: i[1])
        # if ans[0].target != real_ans:
        [i.change_weights(img, False, True) for i in web if i.target == real_ans]
        # ans[0].change_weights(img, True, real_ans == ans[0].target)

    @classmethod
    def test_web(cls, web, real_ans, img):
        arr = [(n, n.test_img(img)) for n in web]
        ans = max(arr, key=lambda i: i[1])
        print([i[0].target + str(i[1])[:2] for i in arr if real_ans == i[0].target][0] + '-' + ans[0].target + str(
            ans[1])[:2], end=', ')
        return int(ans[0].target == real_ans)

    @staticmethod
    def learning_web():
        import os
        neuron_web = [Neuron(target=chr(i)) for i in range(ord("а"), ord("а") + 34)]
        data = [join(root, file) for root, _, files in os.walk(join(os.getcwd(), "img")) for file in files if
                file.endswith(".jpeg")]
        # print(*data, sep='\n')

        data = [[
            split(i)[1].split('.')[0],
            np.array(Image.open(i).convert('1').getdata(), dtype=np.uint8).reshape((30, 30))
        ] for i in data]
        # shuffle(data)
        # print("---- result is:", Neuron.testing(neuron_web, data[200]))
        for iteration in range(30):
            shuffle(data)
            for i in range(50):
                name, img = data[i]
                Neuron.learn(neuron_web, name, img)
                print(name, end=', ')
            print()
            shuffle(data)
            print(iteration, "---- result is:", Neuron.testing(neuron_web, data[:25]))

        [i.save_weights() for i in neuron_web]

    @staticmethod
    def testing(web, imgs):
        return sum([Neuron.test_web(web, name, img) for name, img in imgs]) / len(imgs)


Neuron.learning_web()