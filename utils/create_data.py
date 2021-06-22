"""
Функции для генерации данных для обучения нейронок
"""


import os
import importlib
from random import shuffle
from time import sleep
from PIL import Image, ImageDraw, ImageFont
from os.path import join, split
from numpy import array
import numpy as np


def create_data():

    """Генерирует картинки бкув одного шрифта"""

    from pygame.image import save
    from pygame import Surface
    import pygame

    pygame.init()

    font = pygame.font.Font(None, 40)
    window_size = (width, height) = 30, 30

    cropped = pygame.Surface((30, 30))
    for i in range(ord("а"), ord("а") + 34):
        text = font.render(chr(i), True, (0, 0, 0))
        cropped.fill((255, 255, 255))
        cropped.blit(text, [3, 5])
        # cropped.blit(screen, (0, 0), (0, 0, 30, 30))
        save(cropped, f"img/{chr(i)}.jpeg")


def create_data1():
    """Генерирует картинки бкув разных шрифтов"""

    fonts = [join(root, file) for root, _, files in os.walk(join(os.getcwd(), "fonts")) for file in files if
             file.endswith(".ttf")]
    fonts = {str(ind): ImageFont.truetype(i, size=26) for ind, i in enumerate(fonts)}
    for i in range(ord("а"), ord("а") + 34):
        for num, font in fonts.items():
            im = Image.new('RGB', (30, 30), color=(255, 255, 255))
            draw_text = ImageDraw.Draw(im)
            draw_text.text((3, 3), chr(i), font=font, fill=(0, 0, 0))
            im.save(f"img/{chr(i)}.{num}.jpeg")


# create_data()