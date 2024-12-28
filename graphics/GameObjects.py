import pygame
import numpy as np
from typing import Tuple
from qlearner import DIRECTIONS

class Ball:
    VELOCITY = 1
    WIDTH = 1
    HEIGHT = 1

    def __init__(self, pos: Tuple[int, int]):
        self.direction = self.random_direction()
        self.initial_position = pos
        self.rect = pygame.Rect(pos[0], pos[1], self.WIDTH, self.HEIGHT)

    def random_direction(self) -> Tuple[int, int]:
        return DIRECTIONS[np.random.choice(list(DIRECTIONS.keys()))]

    def draw(self, canvas: pygame.Surface):
        pygame.draw.rect(canvas, pygame.Color("white"), self.rect)

    def set_position(self, x: int, y: int):
        self.rect.x = x
        self.rect.y = y

    def get_position(self) -> Tuple[int, int]:
        return self.rect.x, self.rect.y

    def in_paddle_area(self, paddle) -> bool:
        return paddle.get_position() <= self.rect.y <= paddle.get_position() + Paddle.LENGTH

    def reset(self):
        self.direction = self.random_direction()
        self.rect = pygame.Rect(self.initial_position[0], self.initial_position[1], self.WIDTH, self.HEIGHT)


class Paddle:
    VELOCITY = 1
    LENGTH = 5

    def __init__(self, pos: Tuple[int, int], color: str = "green"):
        self.initial_position = pos
        self.direction = 0
        self.color = color
        self.rect = pygame.Rect(pos[0], pos[1], 1, self.LENGTH)

    def set_position(self, y: int):
        self.rect.y = y

    def get_position(self) -> int:
        return self.rect.y

    def draw(self, canvas: pygame.Surface):
        pygame.draw.rect(canvas, pygame.Color(self.color), self.rect)

    def reset(self):
        self.direction = 0
        self.rect = pygame.Rect(self.initial_position[0], self.initial_position[1], 1, self.LENGTH)
