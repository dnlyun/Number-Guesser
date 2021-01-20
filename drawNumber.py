try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pygame
    import tensorflow as tf

    from tkinter import *
    from tkinter import messagebox
except:
    pass


class Pixel(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (0, 0, 0)
        self.neighbours = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

    def get_neighbours(self, g):
        j = self.x // 20
        i = self.y // 20
        rows = 28
        cols = 28

        if i < cols - 1:
            self.neighbours.append(g.pixels[i + 1][j])
        if i > 0:
            self.neighbours.append(g.pixels[i - 1][j])
        if j < rows - 1:
            self.neighbours.append(g.pixels[i][j + 1])
        if j > 0:
            self.neighbours.append(g.pixels[i][j - 1])

        if i > 0 and j > 0:
            self.neighbours.append(g.pixels[i - 1][j - 1])
        if i > 0 and j < rows - 1:
            self.neighbours.append(g.pixels[i - 1][j + 1])
        if i < cols - 1 and j > 0:
            self.neighbours.append(g.pixels[i + 1][j - 1])
        if i < cols - 1 and j < rows - 1:
            self.neighbours.append(g.pixels[i + 1][j + 1])


class Grid(object):
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.width = width
        self.height = height
        self.add_pixel()
        pass

    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)

    def add_pixel(self):
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows
        self.pixels = []

        for r in range(self.rows):
            self.pixels.append([])
            for c in range(self.cols):
                self.pixels[r].append(Pixel(x_gap * c, y_gap * r, x_gap, y_gap))

        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].get_neighbours(self)

    def clicked(self, pos):
        try:
            t = pos[0]
            w = pos[1]
            g1 = int(t) // self.pixels[0][0].width
            g2 = int(w) // self.pixels[0][0].height

            return self.pixels[g2][g1]
        except:
            pass

    def convert_binary(self):
        li = self.pixels
        newMatrix = [[] for _ in range(len(li))]

        for i in range(len(li)):
            for j in range(len(li[i])):
                if li[i][j].color == (0, 0, 0):
                    newMatrix[i].append(0)
                else:
                    newMatrix[i].append(1)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
        x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
        x_train, x_test = x_train / 255., x_test / 255.

        for x in range(784):
            x_test[0][x] = newMatrix[x // 28][x % 28]

        return x_test[:1]


def guess(li):
    model = tf.keras.models.load_model('nn.model')

    predictions = model.predict(li)
    t = (np.argmax(predictions[0]))
    print("Prediction: " + str(t))


def main():
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (255, 255, 255)
                for n in clicked.neighbours:
                    n.color = (255, 255, 255)
            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked = g.clicked(pos)
                    clicked.color = (0, 0, 0)
                except:
                    pass
            if event.type == pygame.KEYDOWN:
                li = g.convert_binary()
                guess(li)
                g.add_pixel()

        g.draw(win)
        pygame.display.update()


pygame.init()
width = height = 560
win = pygame.display.set_mode((width, height))
g = Grid(28, 28, width, height)
main()

quit()
