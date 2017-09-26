#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class Pic_analyz:
    def __init__(self):
        image_name = "tst2.png"
        self.pic = mpimg.imread(image_name)
        mpimg.imsave("orig.png", self.pic)

    def show_images(self):
        plt.subplot(241)
        plt.title("Default")
        plt.imshow(self.pic)

        self.to_grayscale()

        plt.subplot(242)
        plt.title("Grayscale")
        plt.imshow(self.grayscale)

        self.show_hists()
        #
        #  self.min_filter()
        #  print(self.minf)
        #  plt.subplot(gs[1, 1])
        #  plt.title("Min Filter")
        #  plt.imshow(self.min_filter)

        #  self.show_hists(gs)

    def to_grayscale(self):
        self.grayscale = np.zeros(self.pic.shape)
        i = 0
        j = 0
        for string in self.pic:
            for el in string:
                self.grayscale[i, j] = 0.3 * el[0] + 0.59 * el[1] + 0.11 * el[2]
                j += 1
            i += 1
            j = 0
        mpimg.imsave("grayscale.png", self.grayscale)

    def show_hists(self):
        plt.subplot(245)
        plt.title("Red Histogram")
        plt.hist(self.pic[:, :, 0].ravel(), bins=256, fc='k', ec='k')

        plt.subplot(246)
        plt.title("Green Histogram")
        plt.hist(self.pic[:, :, 1].ravel(), bins=256, fc='k', ec='k')

        plt.subplot(247)
        plt.title("Blue Histogram")
        plt.hist(self.pic[:, :, 2].ravel(), bins=256, fc='k', ec='k')

        plt.subplot(248)
        plt.title("Grayscale Histogram")
        plt.hist(self.grayscale[:, :, 0].ravel(), bins=256, fc='k', ec='k')

    def min_filter(self):
        y = 1
        self.minf = np.zeros((len(self.pic), len(self.pic[0]), len(self.pic[0][0])))
        self.minf[0] = self.pic[0]
        self.minf[len(self.minf) - 1] = self.pic[len(self.pic) - 1]
        r = list()
        g = list()
        b = list()
        while y < len(self.pic) - 1:
            self.minf[y][0] = self.pic[y][0]
            x = 1
            while x < len(self.pic[0]) - 1:
                print(y, x)
                mask = list()
                mask.append(self.pic[y + 1][x - 1])
                mask.append(self.pic[y + 1][x])
                mask.append(self.pic[y + 1][x - 1])
                mask.append(self.pic[y][x - 1])
                mask.append(self.pic[y][x + 1])
                mask.append(self.pic[y - 1][x - 1])
                mask.append(self.pic[y - 1][x])
                mask.append(self.pic[y - 1][x - 1])

                for el in mask:
                    r.append(el[0])
                    g.append(el[1])
                    b.append(el[2])
                self.minf[y][x][0] = np.min(mask, axis=0)
                self.minf[y][x][1] = np.min(mask, axis=1)
                self.minf[y][x][2] = np.min(mask, axis=2)
                x += 1
            self.minf[y][x] = self.pic[y][x]
            y += 1

    def show(self):
        self.show_images()
        plt.show()


if __name__ == "__main__":
    pic_analyz = Pic_analyz()
    pic_analyz.show()
