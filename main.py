#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np


class Pic_analyz:
    def __init__(self):
        self.gs = gridspec.GridSpec(12, 4)
        self.pic = mpimg.imread("lenna.png")
        mpimg.imsave("orig.png", self.pic)

        self.fmin = 0.2
        self.fmax = 0.8
        self.gmin = 0.2
        self.gmax = 0.8

    def show_images(self):
        # default
        plt.subplot(self.gs[0, 0])
        plt.title("Default")
        plt.imshow(self.pic)

        self.to_grayscale()

        plt.subplot(self.gs[0, 1])
        plt.title("Grayscale")
        plt.imshow(self.grayscale)

        self.show_hists(1, self.pic, self.grayscale)

        # preparation D
        prepd = self.linear_contr(self.pic, fmin=self.fmin, fmax=self.fmax)
        plt.subplot(self.gs[2, 0])
        plt.title("Praparation D")
        plt.imshow(prepd)
        mpimg.imsave("prepd.png", prepd)

        prepd_gray = self.linear_contr(self.grayscale, fmin=self.fmin, fmax=self.fmax)
        plt.subplot(self.gs[2, 1])
        plt.title("Praparation D")
        plt.imshow(prepd_gray)
        mpimg.imsave("prepd_gray.png", prepd_gray)

        self.show_hists(3, prepd, prepd_gray)

        # preparation E
        prepe = self.linear_contr(self.pic, gmin=self.gmin, gmax=self.gmax)
        plt.subplot(self.gs[4, 0])
        plt.title("Praparation E")
        plt.imshow(prepe)
        mpimg.imsave("prepe.png", prepe)

        prepe_gray = self.linear_contr(self.grayscale, gmin=self.gmin, gmax=self.gmax)
        plt.subplot(self.gs[4, 1])
        plt.title("Praparation E")
        plt.imshow(prepe_gray)
        mpimg.imsave("prepe_gray.png", prepe_gray)

        self.show_hists(5, prepe, prepe_gray)

        #  # min filter
        #  minf = self.min_filter(self.pic)
        #  plt.subplot(self.gs[4, 0])
        #  plt.title("Min filter")
        #  plt.imshow(minf)
        #  mpimg.imsave("minf.png", minf)
        #
        #  minf_gray = self.min_filter(self.grayscale)
        #  plt.subplot(self.gs[4, 1])
        #  plt.title("Min filter")
        #  plt.imshow(minf_gray)
        #  mpimg.imsave("minf_gray.png", minf_gray)
        #
        #  self.show_hists(5, minf, minf_gray)
        #
        #  # max filter
        #  maxf = self.max_filter(self.pic)
        #  plt.subplot(self.gs[6, 0])
        #  plt.title("Max filter")
        #  plt.imshow(maxf)
        #  mpimg.imsave("maxf.png", maxf)
        #
        #  maxf_gray = self.max_filter(self.grayscale)
        #  plt.subplot(self.gs[6, 1])
        #  plt.title("Max filter")
        #  plt.imshow(maxf_gray)
        #  mpimg.imsave("maxf_gray.png", maxf_gray)
        #
        #  self.show_hists(7, maxf, maxf_gray)
        #
        #  # min-max filter
        #  min_maxf = self.min_max_filter(self.pic)
        #  plt.subplot(self.gs[8, 0])
        #  plt.title("Min-max filter")
        #  plt.imshow(min_maxf)
        #  mpimg.imsave("min_maxf.png", min_maxf)
        #
        #  min_maxf_gray = self.min_max_filter(self.grayscale)
        #  plt.subplot(self.gs[8, 1])
        #  plt.title("Min-max filter")
        #  plt.imshow(min_maxf_gray)
        #  mpimg.imsave("min_maxf_gray.png", min_maxf_gray)
        #
        #  self.show_hists(9, min_maxf, min_maxf_gray)

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

    def linear_contr(self, src, fmin=0.0, fmax=1.0, gmin=0.0, gmax=1.0):
        dst = np.copy(src)
        for y in range(1, src.shape[0]):
            for x in range(1, src.shape[1]):
                dst[y, x, 0] = (src[y, x, 0] - fmin) / (fmax - fmin) * (gmax - gmin) + gmin
                dst[y, x, 1] = (src[y, x, 1] - fmin) / (fmax - fmin) * (gmax - gmin) + gmin
                dst[y, x, 2] = (src[y, x, 2] - fmin) / (fmax - fmin) * (gmax - gmin) + gmin
        return dst

    def show_hists(self, line, orig, gray):
        plt.subplot(self.gs[line, 0])
        plt.title("Red Histogram")
        plt.hist(orig[:, :, 0].ravel(), bins=256, fc='k', ec='k')

        plt.subplot(self.gs[line, 1])
        plt.title("Green Histogram")
        plt.hist(orig[:, :, 1].ravel(), bins=256, fc='k', ec='k')

        plt.subplot(self.gs[line, 2])
        plt.title("Blue Histogram")
        plt.hist(orig[:, :, 2].ravel(), bins=256, fc='k', ec='k')

        plt.subplot(self.gs[line, 3])
        plt.title("Grayscale Histogram")
        plt.hist(gray[:, :, 0].ravel(), bins=256, fc='k', ec='k')

    def min_filter(self, src):
        dst = np.copy(src)
        for y in range(1, src.shape[0]):
            for x in range(1, src.shape[1]):
                dst[y, x, 0] = np.min(src[y-1:y+1, x-1:x+1, 0])
                dst[y, x, 1] = np.min(src[y-1:y+1, x-1:x+1, 1])
                dst[y, x, 2] = np.min(src[y-1:y+1, x-1:x+1, 2])
        return dst

    def max_filter(self, src):
        dst = np.copy(src)
        for y in range(1, src.shape[0]):
            for x in range(1, src.shape[1]):
                dst[y, x, 0] = np.max(src[y-1:y+1, x-1:x+1, 0])
                dst[y, x, 1] = np.max(src[y-1:y+1, x-1:x+1, 1])
                dst[y, x, 2] = np.max(src[y-1:y+1, x-1:x+1, 2])
        return dst

    def min_max_filter(self, src):
        tmp = self.min_filter(src)
        return self.max_filter(tmp)

    def show(self):
        self.show_images()
        plt.show()


if __name__ == "__main__":
    pic_analyz = Pic_analyz()
    pic_analyz.show()
