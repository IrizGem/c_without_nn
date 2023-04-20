from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d


def open_img(filename):
    return np.asarray(Image.open(filename))


# @njit
def perform_conv(img, fil):
    # i = np.fft.fft(img)
    # f = np.fft.fft(fil)
    # return np.fft.ifft(np.multiply(i, f))
    res = np.zeros([img.shape[0], img.shape[1], 3])
    res[:, :, 0] = convolve2d(img[:, :, 0] / 255, fil, mode='same')
    res[:, :, 1] = convolve2d(img[:, :, 1] / 255, fil, mode='same')
    res[:, :, 2] = convolve2d(img[:, :, 2] / 255, fil, mode='same')
    return res

def draw_img(img):
    plt.imshow(img)
    plt.show()


def draw_triple(img, fil, res):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.set_title('Image')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.imshow(np.real(img))

    # ax2.set_title('Filter')
    # ax2.get_xaxis().set_visible(False)
    # ax2.get_yaxis().set_visible(False)
    # ax2.imshow(np.real(fil))

    ax2.set_title('Filter')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    h, w = fil.shape
    ax2.imshow(fil, cmap='Purples')
    for i in range(h):
        for j in range(w):
            ax2.text(j, i, int(fil[i, j] * 100) / 100, ha="center", va="center")
    # tb = plt.table(cellText=fil, loc=(0, 0), cellLoc='center')
    #
    # tc = tb.properties()['child_artists']
    # for cell in tc:
    #     cell.set_height(1.0 / 3)
    #     cell.set_width(1.0 / 3)
    #
    # ax2 = plt.gca()
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    ax3.set_title('Result')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.imshow(np.real(res))

    plt.show()


def process():
    img_name = 'files/masya_sunny.png'
    img = open_img(img_name)

    # fil = np.asarray([[1 / 9, 1 / 9, 1 / 9],
    #                   [1 / 9, 1 / 9, 1 / 9],
    #                   [1 / 9, 1 / 9, 1 / 9]])  # размытие
    # fil = np.asarray([[1, 2, 1],
    #                   [0, 0, 0],
    #                   [-1, -2, -1]])  # горизонтальный Собель
    # fil = np.asarray([[1, 0, -1],
    #                   [2, 0, -2],
    #                   [1, 0, -1]])  # вертикальный Собель
    fil = np.asarray([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]])
    # fil = np.asarray([[-1, -1, -1],
    #                   [-1, 9, -1],
    #                   [-1, -1, -1]])  # резкость
    # fil = np.asarray([[0, -1, 0],
    #                   [-1, 4, -1],
    #                   [0, -1, 0]])  # лаплассиан
    res = perform_conv(img, fil)

    draw_triple(img, fil, res)
    # draw_img(img)
    # draw_img(res)


if __name__ == '__main__':
    process()
