from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


def draw_bunch(img_1, img_2, img_3, img_4, img_5):
    fig = plt.figure()
    ax1 = fig.add_subplot(151)
    ax2 = fig.add_subplot(152)
    ax3 = fig.add_subplot(153)
    ax4 = fig.add_subplot(154)
    ax5 = fig.add_subplot(155)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.imshow(np.real(img_3))

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.imshow(np.real(img_2))

    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.imshow(np.real(img_1))

    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.imshow(np.real(img_4))

    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax5.imshow(np.real(img_5))

    plt.show()


if __name__ == '__main__':
    im_1 = np.asarray(Image.open('files/letters/runes_i.png'))
    im_2 = np.asarray(Image.open('files/letters/runes_dot.png'))
    im_3 = np.asarray(Image.open('files/letters/runes_f.png'))
    im_4 = np.asarray(Image.open('files/letters/runes_4.png'))
    im_5 = np.asarray(Image.open('files/letters/runes_8.png'))

    draw_bunch(im_1, im_2, im_3, im_4, im_5)
