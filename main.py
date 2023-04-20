from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.image import imsave
import numpy as np
from numba import njit
from numba_progress import ProgressBar

# file_name = 'files/img.png'
# file_img = Image.open(file_name).convert('L')
ver = False


def open_for_main_page():
    main_img_name = 'files/runes_2_long_sheet.png'
    part_img_dot = 'files/letters/runes_dot.png'
    part_img_i = 'files/letters/runes_i.png'
    part_img_f = 'files/letters/runes_f.png'
    part_img_8 = 'files/letters/runes_8.png'
    return open_image_grayscale(main_img_name),\
        open_image_grayscale(part_img_dot),\
        open_image_grayscale(part_img_i),\
        open_image_grayscale(part_img_f),\
        open_image_grayscale(part_img_8)


def open_royal():
    # file_name = 'files/img.png'
    main_img_name = 'files/runes_2.png'
    part_img_name = 'files/letters/runes_f.png'
    # main_img_name = 'files/masya_sunny_mawska.png'
    # part_img_name = 'files/masya_sunny_eyesik.png'
    main = open_image_grayscale(main_img_name)
    part = open_image_grayscale(part_img_name)
    return main, part


def open_image_grayscale(file_name):
    img = Image.open(file_name).convert('L')
    # img = img.point(lambda p: 255 if p > 128 else 0)
    # img.thumbnail((scale, scale))
    img = np.asarray(img)
    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # img = ~img - 255
    # img = img // 1.5
    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    return img


def interval_mapping(img, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((img - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


def normalize_image(img):
    return interval_mapping(img, 0, 255, 0.0, 1.0)


def denormalize_image(img):
    return interval_mapping(img, 0.0, 1.0, 0, 255).astype(int)


@njit
def count_correlation(main, part):
    res = np.zeros(main.shape)
    h_f, w_f = main.shape
    h_h, w_h = part.shape
    s_h = np.sum(np.square(part))  # считается всего один раз
    for j1 in range(w_f):
        for j2 in range(h_f):
            k = 0
            s_f = 0  # эту нужно накапливать
            for i1 in range(w_h):
                for i2 in range(h_h):
                    if j1 + i1 >= w_f or j2 + i2 >= h_f:
                        continue
                    s_f += main[j2 + i2, j1 + i1] ** 2
                    k += part[i2, i1] * main[j2 + i2, j1 + i1]
            res[j2, j1] = k / (np.sqrt(s_f * s_h))  # / (w_h * h_h)
    return res


def process_images(dir_ver=True):
    main, part = open_royal()
    # main = part

    fig = plt.figure()
    if dir_ver:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
    else:
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

    ax1.set_title('Part image')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.imshow(np.real(part), cmap='gray', vmin=0, vmax=255)

    ax2.set_title('Main image')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.imshow(np.real(main), cmap='gray', vmin=0, vmax=255)

    ax3.set_title('Result')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    # with ProgressBar(total=main.shape[1]) as progress:
    #     res = count_correlation(normalize_image(main),
    #                             normalize_image(part),
    #                             progress)
    # res = count_correlation(main, part, progress)
    res = count_correlation(normalize_image(main),
                            normalize_image(part))
    # print(res) res = denormalize_image(res)
    res = denormalize_image(res)
    ax3.imshow(np.real(res), cmap='gray', vmin=0, vmax=255)
    plt.show()

    plt.imshow(np.real(res), cmap='gray', vmin=0, vmax=255)
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = np.arange(0, res.shape[1])
    y = np.arange(0, res.shape[0])
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, res,
                    cmap=cm.Wistia,
                    # edgecolor='y',
                    # linewidth=0.005,
                    antialiased=True,
                    shade=False)
    ax.set_zlim(0, 255)
    plt.show()

    # ax.plot_surface(x, y, res,
    #                 cmap=cm.copper,
    #                 edgecolor='w',
    #                 linewidth=0.01,
    #                 antialiased=False)


def process_images_for_main_page():
    main, part_dot, part_i, part_f, part_8 = open_for_main_page()
    res_dot = denormalize_image(
        count_correlation(
            normalize_image(main),
            normalize_image(part_dot)))
    res_i = denormalize_image(
        count_correlation(
            normalize_image(main),
            normalize_image(part_i)))
    res_f = denormalize_image(
        count_correlation(
            normalize_image(main),
            normalize_image(part_f)))
    res_8 = denormalize_image(
        count_correlation(
            normalize_image(main),
            normalize_image(part_8)))

    img = np.dstack((res_dot, res_i, res_8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # img = Image.fromarray(img)
    # img.save('/home/iris/Documents/articles_for_audit/main_pic/test.png', format='PNG')
    imsave('/home/iris/Documents/articles_for_audit/main_pic/test.png', img.astype('uint8'))

if __name__ == '__main__':
    # process_images(ver)
    process_images_for_main_page()
