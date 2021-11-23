import numpy as np
import matplotlib.pyplot as plt

def convert_to_label(pred, confident=0.5):
    return (pred >confident) * 1.0


def convert_to_RGB_image(img):

    img = img.cpu().detach().numpy()
    c, Nx, Ny = np.shape(img)
    RGB_img = np.zeros([Nx, Ny, 3], dtype=np.int)

    for c in range(3):
        RGB_img[:, :, c] = (img[c, :, :] == 1).astype(np.uint8) * 128

    return RGB_img


def save_image(img, path):

    fig = plt.figure(1, figsize=[6, 6])
    if np.ndim(img) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    plt.axis('off')
    fig.savefig(path)
    plt.close(fig)