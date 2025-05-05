import random
import torch
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as T


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


def sample(train_dataset, domain):
    while True:
        idx = random.randint(0, len(train_dataset) - 1)
        if train_dataset[idx][3] != domain:
            return train_dataset[idx]


def get_augmented(images, spinal_cord_masks, gm_masks, domains, train_dataset):
    for i in range(images.size(0)):
        img1, spinal_cord_mask1, gm_mask1, domain1 = images[i], spinal_cord_masks[i], gm_masks[i], domains[i]
        img2, spinal_cord_mask2, gm_mask2, domain2 = sample(train_dataset, domain1)

        t = T.ToPILImage(mode="L")
        img1 = t(img1)
        img2 = t(img2)

        img1 = np.array(img1)
        img2 = np.array(img2)
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)

        # img1, img2 = img1.permute(1, 2, 0).numpy(), img2.permute(1, 2, 0).numpy()

        img21, img12 = colorful_spectrum_mix(img1, img2, alpha=1)
        img1, img2, img21, img12 = torch.from_numpy(img1).permute(2, 0, 1), torch.from_numpy(img2).permute(2, 0, 1), \
                                   torch.from_numpy(img21).permute(2, 0, 1), torch.from_numpy(img12).permute(2, 0, 1)

        img1 = img1 / (img1.max() if img1.max() > 0 else 1)
        img2 = img2 / (img2.max() if img2.max() > 0 else 1)
        img21 = img21 / (img21.max() if img21.max() > 0 else 1)
        img12 = img12 / (img12.max() if img12.max() > 0 else 1)

        img1, img2, img21, img12 = img1.unsqueeze(0), img2.unsqueeze(0), img21.unsqueeze(0), img12.unsqueeze(0)
        spinal_cord_mask1, spinal_cord_mask2 = spinal_cord_mask1.unsqueeze(0), spinal_cord_mask2.unsqueeze(0)
        gm_mask1, gm_mask2 = gm_mask1.unsqueeze(0), gm_mask2.unsqueeze(0)

        images = torch.cat((images, img2, img21, img12), 0)
        spinal_cord_masks = torch.cat((spinal_cord_masks, spinal_cord_mask2, spinal_cord_mask1, spinal_cord_mask2), 0)
        gm_masks = torch.cat((gm_masks, gm_mask2, gm_mask1, gm_mask2), 0)

        domains = list(domains)
        domains += [domain2, domain1, domain2]
        domains = tuple(domains)

    return images, spinal_cord_masks, gm_masks, domains


def plot(data, label, title):
    for i in range(len(data)):
        plt.plot(data[i], label=label[i])

    plt.legend()
    plt.title(title)

    return plt


def show_augmented(data1, data2):
    image1, spinal_cord_mask1, gm_mask1, domain1 = data1
    image2, spinal_cord_mask2, gm_mask2, domain2 = data2

    t = T.ToPILImage(mode="L")

    print(domain1 + " vs " + domain2)
    print("Original")
    image1 = t(image1)
    image1.show()
    print()
    image2 = t(image2)
    image2.show()

    image1 = np.array(image1)
    image2 = np.array(image2)
    image1 = np.expand_dims(image1, axis=2)
    image2 = np.expand_dims(image2, axis=2)

    image21, image12 = colorful_spectrum_mix(image1, image2, alpha=1.0)

    image21 = image21.squeeze()
    image12 = image12.squeeze()

    image21 = Image.fromarray(image21.astype('uint8'), 'L')
    image12 = Image.fromarray(image12.astype('uint8'), 'L')

    print("Augmented")
    image21.show()
    print()
    image12.show()


