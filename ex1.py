import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)
    # return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    # Need to implement here
    pad_size = filter_size // 2
    padded_img = np.pad(img, pad_size, mode='edge')
    return padded_img

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    # Need to implement here

    # Perform padding
    padded_img = padding_img(img, filter_size)

    # Get image shape
    height, width = img.shape

    # Initialize smoothed image
    smoothed_img = np.zeros_like(img)

    # Apply mean filter
    for i in range(height):
        for j in range(width):
            # Extract neighborhood
            neighborhood = padded_img[i:i + filter_size, j:j + filter_size]
            # Apply mean filter
            smoothed_img[i, j] = np.mean(neighborhood)

    return smoothed_img

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    # Need to implement here

    # Perform padding
    padded_img = padding_img(img, filter_size)

    # Get image shape
    height, width = img.shape

    # Initialize smoothed image
    smoothed_img = np.zeros_like(img)

    # Apply median filter
    for i in range(height):
        for j in range(width):
            # Extract neighborhood
            neighborhood = padded_img[i:i + filter_size, j:j + filter_size]
            # Apply median filter
            smoothed_img[i, j] = np.median(neighborhood)

    return smoothed_img

def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    # Need to implement here

    # Calculate Mean Square Error (MSE)
    mse = np.mean((gt_img - smooth_img) ** 2)

    # Maximum possible pixel value
    max_pixel = 255

    # Calculate PSNR
    psnr_score = 10 * math.log10((max_pixel ** 2) / mse)

    return psnr_score

def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()

def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, image)


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    save_image(mean_smoothed_img, "ex1_images/mean_smoothed_img.png")
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    save_image(median_smoothed_img, "ex1_images/median_smoothed_img.png")
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

