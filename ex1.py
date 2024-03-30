import cv2
import matplotlib.pyplot as plt
import numpy as np


# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    image = cv2.imread(image_path)
    return image

# Display an image as function
def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


# grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """

    # Lấy kích thước của ảnh
    height, width = image.shape[:2]

    # Tạo một mảng numpy có cùng kích thước với ảnh và kiểu dữ liệu là uint8 để lưu ảnh xám
    img_gray = np.zeros((height, width), dtype=np.uint8)

    # Chuyển đổi từng điểm ảnh của ảnh màu sang ảnh xám sử dụng công thức
    for i in range(height):
        for j in range(width):
            # Lấy giá trị của các kênh màu
            B = image[i, j, 0]
            G = image[i, j, 1]
            R = image[i, j, 2]

            # Áp dụng công thức để tính giá trị xám
            gray_value = 0.299 * R + 0.587 * G + 0.114 * B

            # Gán giá trị xám cho điểm ảnh tương ứng
            img_gray[i, j] = gray_value

    return img_gray

# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, image)


# flip an image as function
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    flip_image = cv2.flip(image, 1)
    return flip_image

# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    # Lấy chiều rộng và chiều cao của ảnh
    height, width = image.shape[:2]

    # Tính toán điểm trung tâm để quay ảnh xung quanh nó
    center = (width / 2, height / 2)

    # Tạo ma trận quay
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Áp dụng phép xoay để xoay ảnh
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


if __name__ == "__main__":
    # Load an image from file
    img = load_image("images/uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/lena_gray_rotated.jpg")

    # Show the images
    plt.show()
