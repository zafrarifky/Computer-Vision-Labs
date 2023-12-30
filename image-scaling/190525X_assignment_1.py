from PIL import Image
import numpy as np

# Convert the image to grayscale using NumPy
def convert_to_grayscale(img_arr):
    gray_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    grayscale_image = Image.fromarray(gray_array)
    return grayscale_image

def resize_image(image, new_width, new_height):
    original_array = np.array(image)
    width, height = original_array.shape[1], original_array.shape[0]

    resized_array = np.zeros((new_height, new_width), dtype=np.uint8)

    scale_x = (width-1) / new_width
    scale_y = (height-1) / new_height

    for i in range(new_width):
        for j in range(new_height):
            x = j * scale_x
            y = i * scale_y
            x0 = int(x)
            y0 = int(y)
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)
            dx = x - x0
            dy = y - y0

            # Bilinear interpolation
            interpolated_value = (1 - dx) * (1 - dy) * original_array[y0, x0] + \
                                dx * (1 - dy) * original_array[y0, x1] + \
                                (1 - dx) * dy * original_array[y1, x0] + \
                                dx * dy * original_array[y1, x1]

            resized_array[i, j] = int(interpolated_value)

    resized_image = Image.fromarray(resized_array)
    return resized_image

def compute_difference(original_image, resampled_image):
    original_pixels = np.array(original_image)
    resampled_pixels = np.array(resampled_image)
    sq_diff = (original_pixels - resampled_pixels)**2
    avg_sq_diff = np.mean(sq_diff)
    return avg_sq_diff

if __name__ == "__main__":
    # Input file path
    # input_file_path = "lenna_img.png"
    input_file_path = "image.jpeg"


    img = Image.open(input_file_path)
    img_array = np.array(img)

    # Convert to grayscale
    grayscale_image = convert_to_grayscale(img_array)
    grayscale_image.save("grayscale_img.jpeg")

    # Re-sample to 0.7 times original dimensions using linear interpolation
    original_width, original_height = grayscale_image.size
    scaling_factor = 0.7
    downsampled_width = int(original_width * scaling_factor)
    downsampled_height = int(original_height * scaling_factor)
    resized_image = resize_image(grayscale_image, downsampled_width, downsampled_height)
    resized_image.save("resized_img.jpeg")

    # Re-sample back to original dimensions using linear interpolation
    resized_back_image = resize_image(grayscale_image, original_width, original_height)
    resized_back_image.save("resized_back_img.jpeg")

    print("Average Squared Difference: ", compute_difference(grayscale_image, resized_back_image))

