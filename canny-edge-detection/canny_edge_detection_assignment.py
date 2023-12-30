from PIL import Image
import numpy as np

def convert_to_grayscale(image):
    gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return gray_image

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def convolution(image, kernel):
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

def estimate_gradient(image):
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = convolution(image, filter_x)
    gradient_y = convolution(image, filter_y)

    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction

def non_maxima_suppression(gradient_magnitude, gradient_direction):
        M, N = gradient_magnitude.shape
        edge_image = np.zeros((M,N), dtype=np.int32)
        angle = gradient_direction * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1,M-1):
            for j in range(1,N-1):
                q = 255
                r = 255

                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                    edge_image[i,j] = gradient_magnitude[i,j]
                else:
                    edge_image[i,j] = 0

        return edge_image

def double_thresholding(image, lowThresholdRatio, highThresholdRatio):
    highThreshold = image.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = image.shape
    output = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)

    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak

    M, N = output.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (output[i,j] == weak):
                if ((output[i+1, j-1] == strong) or (output[i+1, j] == strong) or (output[i+1, j+1] == strong)
                    or (output[i, j-1] == strong) or (output[i, j+1] == strong)
                    or (output[i-1, j-1] == strong) or (output[i-1, j] == strong) or (output[i-1, j+1] == strong)):
                    output[i, j] = strong
                else:
                    output[i, j] = 0

    return output


if __name__ == "__main__":
    img = Image.open("lenna.png")

    img_array = np.array(img)

    # Convert to grayscale
    gray_image = convert_to_grayscale(img_array)

    # Filter image with a Gaussian kernel to remove noise
    kernel = gaussian_kernel(5)
    denoised_image = convolution(gray_image, kernel)

    # Estimate gradient strength and direction
    gradient_magnitude, gradient_direction = estimate_gradient(denoised_image)

    # Non-maxima suppression
    edge_image = non_maxima_suppression(gradient_magnitude, gradient_direction)

    # Link edge maximum gradient pixels using a dual threshold
    final_image = double_thresholding(edge_image, 0.05, 0.09)

    final_image = final_image.astype(np.uint8)

    Image.fromarray(final_image).save("final_edge_image.jpeg")