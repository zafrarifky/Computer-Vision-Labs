import numpy as np
from PIL import Image

def inter_means(image):

    image = np.array(image)

    initial_threshold = np.mean(image)

    while True:
        R1 = image[image <= initial_threshold]
        R2 = image[image > initial_threshold]

        m1 = np.mean(R1) if len(R1) > 0 else 0
        m2 = np.mean(R2) if len(R2) > 0 else 0

        new_threshold = (m1 + m2) / 2

        if abs(new_threshold - initial_threshold) < 1e-2:
            break

        initial_threshold = new_threshold

    output = (image > initial_threshold) * 255
    output_image = Image.fromarray(output.astype('uint8'))

    return output_image

if __name__ == "__main__":
    image1 = Image.open('image_1.jpeg')
    result1 = inter_means(image1)
    result1.save('segmented_image_1.jpeg')

    image2 = Image.open('image_2.jpeg')
    result2 = inter_means(image2)
    result2.save('segmented_image_2.jpeg')

