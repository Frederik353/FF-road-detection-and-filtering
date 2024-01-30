import numpy as np
import cv2


def generate_distinct_colors(n):
    """
    Generate a list of n distinct colors.

    This function generates distinct colors by evenly sampling
    the hue component in the HSV color space and then converting
    them to the RGB color space. It's useful for generating distinct
    colors for labeling or segmentation tasks.

    Args:
    n (int): The number of distinct colors to generate.

    Returns:
    list: A list of RGB colors, each represented as a tuple of three integers.
    """

    # Generate colors in HSV space. HSV is used because varying the hue
    # with a fixed saturation and value gives good color diversity.
    # OpenCV's Hue range is from 0-180 (instead of 0-360), hence the scaling.
    hsv_colors = [(i * 180 / n, 255, 255) for i in range(n)]

    # Convert HSV colors to RGB. This step is necessary because most image
    # processing applications use the RGB color model.
    rgb_colors = [cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0] for hsv in hsv_colors]

    return rgb_colors


# Example usage:
num_colors = 10
colors = generate_distinct_colors(num_colors)
for i in colors:
    print(i)
