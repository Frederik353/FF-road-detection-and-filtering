import cv2
import numpy as np

import utils
import line_time_averaging
import line_finder
from typing import List, Tuple, Optional


def draw_lines(image: np.ndarray, lines: Optional[List[List[int]]], width: int = 2) -> np.ndarray:
    """
    Draws lines onto an image or mask.

    Args:
        image (np.ndarray): The image to be drawn on. It should be a NumPy array.
        lines (List[List[int]]): An array of lines, where each line is represented as [xstart, ystart, xend, yend].
        width (int, optional): The thickness of the lines to be drawn. Defaults to 2.

    Returns:
        np.ndarray: An image with lines drawn on it.
    """

    # Create an image filled with zeros, having the same dimensions as the input image
    lines_image = np.zeros_like(image)

    # Draw each line on the lines_image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            # Draw the line on the image. Color is set to (1, 0, 0), which is blue for an RGB image.
            # If the input image is grayscale or another format, this color needs to be adjusted.
            cv2.line(lines_image, (x1, y1), (x2, y2), (1, 0, 0), width)

    return lines_image


def remove_da_outside_lines(
    da_seg_mask: np.ndarray, approximate_lines: np.ndarray, frame: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Removes the drivable area outside the improved road lines by applying a flood fill operation from the bottom middle
    of the image and then combining it with the original drivable area mask.

    Args:
        da_seg_mask (np.ndarray): Drivable area mask where 0 means not drivable and 1 means drivable.
        approximate_lines (np.ndarray): Lane lines mask where 0 means no lane line and 1 means lane line.
        frame (tuple): The shape of the images, given as (height, width).

    Returns:
        np.ndarray: The corrected drivable area mask.
    """

    # flood fill begins at the seed_point wich is in the bottom middle of Image
    seed_point = (frame[2] // 2, frame[3] - 10)

    # Flood fill
    floodfill_color = (255, 255, 255)  # white color for flood fill
    mask = np.zeros((frame[3] + 2, frame[2] + 2), np.uint8)
    cv2.floodFill(approximate_lines, mask, seed_point, floodfill_color)

    # combine the original image and the new mask
    corrected_da_mask = cv2.bitwise_and(da_seg_mask, approximate_lines)

    return corrected_da_mask


def filter_da(da_seg_mask: np.ndarray, ll_seg_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Refines the drivable area segmentation by using improved road line estimations to filter out non-road areas.
    This function integrates various processing steps to create a more accurate drivable area mask based on the lane lines.

    Args:
        da_seg_mask (np.ndarray): A binary mask representing the drivable area, where 1 indicates drivable space and 0 indicates non-drivable space.
        ll_seg_mask (np.ndarray): A binary mask representing detected lane lines, where 1 indicates a lane line and 0 indicates no lane line.

    Returns:
        todo update return type
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three np.ndarrays:
            - The original lane line mask (ll_seg_mask)
            - The filtered drivable area mask (filtered_da_seg_mask)
            - The mask with improved road lines (road_lines)
    """

    # Determine the frame dimensions from the drivable area mask
    height, width = da_seg_mask.shape
    frame = [0, 0, width, height]

    # Convert masks to uint8 as expected by OpenCV functions
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    da_seg_mask = da_seg_mask.astype(np.uint8)

    # Improve lines and section of the image
    left_line, right_line = line_finder.filter_lines(ll_seg_mask, frame)

    # Add lines to buffer for time averaging
    line_time_averaging.add_line_to_buffer(left_line, right_line)

    # Calculate Exponentially Moving Weighted Average (EMWA) lines
    left_line, right_line = line_time_averaging.calculate_EMWA_lines()

    # Connect the two lines at the intersection point
    left_line, right_line = line_finder.lines_to_intersection(left_line, right_line, frame)

    # Draw the improved road lines on a new mask
    road_lines = np.zeros_like(ll_seg_mask)
    road_lines = draw_lines(road_lines, [left_line, right_line])

    # Remove the drivable area outside the improved road lines
    filtered_da_seg_mask = remove_da_outside_lines(da_seg_mask, road_lines, frame)

    return ll_seg_mask, filtered_da_seg_mask, road_lines


# ------------------------ delet this (for debugging setup dont have gpu   :(      ) ------------------------

def debug_image(masks, palette=None, is_demo=False, t=10_000, window="debug"):
    """show an image
    Args:
        image (_type_): image to show
        t (int): time to show image in ms
    """
    # if not given define
    if palette == None:
        palette = generate_distinct_colors(len(masks))
    else:
        # if colors are given make sure you have enough for every mask
        assert len(palette) == len(masks)

    np_image = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)

    for label, color in enumerate(palette):
        if masks[label].ndim == 2:
            np_image[masks[label] == 1, :] = color
        elif masks[label].ndim == 3:
            mask = np.any(masks[label] > 0, axis=-1)
            # Convert the mask into a format that can be used for the overlay operation
            mask_3d = np.stack([mask] * 3, axis=-1)

            # Where mask is True, take pixels from the overlay image; else, take pixels from the background
            np_image = np.where(mask_3d, masks[label], np_image)

    # # If image is boolean, we need to convert to 0s and 255s
    # if np.max(image) == 1:
    # np_image = image.astype(np.uint8) * 255
    # else:
    # np_image = image.astype(np.uint8)

    # display over image
    # color_mask = np.mean(color_seg, 2)
    # img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5

    cv2.imshow(window, np_image)
    cv2.waitKey(t)  # Display the image for 10 seconds
    # cv2.destroyAllWindows()


def generate_distinct_colors(n):
    """
    Generate a list of n distinct colors.

    This function generates distinct colors by evenly sampling
    the hue component in the HSV color space and then converting
    them to the RGB color space.

    Args:
    n (int): The number of distinct colors to generate.

    Returns:
    np.array: A list of RGB colors, each represented as a tuple of three integers.
    """

    # Generate colors in HSV space. HSV is used because varying the hue
    # with a fixed saturation and value gives good color diversity.
    # OpenCV's Hue range is from 0-180 (instead of 0-360), hence the scaling.
    hsv_colors = [(i * 180 / n, 255, 255) for i in range(n)]

    # Convert HSV colors to RGB
    rgb_colors = np.array([cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0] for hsv in hsv_colors])
    return rgb_colors


if __name__ == "__main__":
    from dbug import debug_filter_da

    foo = debug_filter_da()
