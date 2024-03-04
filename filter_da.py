import cv2
import numpy as np
import utils
import line_time_averaging

import line_finder

# todo add type hints

def draw_lines(image, lines, width=2):
    """draws lines onto an image/ mask

    Args:
        image (_type_): image to be drawn on
        lines (_type_): array of arrays [[xstart, ystart, xend, yend], [xstart, ystart, xend, yend]]

    Returns:
        (numpy.ndarray, dtype, uint8): image with lines drawn on
    """

    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line  # hvis input averaged lines
            cv2.line(lines_image, (x1, y1), (x2, y2), (1, 0, 0), width)
    return lines_image

def remove_da_outside_lines(da_seg_mask, approximate_lines, frame):
    """removes the drivable area outside the improved road lines

    Args:
        da_seg_mask (numpy.ndarray, dtype, uint8): drivable area mask 0 means not drivable, 1 means drivable
        approximate_lines (numpy.ndarray, dtype, uint8): lane lines mask 0 means no lane line, 1 means lane line
        expected shape of both: (720, 1280)  "720p"
        frame (tuple): shape of images

    Returns:
        (numpy.ndarray, dtype, uint8): corrected_da_mask

    """

    # flood fill begins at the seed_point wich is in the bottom middle of Image
    seed_point = (frame[1] // 2, frame[0] - 10)

    # Flood fill
    floodfill_color = 255  # white color for flood fill
    cv2.floodFill(approximate_lines, None, seed_point, floodfill_color)

    # combine the original image and the new mask
    corrected_da_mask = cv2.bitwise_and(da_seg_mask, approximate_lines)

    return corrected_da_mask


def filter_da(da_seg_mask, ll_seg_mask):
    """this is the main function filtering the drivable area (da)
    yolopv2 somtimes gives incomplete multiple and discontinous lines, it also somtimes miscattecgorises the grass on the side of the road as part of the road.
    This function tries to find a good straigth line aproximation for the track limits and then uses this to remove the da outside the lines somewhat like the fill tool in paint.

    Args:
        da_seg_mask (numpy.ndarray, dtype, int32): drivable area mask 0 means not drivable, 1 means drivable
        ll_seg_mask (numpy.ndarray, dtype, int32): lane lines mask 0 means no lane line, 1 means lane line
        expected shape of both: (720, 1280)  "720p"

    Returns:
        (numpy.ndarray, dtype, uint8): filtered drivable area mask
        (numpy.ndarray, dtype, uint8): improved road line mask
    """

    # Frame  (ymax, xmax)
    # todo remove frame and use np.shape instead or use frame and change from (y,x) to (x,y) for more consistent and easier to understand code
    frame = da_seg_mask.shape

    # change dtype to uint8 expected by cv2
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    da_seg_mask = da_seg_mask.astype(np.uint8)

    # tries to improve lines and section of the image
    left_line, right_line = line_finder.filter_lines(ll_seg_mask, frame)  # lines in (slope, y-intercept) format

    line_time_averaging.add_line_to_buffer(left_line, right_line)

    left_line, right_line = line_time_averaging.calculate_EMWA_lines()

    # connects the two lines at the intersection point to section of the image
    left_line, right_line = line_finder.lines_to_intersection(left_line, right_line, frame)

    # draws them onto the image
    road_lines = np.zeros_like(ll_seg_mask)
    road_lines = draw_lines(road_lines, [left_line, right_line])

    # removes the da outside the lines
    da_seg_mask = remove_da_outside_lines(da_seg_mask, road_lines, frame)

    return ll_seg_mask, da_seg_mask, road_lines


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
