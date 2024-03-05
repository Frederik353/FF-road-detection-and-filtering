import numpy as np
import cv2
import utils
import constants
from typing import List, Tuple, Optional


def sort_lines_LR(
    lines: Optional[List[np.ndarray]], frame: Tuple[int, int]
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Sorts lines into left and right groups based on their position and slope.

    Args:
        lines (Optional[List[np.ndarray]]): A list of line coordinates.
        frame (Tuple[int, int]): The dimensions of the frame (width, height).

    Returns:
        Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]: Two lists containing the slope and y-intercept
                                                                      of the left and right lines, respectively.
    """
    left, right = [], []

    # Define the midpoint in the x-direction
    mid_x = frame[2] // 2

    # Return empty lists if no lines are provided
    if lines is None:
        return left, right

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Calculate the slope and y-intercept of the line
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]

        # Assign lines to the left group if they are on the left side and have a negative slope
        if x1 < mid_x and x2 < mid_x and slope < 0:
            left.append((slope, y_int))
        # Assign lines to the right group if they are on the right side and have a positive slope
        elif x1 >= mid_x and x2 >= mid_x and slope > 0:
            right.append((slope, y_int))

    return left, right


# todo is connectivity threshold needed?

# new
# average time: -0.05599002288968376
# median time: -0.050994157791137695


def mark_first_white_pixels(img: np.ndarray, connectivity_threshold: int = 2) -> np.ndarray:
    """
    Marks the first white pixel found in each row for the left and right halves of the image and from bottom up vertically.

    Args:
        img (np.ndarray): The input binary image (0s and 1s) to process.
        connectivity_threshold (int): The number of pixels to mark around the first white pixel found.

    Returns:
        np.ndarray: An image with the first white pixels marked in each row and column.
    """

    # Create an output image filled with zeros
    marked_img = np.zeros_like(img)

    # Get the middle index to divide the image into left and right halves
    middle = img.shape[1] // 2

    # Split the image into left and right halves
    left_half = img[:, :middle][:, ::-1]  # Flip the left half for easier processing
    right_half = img[:, middle:]

    # Find the first white pixel in each row for both halves
    left_indices = np.argmax(left_half == 1, axis=1)
    right_indices = np.argmax(right_half == 1, axis=1)

    # Correct indices for the left half
    left_indices[left_indices > 0] = middle - left_indices[left_indices > 0]

    # Correct indices for the right half
    right_indices[right_indices > 0] += middle

    # Marking the pixels based on the connectivity threshold
    for row in range(img.shape[0]):
        if left_indices[row] > 0:
            start_index = left_indices[row]
            marked_img[row, max(0, start_index - connectivity_threshold) : start_index] = 1

        if right_indices[row] > 0:
            start_index = right_indices[row]
            marked_img[row, start_index : min(start_index + connectivity_threshold, img.shape[1])] = 1

    # Scan from bottom up for each column to mark the first white pixel found
    for col in range(img.shape[1]):
        # Find the bottom-most white pixel in the column
        col_pixels = img[:, col]
        non_zero_indices = np.nonzero(col_pixels)[0]
        if non_zero_indices.size > 0:
            bottom_most_index = non_zero_indices[-1]  # Take the last non-zero index
            # Mark the pixels with the connectivity threshold for vertical scanning
            marked_img[max(0, bottom_most_index - connectivity_threshold) : bottom_most_index + 1, col] = 1

    return marked_img


# def mark_first_white_pixels(img: np.ndarray, ct) -> np.ndarray:
#     """
#     Marks the first white pixel found in each row for the left and right halves of the image and from bottom up vertically.

#     Args:
#         img (np.ndarray): The input binary image (0s and 1s) to process.

#     Returns:
#         np.ndarray: An image with the first white pixels marked in each row and column.
#     """

#     # Create an output image filled with zeros
#     marked_img = np.zeros_like(img)

#     # Get the middle index to divide the image into left and right halves
#     middle = img.shape[1] // 2

#     # Split the image into left and right halves
#     left_half = img[:, :middle][:, ::-1]  # Flip the left half for easier processing
#     right_half = img[:, middle:]

#     # Find the first white pixel in each row for both halves
#     left_indices = np.argmax(left_half == 1, axis=1)
#     right_indices = np.argmax(right_half == 1, axis=1)

#     # Mark the first white pixel for the left half, if it exists
#     for row, col in enumerate(middle - left_indices):
#         if left_half[row, middle - col - 1] == 1:
#             marked_img[row, col] = 1

#     # Mark the first white pixel for the right half, if it exists
#     for row, col in enumerate(right_indices + middle):
#         if right_half[row, col - middle] == 1:
#             marked_img[row, col] = 1

#     # Scan from bottom up for each column to mark the first white pixel found
#     for col in range(img.shape[1]):
#         # Find the bottom-most white pixel in the column
#         col_pixels = img[:, col]
#         non_zero_indices = np.nonzero(col_pixels)[0]
#         if non_zero_indices.size > 0:
#             bottom_most_index = non_zero_indices[-1]  # Take the last non-zero index
#             # Mark the pixel
#             marked_img[bottom_most_index, col] = 1

#     return marked_img


# todo althoug a low ct is more accurate as high ct will mess up the lines more if lines cross center e.g. left line cross into right side and blocks right line and left search might find a second line messing up left line too
# todo try spline interpolation or split and merge line fittng or curve fitting


def filter_lines(ll_seg_mask: np.ndarray, frame: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes the lane lines mask to improve detection and identify the primary left and right lane lines.

    Args:
        ll_seg_mask (np.ndarray): A binary mask where 1 indicates a detected lane line, and 0 indicates no lane line.
        frame (Tuple[int, int]): The dimensions (height, width) of the input image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the median lines for the left and right lanes, represented
                                       as (slope, y-intercept).
    """

    """ calculating desired connectivity threshold
    ct = connectivity_threshold

    1 / tan( theta )

    theta is assumed (max) angle of the track lines in the image in radians
    radian = deg * pi/180
    this gives you how many pixels you have to match for the line to still be fully connected (no gap in the line)
    90 deg is vertical, 0 deg is horizontal
    10 deg => 6
    30 deg => 2
    low ct => connect components will have to do a lot more work depending on how steep the line is.
    ct = 10 => 25-30 components
    ct = 1 => 60 - 250 components

    """

    # todo is ct necessary with new scan method
    # Define the connectivity threshold for marking white pixels

    connectivity_threshold = 1

    # Mark the first white pixel from the middle outwards and remove the rest, applying the connectivity threshold
    removed_outer = mark_first_white_pixels(ll_seg_mask, connectivity_threshold)

    # Connect nearby components based on a maximum distance
    connected = utils.connect_components(removed_outer, max_distance=constants.CC_MAX_DISTANCE)

    # Repeat the process of marking first white pixels with the connected components
    removed_outer = mark_first_white_pixels(connected, connectivity_threshold)

    # Apply Hough Line Transform to find lines in the mask
    lines = cv2.HoughLinesP(
        removed_outer,
        constants.RHO,
        constants.THETA,
        constants.THRESHOLD,
        minLineLength=constants.MIN_LINE_LENGTH,
        maxLineGap=constants.MAX_LINE_GAP,
    )

    # Sort the detected lines into left and right based on their slope and position
    left_lines, right_lines = sort_lines_LR(lines, frame)

    # Provide fallback lines if no lines were detected
    if not left_lines:
        left_lines.append(constants.LEFT_FALLBACK_LINE)
    if not right_lines:
        right_lines.append(constants.RIGHT_FALLBACK_LINE)

    # Calculate the median line parameters (slope, y-intercept) for left and right lines
    left_line = np.median(left_lines, axis=0) if left_lines else np.array([np.nan, np.nan])
    right_line = np.median(right_lines, axis=0) if right_lines else np.array([np.nan, np.nan])

    return left_line, right_line


# todo splitt up
def lines_to_intersection(
    left_line: Tuple[float, float], right_line: Tuple[float, float], frame: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusts two lines to intersect within a given frame, if their intersection point lies inside the frame.

    Parameters:
    - left_line (Tuple[float, float]): The slope and y-intercept of the left line.
    - right_line (Tuple[float, float]): The slope and y-intercept of the right line.
    - frame (Tuple[int, int]): The dimensions (width, height) of the frame.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Two adjusted lines represented by their endpoints. If the intersection is inside the frame,
      lines are adjusted to meet at the intersection point. Otherwise, the lines are returned without modification.
    """
    # Assuming utils.line_intersection_slope_y_intercept calculates the intersection of two lines
    intersection = utils.line_intersection_slope_y_intercept(left_line, right_line)

    # Assuming utils.make_line_points converts line representations from slope and intercept to endpoint coordinates
    left_line_points = utils.make_line_points(left_line, frame)
    right_line_points = utils.make_line_points(right_line, frame)

    # Assuming utils.sort_line_bottom_first ensures the line points are ordered from bottom to top
    left_line_sorted = utils.sort_line_bottom_first(left_line_points)
    right_line_sorted = utils.sort_line_bottom_first(right_line_points)

    # left_line = utils.clamp_line_inside_frame(left_line_sorted, frame)
    # right_line = utils.clamp_line_inside_frame(right_line_sorted, frame)

    # Check if the intersection point is within the frame and adjust the lines if it is
    if intersection is not None and utils.is_point_in_frame(intersection, frame):
        # Adjust the lines to end at the intersection point
        left_line_adjusted = np.concatenate([left_line_sorted[:2], intersection])
        right_line_adjusted = np.concatenate([right_line_sorted[:2], intersection])
    else:
        # No valid intersection within the frame, return the original lines
        left_line_adjusted = left_line_sorted
        right_line_adjusted = right_line_sorted

    print(left_line_adjusted, right_line_adjusted)
    return left_line_adjusted, right_line_adjusted


# ------------------------------------------------------------


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
