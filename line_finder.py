import numpy as np
import cv2
import utils
import constants


def sort_lines_LR(lines, frame):
    left, right = [], []

    # Define the midpoint in the x-direction
    mid_x = frame[1] // 2

    # Process each line to sort by side and then by correct slope direction
    # todo necessary?
    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]

        if x1 < mid_x and x2 < mid_x and slope < 0:
            left.append((slope, y_int))
        elif x1 >= mid_x and x2 >= mid_x and slope > 0:
            right.append((slope, y_int))
    return left, right


def mark_first_white_pixels(img, connectivity_threshold=2):
    """
    this function scans from middle out to the left and right and mark first  pixel found and remove the rest

    Args:

        img (numpy.ndarray, dtype, uint8): lane lines mask 0 means no lane line, 1 means lane line categorised by yolopv2
        connectivity_threshold (int, optional): how many extra pixels to add to the left/right of the first pixel found. Defaults to 2.

    Returns:
        (numpy.ndarray, dtype, uint8): image mask with only the first white pixels found in each row
    """

    # Create an output image filled with zeros
    marked_img = np.zeros_like(img)

    # Get the middle index
    middle = img.shape[1] // 2

    # Split the image into left and right halves
    left_half = img[:, :middle][:, ::-1]  # Flip the left half
    right_half = img[:, middle:]
    flipped = img[::-1]

    # Find the first white pixel in each row for both halves
    left_indices = np.argmax(left_half == 1, axis=1)
    right_indices = np.argmax(right_half == 1, axis=1)
    bottom_indices = np.argmax(flipped == 1, axis=0)

    # Correct indices for left half
    left_indices[left_indices > 0] = middle - left_indices[left_indices > 0]

    # Correct indices for right half
    right_indices[right_indices > 0] += middle

    # correct coordinate to top down
    bottom_indices[bottom_indices > 0] -= img.shape[0]

    # todo is connectivity threshold needed?
    # Marking the pixels with connectivity threshold
    # Scan from middle out
    for row in range(img.shape[0]):
        if left_indices[row] > 0:
            start_index = left_indices[row]
            marked_img[row, start_index - connectivity_threshold : start_index] = 1

        if right_indices[row] > 0:
            start_index = right_indices[row]
            marked_img[row, start_index : start_index + connectivity_threshold] = 1

    # Scan from bottom up for vertical scan
    for col in range(img.shape[1]):
        # Find the bottom-most white pixel in the column
        col_pixels = img[:, col]
        non_zero_indices = np.nonzero(col_pixels)[0]
        if non_zero_indices.size > 0:
            bottom_most_index = non_zero_indices[-1]  # Take the last non-zero index
            # Mark the pixels with connectivity threshold for vertical scan
            marked_img[max(0, bottom_most_index - connectivity_threshold) : bottom_most_index + 1, col] = 1

    return marked_img


def filter_lines(ll_seg_mask, frame):
    global line_buffer
    """this function tries to improve the lane lines and section of the image

    Args:
        ll_seg_mask (numpy.ndarray, dtype, uint8): lane lines mask 0 means no lane line, 1 means lane line categorised by yolopv2
        expected shape: (720, 1280)  "720p"
        frame (tuple): shape of image

    Returns:
        (numpy.ndarray, dtype, uint8): corrected_da_mask
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

    # todo althoug a low ct is more accurate as high ct will mess up the lines more if lines cross center e.g. left line cross into right side and blocks right line and left search might find a second line messing up left line too

    # scan from middle out to the left and right and mark first  pixel found and remove the rest
    # connectivity_threshold is how many pixels to match on each line
    # todo when marking detect if two lines detected, look at size and delete smallest line
    removed_outer = mark_first_white_pixels(ll_seg_mask, connectivity_threshold=1)

    # tries to connect the lines in the mask, if they are close enough in angle and distance
    # img, pixel_max_distance
    connected = utils.connect_components(removed_outer, max_distance=constants.CC_MAX_DISTANCE)

    # same process as before but with the connected lines
    # a quite high connectivity threshold does not hurt hough line
    removed_outer = mark_first_white_pixels(connected, connectivity_threshold=1)

    # todo try spline interpolation or split and merge line fittng or curve fitting
    # todo time avg line, look at line movment over time and see if it makes sense for the line to move that much in that time
    # does a houghtransform to find lines in the mask
    # Applying the Hough Line Transform
    lines = cv2.HoughLinesP(
        removed_outer,
        constants.RHO,
        constants.THETA,
        constants.THRESHOLD,
        minLineLength=constants.MIN_LINE_LENGTH,
        maxLineGap=constants.MAX_LINE_GAP,
    )

    # might find many lines, take the average of them
    # left_line, right_line = average_lines(ll_seg_mask, lines, frame)
    left_lines, right_lines = sort_lines_LR(lines, frame)

    # if no line is found we try to make an educated guess
    if not left_lines:
        print("no left lines found")
        left_lines.append(constants.LEFT_FALLBACK_LINE)
    if not right_lines:
        print("no right lines found")
        right_lines.append(constants.RIGHT_FALLBACK_LINE)

    # Calculate the median slope and y-intercept for both the left and right lines
    left_line = np.median(left_lines, axis=0) if left_lines else np.nan
    right_line = np.median(right_lines, axis=0) if right_lines else np.nan

    # TODO: combine houglines with first pixels ?

    return left_line, right_line


def lines_to_intersection(left_line, right_line, frame):
    """
    Calculate the intersection points of two lines within a given frame.

    This function takes two lines and a frame (defined by its width and height), and calculates the
    intersection point of these lines. If the intersection point is within the frame, the function
    returns new lines that extend from the original line start points to the intersection point.
    If there is no intersection or the intersection point is outside the frame, the original lines
    are returned.

    Parameters:
    lines (list of lists): A list containing two lines, where each line is represented by a list of four integers [x1, y1, x2, y2].
    frame (tuple): A tuple of two integers representing the width and height of the frame.

    Returns:
    list of lists: A list containing two lines (each a list of four integers). If an intersection within the frame is found,
                   these lines extend to the intersection point. Otherwise, the original lines are returned.
    """

    intersection = utils.line_intersection_slope_y_intercept(left_line, right_line)

    left_line = utils.make_line_points(left_line, frame)
    right_line = utils.make_line_points(right_line, frame)

    left_line = utils.sort_line_bottom_first(left_line)
    right_line = utils.sort_line_bottom_first(right_line)

    # todo fix fucked frame and interface
    if utils.is_point_in_frame(intersection, 0, 0, *frame[::-1]):
        left_line = np.concatenate([left_line[:2], intersection])
        right_line = np.concatenate([right_line[:2], intersection])
    else:
        print("intersection outside frame")

    return left_line, right_line
