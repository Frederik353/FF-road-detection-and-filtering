import cv2
import numpy as np

# class filter_da:
    # def __init__( self,):



def average_line(image, lines):
    left = []
    right = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            # lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))

    # takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    # create lines based on averages calculates
    if not np.isnan(left_avg).any():
        left_line = make_points(image, left_avg)
    else:
        left_line = [0,0,0,0]
    if not np.isnan(right_avg).any():
        right_line = make_points(image, right_avg)
    else:
        right_line = [0,0,0,0]

    return np.array([left_line, right_line])


def find_line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate the coefficients for the equations of the lines
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # If denominator is zero, lines are parallel and have no intersection within any frame
    if den == 0:
        return None

    # Compute the intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

    return int(np.ceil(px)), int(np.ceil(py))

def is_point_in_frame(px, py, minx, miny, maxx, maxy):
    return minx <= px <= maxx and miny <= py <= maxy

def line_intersection(lines, frame):
    # Unpack points from lines
    # [x1, y1, x2, y2] line format
    l1 = lines[0]
    l2 = lines[1]

    # Unpack the frame boundaries

    # Compute the intersection
    intersection = find_line_intersection(*l1, *l2)

    # If there's no intersection, return the lines as they are
    if intersection is None:
        return lines

    # If the intersection point is within the frame, return that point
    px, py = intersection

    # minx, miny, maxx, maxy
    if is_point_in_frame(px, py, 0, 0, frame[0], frame[1]):
        return [[l1[0], l1[1], px, py], [l2[0], l2[1], px, py]]
    else:
        return lines


def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    # how long we want our lines to be --> 3/5 the size of the image
    # y2 = int(y1 * (3/5))
    y2 = int(y1 * 0)
    # determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


def draw_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line  # hvis input averaged lines
            cv2.line(lines_image, (x1, y1), (x2, y2), (1, 0, 0), 2)
    return lines_image


def connect_components_directionally(img, frame, max_distance=10, desired_angle=0, angle_tolerance=10):
    """
    This function tries to connect components in a binary image in a given direction with a specified tolerance and maximum distance. 
    It finds the closest edge points between components that are within the given angle tolerance and distance, and draws a line to connect them.

    Args:
        img (np.ndarray, dtype uint8): A binary image in which components are to be connected.
        frame (tuple(int, int)): The size (width, height) of the image.
        max_distance (int, optional): The maximum distance between components for a connection to be considered. Defaults to 10.
        desired_angle (int, optional): The angle (in degrees) in which the connection is desired. Defaults to 0 (horizontal).
        angle_tolerance (int, optional): The tolerance (in degrees) for the angle difference between the actual connection and the desired angle. Defaults to 10.

    Returns:
        np.ndarray: An image with the same dimensions as the input, showing the original components and the connections between them.
    """
    # Find the connected components in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Convert the desired angle and angle tolerance from degrees to radians
    desired_angle_rad = np.deg2rad(desired_angle)
    angle_tolerance_rad = np.deg2rad(angle_tolerance)

    # Create a new image for drawing the connections
    connected_img = np.copy(img)

    print("number of lines detected: ", num_labels - 1) # -1 as the first label is the background

    # Iterate over every pair of components to consider potential connections
    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):
            # Find the closest edge points between components that match the specified direction and angle tolerance
            pt1, pt2, distance = find_closest_edge_points(i, j, labels, desired_angle_rad, angle_tolerance_rad, frame)

            # Connect the components if the closest points are within the specified maximum distance
            if pt1 is not None and distance <= max_distance:
                cv2.line(connected_img, tuple(pt1[::-1]), tuple(pt2[::-1]), 1, 3)  # Draw a line to connect the components

    return connected_img


def find_closest_edge_points(component1, component2, labels, desired_angle_rad, angle_tolerance_rad, frame):
    # Extract the edge points for each component
    edge_points1 = np.argwhere(labels == component1)
    edge_points2 = np.argwhere(labels == component2)

    # Initialize variables to store the closest points and minimum distance
    min_distance = np.inf
    closest_point1 = None
    closest_point2 = None

    # Iterate through all pairs of edge points to find the closest ones
    for pt1 in edge_points1:
        if pt1[1] > (frame[0] / 2):
            center = 1
        else: center = -1

        for pt2 in edge_points2:
            # Calculate the vector from pt1 to pt2
            vector = pt2 - pt1
            distance = np.linalg.norm(vector)
            angle = np.arctan2(vector[1], vector[0])

            # Check if the angle is within the tolerance
            if desired_angle_rad * center - angle_tolerance_rad <= angle <= desired_angle_rad * center + angle_tolerance_rad:
                # Update the closest points if the distance is less than the minimum found so far
                if distance < min_distance:
                    min_distance = distance
                    closest_point1 = pt1
                    closest_point2 = pt2

    return closest_point1, closest_point2, min_distance

def mark_first_white_pixels(img, connectivity_threshold=3):
    # Create an output image filled with zeros
    marked_img = np.zeros_like(img)

    # Get the middle index
    middle = img.shape[1] // 2

    # Iterate over each row to find the first white pixel from the middle outwards
    for row in range(img.shape[0]):
        # For the left side, start from the middle and go left
        left_index = np.argmax(img[row, :middle][::-1] == 1)
        if left_index > 0:  # If a white pixel is found
            # marked_img[row, middle - left_index] = 1

            start_index = middle - left_index
            marked_img[row, start_index - connectivity_threshold :start_index] = 1


        # For the right side, start from the middle and go right
        right_index = np.argmax(img[row, middle:] == 1)
        if right_index > 0:  # If a white pixel is found
            start_index = middle + right_index
            marked_img[row, start_index:start_index + connectivity_threshold ] = 1

    return marked_img

def filter_da(da_seg_mask, ll_seg_mask):
    """ this is the main function filtering the drivable area (da)
    yolopv2 somtimes gives incomplete multiple and discontinous lines, it also somtimes miscattecorises the grass on the side of the road as da
    this function tries to find a good aproximation for the road lines and then uses this to remove the da outside the lines

    Args:
        da_seg_mask (numpy.ndarray, dtype, int32): drivable area mask 0 means not drivable, 1 means drivable
        ll_seg_mask (numpy.ndarray, dtype, int32): lane lines mask 0 means no lane line, 1 means lane line
        expected shape of both: (720, 1280)  "720p"

    Returns:
        numpy.ndarray: filtered drivable area mask with better road lines
    """

    # Frame  (xmax, ymax)
    frame = da_seg_mask.shape

    # change dtype to uint8 expected by cv2
    ll_seg_mask = np.array(ll_seg_mask, dtype=np.uint8)

    # scan from middle out to the left and right and mark first  pixel found and remove the rest
    removed_outer =  mark_first_white_pixels(ll_seg_mask)

    # tries to connect the lines in the mask, if they are close enough in angle and distance
    # img, img_size, pixel_distance, angle, angle_tolerance
    connected = connect_components_directionally(removed_outer, frame, 100, 45, 10)

    removed_outer =  mark_first_white_pixels(connected)

    lines = cv2.HoughLinesP(removed_outer, 2, np.pi/180, 100, minLineLength=20, maxLineGap=50)

    averaged_lines = average_line(ll_seg_mask, lines)
    lines_to_intersection = line_intersection(averaged_lines, frame)

    road_lines = draw_lines(ll_seg_mask, lines_to_intersection)

    return  road_lines, road_lines





