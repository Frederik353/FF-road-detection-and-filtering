import cv2
import numpy as np


def find_line_intersection(line1, line2):
    # Calculate the coefficients A, B, and C for each line
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # Form the matrix and the constant vector
    matrix = np.array([[A1, B1], [A2, B2]], dtype=np.float64)
    constants = np.array([C1, C2], dtype=np.float64)

    # Check for parallel lines (determinant is zero)
    if np.linalg.det(matrix) == 0:
        return None  # Parallel or coincident lines

    # Calculate the intersection point
    # print("matrix", matrix)
    # print("constants", constants)
    intersection = np.linalg.solve(matrix, constants)
    return (int(round(intersection[0])), int(round(intersection[1])))


def line_intersection_slope_y_intercept(line1, line2):
    """
    Finds the intersection point of two lines given in slope-intercept form.

    Parameters:
    - line1: tuple (m1, b1) for the first line
    - line2: tuple (m2, b2) for the second line

    Returns:
    - (x, y): The intersection point of the two lines
    """

    # Unpack the tuples
    m1, b1 = line1
    m2, b2 = line2

    # Construct the matrices for the system of equations
    A = np.array([[m1, -1], [m2, -1]])
    b = np.array([-b1, -b2])

    # Solve the system of equations
    x, y = np.linalg.solve(A, b)

    return np.array([int(round(x)), int(round(y))])


def sort_line_bottom_first(line):
    arr = line.reshape(2, 2)  # Reshape line into a 2x2 matrix [[x1, y1], [x2, y2]]
    if arr[0, 1] < arr[1, 1]:
        arr = arr[::-1]  # Reverse the order if the first point is higher than the second
    return arr.flatten()  # Flatten the array back into a 1D array


def is_point_in_frame(point, minx, miny, maxx, maxy):
    """
    Determine if a point is within a specified rectangular frame.

    This function checks if a given point (px, py) lies within a rectangular frame defined by its
    minimum and maximum x and y coordinates (minx, miny, maxx, maxy). It returns True if the point
    is inside the frame, including the boundaries, and False otherwise.

    Parameters:
    px (int or float): The x-coordinate of the point.
    py (int or float): The y-coordinate of the point.
    minx (int or float): The minimum x-coordinate of the frame.
    miny (int or float): The minimum y-coordinate of the frame.
    maxx (int or float): The maximum x-coordinate of the frame.
    maxy (int or float): The maximum y-coordinate of the frame.

    Returns:
    bool: True if the point (px, py) is within the frame defined by (minx, miny, maxx, maxy), False otherwise.
    """
    px, py = point
    return minx <= px <= maxx and miny <= py <= maxy


def connect_components(img, max_distance=10):
    """
    This function tries to connect components in a binary image in a given direction with a specified tolerance and maximum distance.
    It finds the closest edge points between components that are within the given angle tolerance and distance, and draws a line to connect them.

    Args:
        img (np.ndarray, dtype uint8): A binary image in which components are to be connected.
        frame (tuple(int, int)): The size (width, height) of the image.
        max_distance (int, optional): The maximum distance between components for a connection to be considered. Defaults to 10.

    Returns:
        (numpy.ndarray, dtype, uint8): An image with the same dimensions as the input, showing the original components and the connections between them.
    """
    # Find the connected components in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    print("number of components detected: ", num_labels - 1)  # -1 as the first label is the background

    # Create a new image for drawing the connections
    connected_img = np.copy(img)

    # todo look into a way to not compare all centroids and discard already connected centroids

    # Iterate over every pair of components to consider potential connections
    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):
            point1 = (int(centroids[i][0]), int(centroids[i][1]))
            point2 = (int(centroids[j][0]), int(centroids[j][1]))
            distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

            # Connect the components if the closest points are within the specified maximum distance
            if distance <= max_distance:
                # Draw a line on 'connected_img' from 'point1' to 'point2' with a color value of 1 and a thickness of 3 pixels.
                cv2.line(connected_img, point1, point2, 1, thickness=1)  # Draw a line to connect the components
                # debug_image([img, connected_img], t=10)

    return connected_img


def make_line_points(average, frame):
    """
    Calculate two points that define a line on an image.

    This function takes an image and a tuple representing the average (slope and y-intercept)
    of a set of lines. It calculates and returns two points (x1, y1) and (x2, y2) that define
    a line within the image. The line is determined algebraically using the slope and y-intercept,
    and is designed to be a specific length relative to the size of the image.

    Parameters:
    image (ndarray): The image on which the line will be drawn. This is used to determine the size of the image.
    average (tuple): A tuple of two elements (slope, y_int) representing the average slope and y-intercept of a set of lines.
    frame (xmax, ymax): clamp points to be within the image frame

    Returns:
    ndarray: A numpy array of four integers [x1, y1, x2, y2] representing two points that define a line within the image.
    """
    slope, y_int = average
    y1 = frame[0]
    y2 = int(y1 * 0)
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)

    return np.array([x1, y1, x2, y2])


def clamp_line_inside_frame(line, frame):
    # will get int overflow if not kept within image frame
    top_border_line = (0, 0, frame[1], 0)
    bottom_border_line = (0, frame[0], frame[1], frame[0])
    left_border_line = (0, 0, 0, frame[0])
    right_border_line = (frame[1], 0, frame[1], frame[0])

    intersections = []

    # todo fix fucked interface
    intersections.append(find_line_intersection(line, bottom_border_line))
    intersections.append(find_line_intersection(line, left_border_line))
    intersections.append(find_line_intersection(line, right_border_line))
    intersections.append(find_line_intersection(line, top_border_line))

    # the correct points will intersect border within the frame
    points = []
    for point in intersections:
        # todo fix fucked interface
        if is_point_in_frame(point, 0, 0, *frame[::-1]) and len(points) < 2:
            points.append(point)

    return [*points[0], *points[1]]
