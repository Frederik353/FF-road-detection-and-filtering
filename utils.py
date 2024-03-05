import cv2
import numpy as np
from typing import Tuple, Optional


def find_line_intersection(
    line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int]]:
    """
    Calculate the intersection point of two lines, each defined by a pair of points.

    Parameters:
    line1 (Tuple[int, int, int, int]): The x and y coordinates of the first and second points defining the first line.
                                        It should be provided as (x1, y1, x2, y2).
    line2 (Tuple[int, int, int, int]): The x and y coordinates of the first and second points defining the second line.
                                        It should be provided as (x3, y3, x4, y4).

    Returns:
    Optional[Tuple[int, int]]: The x and y coordinates of the intersection point, rounded to the nearest integer.
                                Returns None if the lines are nearly parallel or coincident (no unique intersection point).

    Example:
    >>> find_line_intersection((1, 2, 3, 4), (4, 3, 2, 1))
    (3, 3)
    """

    # Extract the coordinates from the input tuples
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Compute the coefficients A, B, and C for the first line using the formula: Ax + By = C
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    # Compute the coefficients for the second line
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # Set up the matrix and constant vector to represent the system of equations
    matrix = np.array([[A1, B1], [A2, B2]], dtype=np.float64)
    constants = np.array([C1, C2], dtype=np.float64)

    # Determine if the lines are nearly parallel by checking if the determinant of the matrix is close to zero
    determinant = np.linalg.det(matrix)
    if np.abs(determinant) < 1e-10:  # Threshold for considering lines as parallel
        return None  # The lines are nearly parallel or coincident, no unique intersection

    # Solve the system of equations to find the intersection point
    intersection = np.linalg.solve(matrix, constants)

    # Round the intersection coordinates to the nearest integer and return them
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

    # Unpack the slope and y-intercept for both lines
    m1, b1 = line1
    m2, b2 = line2

    # Setup the system of equations:
    # m1 * x - y = -b1
    # m2 * x - y = -b2
    # This can be represented in matrix form as AX = B, where A = [[m1, -1], [m2, -1]] and B = [-b1, -b2]
    A = np.array([[m1, -1], [m2, -1]])
    b = np.array([-b1, -b2])

    # Solve the system of equations to find the intersection point (x, y)
    # numpy.linalg.solve solves for X in AX = B
    x, y = np.linalg.solve(A, b)

    # Round the x and y coordinates to the nearest integer and return them
    return (int(round(x)), int(round(y)))

def sort_line_bottom_first(line):
    arr = line.reshape(2, 2)  # Reshape line into a 2x2 matrix [[x1, y1], [x2, y2]]
    if arr[0, 1] < arr[1, 1]:
        arr = arr[::-1]  # Reverse the order if the first point is higher than the second
    return arr.flatten()  # Flatten the array back into a 1D array


def is_point_in_frame(point: Tuple[float, float], frame: Tuple[float, float, float, float]) -> bool:
    """
    Determines if a point is inside a given rectangular frame.

    Parameters:
    - point (Tuple[float, float]): A tuple (px, py) representing the x and y coordinates of the point.
    - frame (Tuple[float, float, float, float]): A tuple (min_x, min_y, max_x, max_y) representing the coordinates
      of the bottom-left and top-right corners of the frame.

    Returns:
    - bool: True if the point is within the frame, including the edges; False otherwise.

    Example:
    >>> is_point_in_frame((5, 5), (0, 0, 10, 10))
    True
    """

    # Unpack the frame and point coordinates
    min_x, min_y, max_x, max_y = frame
    px, py = point

    # Check if the point's x coordinate is within the frame's x bounds and
    # the point's y coordinate is within the frame's y bounds
    return min_x <= px <= max_x and min_y <= py <= max_y


def connect_components(img: np.ndarray, max_distance: int = 10) -> np.ndarray:
    """
    Connects components in a binary image based on proximity. Each component is connected to its nearest neighbor
    if the distance between them is less than or equal to max_distance. This function modifies the input image.

    Args:
        img (np.ndarray): A binary image where components are to be connected. The array should be of dtype uint8.
        max_distance (int, optional): The maximum distance between components for a connection to be made. Defaults to 10.

    Returns:
        np.ndarray: An image with the same dimensions as the input, showing the original components and the connections between them.
    """

    # Clone the input image to avoid modifying the original image
    connected_img = img.copy()

    # Find the connected components in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # todo look into optimizations might be able to utilize the way cv2 finds components or numpy
    # Iterate over every pair of components to consider potential connections
    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):
            point1 = (int(centroids[i][0]), int(centroids[i][1]))
            point2 = (int(centroids[j][0]), int(centroids[j][1]))
            distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

            # Connect the components if they are within the specified maximum distance
            if distance <= max_distance:
                cv2.line(connected_img, point1, point2, 255, thickness=1)  # Draw a white line to connect the components

    return connected_img

def make_line_points(SI_line, frame):
    # Starting and ending x values
    m, b = SI_line
    x1, _, x2, _ = frame
    x_min, y_min, x_max, y_max = frame

    # Calculate corresponding y values
    y1 = m * x1 + b
    y2 = m * x2 + b

    # Check if y values are within the frame, adjust if necessary
    if y1 < 0:
        y1 = 0
        x1 = (y1 - b) / m
    elif y1 > y_max:
        y1 = y_max
        x1 = (y1 - b) / m

    if y2 < 0:
        y2 = 0
        x2 = (y2 - b) / m
    elif y2 > y_max:
        y2 = y_max
        x2 = (y2 - b) / m

    return np.array([int(x1), int(y1), int(x2), int(y2)])


# def make_line_points(SI_line: Tuple[float, float], frame: Tuple[int, int, int, int]) -> np.ndarray:
#     """
#     Calculate two points that define a line within a frame based on its slope and y-intercept.

#     Parameters:
#     - SI_line (Tuple[float, float]): A tuple containing the slope and y-intercept of the line.
#     - frame (Tuple[int, int, int, int]): A tuple defining the frame within which the line is calculated,
#       specified as (xmin, ymin, xmax, ymax).

#     Returns:
#     - np.ndarray: An array containing the coordinates of the two endpoints [x1, y1, x2, y2].
#     """

#     slope, y_intersect = SI_line
#     y1 = frame[3]
#     y2 = int(y1 * 0)
#     x1 = int((y1 - y_intersect) // slope)
#     x2 = int((y2 - y_intersect) // slope)

#     print(x1, y1, x2, y2)
#     return np.array([x1, y1, x2, y2])


def clamp_line_inside_frame(
    line: Tuple[int, int, int, int], frame: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """
    Adjusts a line to ensure its endpoints are within a specified rectangular frame by finding the line's intersections
    with the frame's borders and selecting valid intersection points.

    Parameters:
    - line (Tuple[int, int, int, int]): The line to be clamped, represented by its endpoints (x1, y1, x2, y2).
    - frame (Tuple[int, int, int, int]): The frame within which to clamp the line, specified by (xmin, ymin, xmax, ymax).

    Returns:
    - Tuple[int, int, int, int]: The endpoints of the adjusted line within the frame, represented as [x1, y1, x2, y2].
    """

    # Define lines representing the borders of the frame
    top_border_line = (0, 0, frame[2], 0)
    bottom_border_line = (0, frame[3], frame[2], frame[3])
    left_border_line = (0, 0, 0, frame[3])
    right_border_line = (frame[2], 0, frame[2], frame[3])

    intersections = []

    # Find intersections of the line with each of the frame borders
    for border_line in [bottom_border_line, left_border_line, right_border_line, top_border_line]:
        intersection = find_line_intersection(line, border_line)
        if intersection:
            intersections.append(intersection)

    # Filter out intersection points that are outside the frame
    points = [point for point in intersections if is_point_in_frame(point, frame)]

    # Handle cases where the line doesn't intersect with two borders
    if len(points) < 2:
        # raise ValueError("The line does not intersect the frame in two distinct points.")
        # clamp_point_to_frame(points[0], frame)
        return None

    return [*points[0], *points[1]]


def clamp_point_to_frame(point, frame):
    """
    Clamp a point to be within the boundaries of an image.

    Parameters:
    x (int): The x-coordinate of the point.
    y (int): The y-coordinate of the point.
    image_width (int): The width of the image.
    image_height (int): The height of the image.

    Returns:
    tuple: A tuple (x, y) where the point coordinates are clamped within the image dimensions.
    """
    x, y = point
    x_min, y_min, x_max, y_max = frame

    x_clamped = max(y_min, min(x, x_max - 1))
    y_clamped = max(y_min, min(y, y_max - 1))
    return (x_clamped, y_clamped)
