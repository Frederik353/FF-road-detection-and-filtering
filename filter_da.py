import cv2
import numpy as np



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
    return minx <= px <= maxx and miny <= py <= maxy

def is_point_in_frame(px, py, minx, miny, maxx, maxy):
    return minx <= px <= maxx and miny <= py <= maxy

def line_intersection(lines, frame):
    """
    Calculate the intersection point of two lines within a given frame.

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

    # Unpack points from the two lines [x1, y1, x2, y2] line format
    l1 = lines[0]
    l2 = lines[1]

    # Compute the intersection of the two lines
    intersection = find_line_intersection(*l1, *l2)

    # If there's no intersection or the intersection is outside the frame, return the original lines
    if intersection is None:
        return lines

    # Unpack the intersection point
    px, py = intersection

    # Check if the intersection point is within the boundaries of the frame
    if is_point_in_frame(px, py, 0, 0, frame[0], frame[1]):
        # If the intersection is within the frame, create new lines that extend to the intersection point
        return [[l1[0], l1[1], px, py], [l2[0], l2[1], px, py]]
    else:
        # If the intersection is outside the frame, return the original lines
        return lines

def average_line(image, lines):
    """
    Calculate the average left and right lines for a given set of lines on an image.

    This function processes a list of lines (each defined by two points) and classifies them
    as either part of the left or right side based on their slope. It then calculates the
    average slope and y-intercept for the lines on each side and uses these averages to
    create two average lines, one for the left and one for the right.

    Parameters:
    image (ndarray): The image where the lines are found. Used for determining the length and position of the average lines.
    lines (list of ndarray): A list of lines, where each line is represented by an ndarray of four integers [x1, y1, x2, y2].

    Returns:
    ndarray: A numpy array containing two lines (each an array of four integers), representing the average left and right lines.
    """

    left = []
    right = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Fit a linear polynomial to the x and y coordinates and retrieve the slope and y-intercept
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            # Lines with a negative slope are considered left lines, positive slopes are right lines
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))

    # Calculate the average slope and y-intercept for both the left and right lines
    right_avg = np.average(right, axis=0) if right else np.nan
    left_avg = np.average(left, axis=0) if left else np.nan

    # Generate the average left and right lines using the calculated averages, or default to a zero line if no average was found
    left_line = make_points(image, left_avg) if not np.isnan(left_avg).any() else [0,0,0,0]
    right_line = make_points(image, right_avg) if not np.isnan(right_avg).any() else [0,0,0,0]

    return np.array([left_line, right_line])

def make_points(image, average):
    """
    Calculate two points that define a line on an image.

    This function takes an image and a tuple representing the average (slope and y-intercept)
    of a set of lines. It calculates and returns two points (x1, y1) and (x2, y2) that define
    a line within the image. The line is determined algebraically using the slope and y-intercept,
    and is designed to be a specific length relative to the size of the image.

    Parameters:
    image (ndarray): The image on which the line will be drawn. This is used to determine the size of the image.
    average (tuple): A tuple of two elements (slope, y_int) representing the average slope and y-intercept of a set of lines.

    Returns:
    ndarray: A numpy array of four integers [x1, y1, x2, y2] representing two points that define a line within the image.
    """
    slope, y_int = average
    y1 = image.shape[0]
    y2 = int(y1 * 0)
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def draw_lines(image, lines, width=2):
    """ draws lines onto an image/ mask

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

def connect_components(img, max_distance=10):
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
        (numpy.ndarray, dtype, uint8): An image with the same dimensions as the input, showing the original components and the connections between them.
    """
    # Find the connected components in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Create a new image for drawing the connections
    connected_img = np.copy(img)

    print("number of lines detected: ", num_labels - 1) # -1 as the first label is the background

    # Iterate over every pair of components to consider potential connections
    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):

            point1 = (int(centroids[i][0]), int(centroids[i][0]))
            point2 = (int(centroids[j][0]), int(centroids[j][0]))
            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

            # Connect the components if the closest points are within the specified maximum distance
            if distance <= max_distance:
                print("points",point1, point2)
                cv2.line(connected_img, point1, point2, 1, 3)  # Draw a line to connect the components

    return connected_img

def mark_first_white_pixels(img, connectivity_threshold=2):
    """ this function scans from middle out to the left and right and mark first  pixel found and remove the rest

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

    # Find the first white pixel in each row for both halves
    left_indices = np.argmax(left_half == 1, axis=1)
    right_indices = np.argmax(right_half == 1, axis=1)

    # Correct indices for left half
    left_indices[left_indices > 0] = middle - left_indices[left_indices > 0]

    # Correct indices for right half
    right_indices[right_indices > 0] += middle

    # Marking the pixels with connectivity threshold
    for row in range(img.shape[0]):
        if left_indices[row] > 0:
            start_index = left_indices[row]
            marked_img[row, start_index - connectivity_threshold:start_index] = 1
        
        if right_indices[row] > 0:
            start_index = right_indices[row]
            marked_img[row, start_index:start_index + connectivity_threshold] = 1

    return marked_img

def approximate_lines(ll_seg_mask, frame):
    """ this function tries to improve the lane lines and section of the image

    Args:
        ll_seg_mask (numpy.ndarray, dtype, uint8): lane lines mask 0 means no lane line, 1 means lane line categorised by yolopv2
        expected shape: (720, 1280)  "720p"
        frame (tuple): shape of image

    Returns:
        (numpy.ndarray, dtype, uint8): corrected_da_mask
    """

    # scan from middle out to the left and right and mark first  pixel found and remove the rest
    removed_outer =  mark_first_white_pixels(ll_seg_mask)


    # tries to connect the lines in the mask, if they are close enough in angle and distance
    # img, pixel_max_distance
    connected = connect_components(removed_outer, 100)

    # same process as before but with the connected lines
    removed_outer =  mark_first_white_pixels(connected)

    # does a houghtransform to find lines in the mask
    lines = cv2.HoughLinesP(removed_outer, 2, np.pi/180, 100, minLineLength=20, maxLineGap=50)

    # might find many lines, take the average of them
    averaged_lines = average_line(ll_seg_mask, lines)

    # connects the two lines at the intersection point to section of the image
    lines_to_intersection = line_intersection(averaged_lines, frame)

    # draws them onto the image
    road_lines = draw_lines(ll_seg_mask, lines_to_intersection)

    return road_lines

def remove_da_outside_lines(da_seg_mask, approximate_lines, frame):
    """ removes the drivable area outside the improved road lines

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
    """ this is the main function filtering the drivable area (da)
    yolopv2 somtimes gives incomplete multiple and discontinous lines, it also somtimes miscattecgorises the grass on the side of the road as part of the road.
    This function tries to find a good aproximation for the road lines and then uses this to remove the da outside the lines

    Args:
        da_seg_mask (numpy.ndarray, dtype, int32): drivable area mask 0 means not drivable, 1 means drivable
        ll_seg_mask (numpy.ndarray, dtype, int32): lane lines mask 0 means no lane line, 1 means lane line
        expected shape of both: (720, 1280)  "720p"

    Returns:
        (numpy.ndarray, dtype, uint8): filtered drivable area mask
        (numpy.ndarray, dtype, uint8): improved road line mask
    """

    # Frame  (xmax, ymax)
    frame = da_seg_mask.shape

    # change dtype to uint8 expected by cv2
    ll_seg_mask = np.array(ll_seg_mask, dtype=np.uint8)
    da_seg_mask = np.array(da_seg_mask, dtype=np.uint8)



    # tries to improve lines and section of the image
    line_aproximation = approximate_lines(ll_seg_mask, frame)

    # removes the da outside the lines
    da_seg_mask = remove_da_outside_lines(da_seg_mask, line_aproximation, frame)

    return  da_seg_mask, line_aproximation


