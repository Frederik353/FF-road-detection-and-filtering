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

    print("number of lines detected: ", num_labels - 1)  # -1 as the first label is the background

    # Iterate over every pair of components to consider potential connections
    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):
            # Find the closest edge points between components that match the specified direction and angle tolerance
            pt1, pt2, distance = find_closest_edge_points(i, j, labels, desired_angle_rad, angle_tolerance_rad, frame)

            # Connect the components if the closest points are within the specified maximum distance
            if pt1 is not None and distance <= max_distance:
                cv2.line(
                    connected_img, tuple(pt1[::-1]), tuple(pt2[::-1]), 1, 3
                )  # Draw a line to connect the components

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
        else:
            center = -1

        for pt2 in edge_points2:
            # Calculate the vector from pt1 to pt2
            vector = pt2 - pt1
            distance = np.linalg.norm(vector)
            angle = np.arctan2(vector[1], vector[0])

            # Check if the angle is within the tolerance
            if (
                desired_angle_rad * center - angle_tolerance_rad
                <= angle
                <= desired_angle_rad * center + angle_tolerance_rad
            ):
                # Update the closest points if the distance is less than the minimum found so far
                if distance < min_distance:
                    min_distance = distance
                    closest_point1 = pt1
                    closest_point2 = pt2

    return closest_point1, closest_point2, min_distance


# convert into grey scale image
def grey(image):
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Gaussian blur to reduce noise and smoothen the image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# Canny edge detection
def canny(image, nx, ny):
    edges = cv2.Canny(image, 1, 1)
    return edges


def region(image):
    height, width = image.shape
    triangle = np.array([[(100, height), (475, 325), (width, height)]])

    mask = np.zeros_like(image)

    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


# Skeletonize the image
# skeleton = skeletonize(np_image)
# cleaned_skeleton = remove_small_components(skeleton, 10)


def skeletonize(img):
    # Ensure the image is binary
    # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    # Get a cross-shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Repeat until there are no changes in the image
    done = False
    skel = np.zeros(img.shape, np.uint8)
    while not done:
        # Perform erosion
        eroded = cv2.erode(img, element)
        # Perform opening on the eroded image
        temp = cv2.dilate(eroded, element)
        # Subtract the opened image from the original image
        temp = cv2.subtract(img, temp)
        # Bitwise OR the temp image (skeleton part) with the skel image
        skel = cv2.bitwise_or(skel, temp)
        # Update the image for the next iteration
        img = eroded.copy()

        # If the image is fully eroded, we are done
        done = cv2.countNonZero(img) == 0

    return skel


def remove_small_components(image, min_size_threshold):
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # Create an output image that will hold the cleaned skeleton
    clean = np.zeros_like(image)

    # Go through all found components
    for i in range(1, num_labels):  # starting from 1 to ignore the background
        if stats[i, cv2.CC_STAT_AREA] >= min_size_threshold:
            # If the component is larger than the threshold, add it to the output image
            clean[labels == i] = 1

    return clean


def plot(da_seg_mask, ll_seg_mask, filtered_da):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(da_seg_mask, cmap="gray")
    ax[0].set_title("o1")
    ax[1].imshow(ll_seg_mask, cmap="gray")
    ax[1].set_title("o2")
    ax[2].imshow(filtered_da, cmap="gray")
    ax[2].set_title("o3")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def mark_first_white_pixels(img, connectivity_threshold=2):
    """this function scans from middle out to the left and right and mark first  pixel found and remove the rest

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
            marked_img[row, start_index - connectivity_threshold : start_index] = 1

        if right_indices[row] > 0:
            start_index = right_indices[row]
            marked_img[row, start_index : start_index + connectivity_threshold] = 1

    return marked_img


# ! depricated was used to deal with intager overflow in find_line_intersection but unlikley with new method and cv2 handles drawing lines with parts outside the image fine
def clamp_lines_inside_frame(left_line, right_line, frame):
    # will get int overflow if not kept within image frame
    top_border_line = (0, 0, frame[1], 0)
    bottom_border_line = (0, frame[0], frame[1], frame[0])
    left_border_line = (0, 0, 0, frame[0])
    right_border_line = (frame[1], 0, frame[1], frame[0])

    # assuming slope not steep so likly to cross left and right as opposed to top and bottom

    # left top point avg
    left_p1 = find_line_intersection(*left_line, *right_border_line)
    if not is_point_in_frame(*left_p1, 0, 0, *frame[::-1]):
        left_p1 = find_line_intersection(*left_line, *top_border_line)

    # left bottom point avg
    left_p2 = find_line_intersection(*left_line, *left_border_line)
    if not is_point_in_frame(*left_p2, 0, 0, *frame[::-1]):
        left_p2 = find_line_intersection(*left_line, *bottom_border_line)

    # right top point avg
    right_p1 = find_line_intersection(*right_line, *left_border_line)
    if not is_point_in_frame(*right_p1, 0, 0, *frame[::-1]):
        right_p1 = find_line_intersection(*right_line, *top_border_line)

    # right bottom point avg
    right_p2 = find_line_intersection(*right_line, *right_border_line)
    if not is_point_in_frame(*right_p2, 0, 0, *frame[::-1]):
        right_p2 = find_line_intersection(*right_line, *bottom_border_line)

    # todo change to left right line same for line intersection for consitency
    averaged_lines = [[*left_p2, *left_p1], [*right_p2, *right_p1]]
    return averaged_lines


def calculate_optical_flow(prev, current):

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev,
        next=current,
        flow=None,
        pyr_scale=0.75,
        levels=5,
        winsize=100,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Compute magnitude and angle (currently unused) of the flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize magnitude for visualization
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Normalize angle (0 to 2*pi) to fit into the Hue channel (0-180)
    hue = angle * (180 / np.pi) / 2

    # Set saturation to maximum
    saturation = np.ones_like(magnitude) * 255

    # Normalize magnitude for Value channel
    value = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Stack channels to form HSV image, convert types as needed
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.uint8)

    # Convert HSV to BGR (or RGB) for visualization
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ------------------------------------------------

    # Create a copy of the current frame for drawing (convert to color if necessary)
    if len(current.shape) == 2 or current.shape[2] == 1:  # if the image is grayscale
        frame_with_arrows = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
    else:
        frame_with_arrows = np.copy(current)

    # Parameters for arrow drawing
    scale = 1  # Scale factor for the length of the arrows

    # Iterate through each pixel
    for y in range(flow.shape[0]):
        for x in range(flow.shape[1]):
            # Determine the end point of the vector
            if np.sqrt(flow[y, x, 0] ** 2 + flow[y, x, 1] ** 2) < 1:
                continue
            end_point = (int(x + flow[y, x, 0] * scale), int(y + flow[y, x, 1] * scale))
            cv2.arrowedLine(frame_with_arrows, (x, y), end_point, (0, 255, 0), thickness=1, tipLength=0.3)

    return normalized_magnitude, frame_with_arrows


prev_magnitudes = []
prev_mask = []

global prev_mask, prev_magnitudes



    # optical flow
    if len(prev_magnitudes) < 2:
        average_magnitude = np.zeros_like(line_aproximation)
        prev_magnitudes.append(average_magnitude)
    else:
        magnitudes, vis = calculate_optical_flow(prev_mask, ll_seg_mask)
        magnitudes = np.array(magnitudes)
        prev_magnitudes.append(magnitudes)
        prev_magnitudes = prev_magnitudes[-10:]

        average_magnitude = np.mean(prev_magnitudes)
        debug_image([ll_seg_mask, prev_mask, vis], t=1000)
    prev_mask = ll_seg_mask




def cast_ray(mask, start, angle_deg, max_distance=1000):
    angle_rad = np.radians(angle_deg)
    step = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    current_pos = np.array(start, dtype=np.float32)

    for _ in range(max_distance):
        current_pos += step
        x, y = int(current_pos[0]), int(current_pos[1])

        if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
            # break  # Ray is out of bounds
            return (x - 2, y - 2)

        if mask[y, x] != 0:
            return (x, y)  # First non-zero pixel found

    return None  # No non-zero pixel found


def raytrace_from_point(mask, start, angular_resolution=1):
    # hits = []
    marked = np.zeros_like(mask)
    for angle in range(0, 180, angular_resolution):
        hit = cast_ray(mask, start, angle)
        # hits.append(hit)
        cv2.circle(marked, hit, 1, (255, 0, 0), 10)  # Draw each hit
        cv2.line(marked, start, hit, (255, 0, 0), 1)  # Draw the ray path

    debug_image([marked], t=10_000)
    return marked


def average_lines(image, lines, frame):
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

    # todo debug and reread this function, super tired now

    left, right = [], []

    # Define the midpoint in the x-direction
    # Assuming 'frame' is a tuple or list with (height, width), hence frame[1] is the width
    mid_x = frame[1] // 2

    # Process each line to sort by side and then by correct slope direction
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # Calculate slope
        # todo not suited since simular x
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]

        # Check if line is on the left side of the screen
        if x1 < mid_x and x2 < mid_x:
            # Further check if the slope is negative (expected direction for left)
            if slope < 0:
                left.append((slope, y_int))
        # Check if line is on the right side of the screen
        elif x1 >= mid_x and x2 >= mid_x:
            # Further check if the slope is positive (expected direction for right)
            if slope > 0:
                right.append((slope, y_int))

    # Calculate the average slope and y-intercept for both the left and right lines
    right_avg = np.average(right, axis=0) if right else np.nan
    left_avg = np.average(left, axis=0) if left else np.nan

    # Generate the average left and right lines using the calculated averages, or default to a zero line if no average was found
    left_line = make_line_points(image, left_avg, frame) if not np.isnan(left_avg).any() else None
    right_line = make_line_points(image, right_avg, frame) if not np.isnan(right_avg).any() else None

    return left_line, right_line

def clamp_line_inside_frame(line, frame):
    # will get int overflow if not kept within image frame
    top_border_line = (0, 0, frame[1], 0)
    bottom_border_line = (0, frame[0], frame[1], frame[0])
    left_border_line = (0, 0, 0, frame[0])
    right_border_line = (frame[1], 0, frame[1], frame[0])

    # assuming slope not steep so likly to cross left and right as opposed to top and bottom

    # left top point avg
    if left_line is not None:
        left_p1 = utils.find_line_intersection(*left_line, *right_border_line)
        if not utils.is_point_in_frame(*left_p1, 0, 0, *frame[::-1]):
            left_p1 = utils.find_line_intersection(*left_line, *top_border_line)

        # left bottom point avg
        left_p2 = utils.find_line_intersection(*left_line, *left_border_line)
        if not utils.is_point_in_frame(*left_p2, 0, 0, *frame[::-1]):
            left_p2 = utils.find_line_intersection(*left_line, *bottom_border_line)

        left_line = [*left_p2, *left_p1]

    # right top point avg
    if right_line is not None:
        right_p1 = utils.find_line_intersection(*right_line, *left_border_line)
        if not utils.is_point_in_frame(*right_p1, 0, 0, *frame[::-1]):
            right_p1 = utils.find_line_intersection(*right_line, *top_border_line)

        # right bottom point avg
        right_p2 = utils.find_line_intersection(*right_line, *right_border_line)
        if not utils.is_point_in_frame(*right_p2, 0, 0, *frame[::-1]):
            right_p2 = utils.find_line_intersection(*right_line, *bottom_border_line)

        right_line = [*right_p2, *right_p1]

    # todo change to left right line same for line intersection for consitency
    averaged_lines = [left_line, right_line]
    return averaged_lines
