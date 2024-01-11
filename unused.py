
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


def remove_small_components(image ,min_size_threshold):
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

