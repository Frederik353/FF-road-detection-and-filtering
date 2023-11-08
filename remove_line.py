import numpy as np
import matplotlib.pyplot as plt
import time
import cv2


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


def average(image, lines):
    left = []
    right = []

    if lines is not None:
        for line in lines:
            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            # fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            # print(parameters)
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
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])


def compute_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate the coefficients for the equations of the lines
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # If den is zero, lines are parallel and have no intersection within any frame
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
    minx, miny, maxx, maxy = frame
    
    # Compute the intersection
    intersection = compute_intersection(*l1, *l2)
    
    # If there's no intersection, return the lines as they are
    if intersection is None:
        return lines
    
    # If the intersection point is within the frame, return that point
    px, py = intersection
    if is_point_in_frame(px, py, minx, miny, maxx, maxy):
        return [[l1[0], l1[1], px, py], [l2[0], l2[1], px, py]]
    else:
        return lines


def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    # y2 = int(y1 * (3/5))
    y2 = int(y1*0)
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


def display_lines(image, lines):

    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # x1, y1, x2, y2 = line[0] # hvis input lines
            x1, y1, x2, y2 = line # hvis input averaged lines
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

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

def filter_da(da_seg_mask, ll_seg_mask):
    t = time.time()
    ny = len(da_seg_mask)
    nx = len(da_seg_mask[0])
    # Frame boundaries (xmin, ymin, xmax, ymax)
    frame = (0, 0, nx, ny)



    copy = np.copy(ll_seg_mask)

    np_image = np.array(ll_seg_mask, dtype=np.uint8)


    te = time.time()

    # Assuming ll_seg_mask is your binary image, convert it to a numpy array
    np_image = np.array(ll_seg_mask, dtype=np.uint8)

    # Skeletonize the image
    skeleton = skeletonize(np_image)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

    # Set a minimum size threshold for components to keep
    min_size_threshold = 10  # this is an example value; adjust it according to your needs

    # Create an output image that will hold the cleaned skeleton
    cleaned_skeleton = np.zeros_like(skeleton)

    # Go through all found components
    for i in range(1, num_labels):  # starting from 1 to ignore the background
        if stats[i, cv2.CC_STAT_AREA] >= min_size_threshold:
            # If the component is larger than the threshold, add it to the output image
            cleaned_skeleton[labels == i] = 255

    return cleaned_skeleton, cleaned_skeleton


def plot(da_seg_mask, ll_seg_mask, filtered_da):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(da_seg_mask, cmap='gray')
    ax[0].set_title('o1')
    ax[1].imshow(ll_seg_mask, cmap='gray')
    ax[1].set_title('o2')
    ax[2].imshow(filtered_da, cmap='gray')
    ax[2].set_title('o3')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()


list_of_images = []
# max 150
for i in range(1, 20):

    try:
        file_path = f"iteration/{i}.npz"
        npzfile = np.load(file_path)

        ll_seg_mask = npzfile["ll_seg_mask"]
        sizex = len(ll_seg_mask)
        sizey = len(ll_seg_mask[0])

        da_seg_mask = npzfile["da_seg_mask"]
        masks = filter_da(da_seg_mask, ll_seg_mask)
        list_of_images.append(masks)
    except FileNotFoundError:
        print("file not found ", i)


cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
for i, image in enumerate(list_of_images):
    # BGR colors for each mask
    colors = [(255, 0, 0), (255, 255, 255), (0, 0, 255)]

    # # Create an empty canvas with 3 channels (for BGR)
    # height, width = image[0].shape
    # overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # # Create an empty canvas with 3 channels (for BGR)
    height, width = image[0].shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # # Overlay each mask with a different color
    for mask, color in zip(image, colors):
        overlay[mask == 1] = color

    # Display the result
    # cv2.imshow(f'imag', overlay)

    cv2.imshow('Image Window', overlay)
    # cv2.imshow('Image Window', mask[0])

    cv2.waitKey(50)

cv2.destroyAllWindows()
