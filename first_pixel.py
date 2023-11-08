import numpy as np
import cv2

def mark_first_white_pixels(img):
    # Ensure the image is in binary form
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find the middle row
    middle_row = binary_img.shape[0] // 2

    # Create an output image filled with zeros
    marked_img = np.zeros_like(binary_img)

    # Find indices of the first white pixel in each row by using a vectorized approach
    # Left half (from center to left)
    left_indices = np.argmax(binary_img[:, :binary_img.shape[1]//2] == 255, axis=1)
    left_indices[left_indices != 0] += binary_img.shape[1]//2 - 1  # Adjust indices because of slicing

    # Right half (from center to right)
    right_indices = np.argmax(binary_img[:, binary_img.shape[1]//2:] == 255, axis=1) + binary_img.shape[1]//2

    # Mark the found white pixels
    for row in range(middle_row, binary_img.shape[0]):
        if binary_img[row, left_indices[row]] == 255:
            marked_img[row, left_indices[row]] = 255
        if binary_img[row, right_indices[row]] == 255:
            marked_img[row, right_indices[row]] = 255

    for row in range(middle_row, -1, -1):
        if binary_img[row, left_indices[row]] == 255:
            marked_img[row, left_indices[row]] = 255
        if binary_img[row, right_indices[row]] == 255:
            marked_img[row, right_indices[row]] = 255

    return marked_img

# Assuming 'np_image' is a binary NumPy array of the image
marked_image = mark_first_white_pixels(np_image)

# Show the result
cv2.imshow('Marked Image', marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()