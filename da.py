import numpy as np
import matplotlib.pyplot as plt
import time



# Function to filter drivable area based on segmentation masks
def filter_da(da_seg_mask, ll_seg_mask):
    """
    Filters the drivable area based on segmentation masks.
    
    Parameters:
    - da_seg_mask: Drivable area segmentation mask.
    - ll_seg_mask: Lane line segmentation mask.
    
    Returns:
    - filtered_da_seg_mask: Filtered drivable area segmentation mask.
    """

    # Assertions to ensure the masks have the same dimensions
    assert len(da_seg_mask) == len(ll_seg_mask)
    assert len(da_seg_mask[0]) == len(ll_seg_mask[0])
    t = time.time()

    #columns, rows = da_seg_mask.shape
    # Get dimensions of the masks
    # # ny = len(da_seg_mask)
    nx = len(da_seg_mask[0])

    # Define safety region and pixel cutoff parameters
    safety_region_percentage = 1 / 5
    # # pixel_cutoff = 1 / 2

    # Calculate middle column and safety regions
    middle_column = nx // 2
    safety_region_left = int(middle_column - nx * safety_region_percentage)
    safety_region_right = int(middle_column + nx * safety_region_percentage)

    print("safety regions")
    print("left", safety_region_left)
    print("right", safety_region_right)
    print()

    # Create a copy of the drivable area segmentation mask
    filtered_da_seg_mask = np.copy(da_seg_mask)

    # Define indices for the safety regions
    mask_indices_left = np.arange(0, safety_region_left, 1)
    mask_indices_right = np.arange(safety_region_right, nx, 1)
    # print("mask indicies")
    # print("left ",mask_indices_left)
    # print("right ", mask_indices_right)
    # print()

    #negated mask
    # Create masks for the safety regions
    mask_left = np.zeros_like(da_seg_mask, dtype=bool)
    mask_right = np.zeros_like(da_seg_mask, dtype=bool)


    # Apply lane line segmentation mask to the safety regions
    mask_left[:, mask_indices_left] = ll_seg_mask[:, mask_indices_left]
    mask_right[:, mask_indices_right] = ll_seg_mask[:, mask_indices_right]



    # Update masks based on cumulative sum
    mask_left = (np.cumsum(mask_left[:, ::-1], axis=1) > 0)[:, ::-1]
    mask_right = np.cumsum(mask_right, axis=1) > 0

    # Combine the masks and apply to the drivable area segmentation mask
    mask = np.logical_and(np.logical_not(mask_left), np.logical_not(mask_right))

    filtered_da_seg_mask *= mask


    te = time.time()
    print(f'Time filter da: {te - t}')

    # return da_seg_mask
    # Return the filtered drivable area segmentation mask
    return filtered_da_seg_mask 
    # return ll_seg_mask 


def print_mask(s, mask):
    print(s)
    for i in mask:
        for j in i:
            if j == True:
                print(f" 1 ", end="")
            elif j == False:
                print(f" 0 ", end="")
            else:
                print(f" {j} ", end="")
        print()
    print()










large = True


# Create simple 2D arrays for visualization
da_seg_mask = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
])

ll_seg_mask = np.array([
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0]
])

# if large:
#     size = 500
#     # Create larger 2D arrays for visualization (50x50)
#     da_seg_mask_large = np.ones((size, size))

#     ll_seg_mask_large = np.zeros((size, size))
#     # Adding lane lines in the large mask
#     for i in range(size):
#         if i % 10 == 0 or i % 10 == 9:  # Creating multiple lane lines
#             ll_seg_mask_large[:, i] = 1


# Apply the function to get the filtered drivable area


# if not large:
    # filtered_da = filter_da(da_seg_mask, ll_seg_mask)
    # print_mask("da_seg_mask", da_seg_mask_large)
    # print_mask("ll_seg_mask", ll_seg_mask_large)
    # print_mask("filtered", filtered_da)

    # print("safty region masks")
    # print_mask("left", mask_left)
    # print_mask("right", mask_right)
    # gjør foreløpig ingenting
    # print("lane line")
    # print("left")
    # print_mask(mask_left)
    # print("right")
    # print_mask(mask_right)
    # print("cumsum")
    # print_mask("left", mask_left)
    # print_mask("right", mask_right)
    # print_mask("mask", mask)


def plot(da_seg_mask, ll_seg_mask, filtered_da):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(da_seg_mask, cmap='gray')
    ax[0].set_title('Drivable Area Mask')
    ax[1].imshow(ll_seg_mask, cmap='gray')
    ax[1].set_title('Lane Line Mask')
    ax[2].imshow(filtered_da, cmap='gray')
    ax[2].set_title('Filtered Drivable Area')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

if large:
    i = 1
    file_path = f"iteration/{i}.npz"
    npzfile = np.load(file_path)

    ll_seg_mask = npzfile["ll_seg_mask"]
    sizex = len(ll_seg_mask)
    sizey = len(ll_seg_mask[0])
    # da_seg_mask = np.zeros((sizex, sizey))
    # da_seg_mask = np.full((sizex, sizey), 1)

    da_seg_mask = npzfile["da_seg_mask"]
    filtered_da = filter_da(da_seg_mask, ll_seg_mask)
    # Visualize the input and output
    plot(da_seg_mask,ll_seg_mask, filtered_da)