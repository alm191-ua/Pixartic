# convert images to pixel art
import numpy as np
from scipy.ndimage import convolve
import os
import cv2
import random
import argparse
import pallettes.pallettes as pallettes

RESOLUTION = (128, 128)
FINAL_RESOLUTION = (512, 512)
KEEP_PROPORTIONS = True
BORDERS = False
SHADOWING = False
SHADOWING_OFFSET = 20
ZONE = 0 # divide pixels by zone, the zone is a square of ZONE x ZONE pixels
MARKS_OFFSET = 0 # mark zones in the image, doesn't need --zone-pixels option
MARK_COL0R = [0, 0, 0]
OUTPUT_PATH = None

DEFAULT_PALLETTE = pallettes.LOSPEC500
USED_PALLETTE = DEFAULT_PALLETTE

def get_palette(pallette_name):
    return getattr(pallettes, pallette_name)

def get_quantized_color(color, pallette=DEFAULT_PALLETTE) -> int:
    '''
    Get the index of the closest color in the pallette to the given color
    
    Parameters:
        color (np.ndarray): the color to quantize
        pallette (np.ndarray): the color pallette
    
    Returns:
        int: the index of the closest color in the pallette
    '''
    # calculate the distance to the colors in the pallette
    distances = np.linalg.norm(pallette - color, axis=1)
    # get the index of the closest color
    return int(np.argmin(distances))

def get_quantized_color_vectorized(res_image, used_palette):
    # res_image shape: (M, N, C)
    # used_palette shape: (K, C)
    
    # M, N, C = res_image.shape
    # K, _ = used_palette.shape
    
    # Expand dimensions to broadcast
    res_image_exp = np.expand_dims(res_image, axis=2)  # shape: (M, N, 1, C)
    palette_exp = np.expand_dims(used_palette, axis=0)  # shape: (1, 1, K, C)
    
    # Calculate the squared Euclidean distance
    distances = np.sum((res_image_exp - palette_exp) ** 2, axis=3)  # shape: (M, N, K)
    
    if SHADOWING:
        # -------
        # order distances
        distances_norm = np.argsort(distances, axis=2)
        # get the indexes of the values 0 and 1
        # do a search to get the indexes of the values 0 and 1
        indices = np.zeros((distances_norm.shape[0], distances_norm.shape[1], 2, 2), dtype=np.int32)
        # print(indices.shape)
        for i in range(distances_norm.shape[0]):
            for j in range(distances_norm.shape[1]):
                count = 0
                for k in range(distances_norm.shape[2]):
                    if count >= 2:
                        break
                    if distances_norm[i, j, k] == 0 or distances_norm[i, j, k] == 1:
                        indices[i, j, count] = [k, distances[i, j, k]]
                        # print(indices[i, j, count])
                        count += 1

        # compare the 2 colors, if the difference is too big, use the less different color
        # print(indices.shape)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                # print(np.abs(indices[i, j, 0, 1] - indices[i, j, 1, 1]))
                if np.abs(indices[i, j, 0, 1] - indices[i, j, 1, 1]) > SHADOWING_OFFSET:
                    # print(indices[i, j, 0, 1], indices[i, j, 1, 1])
                    if indices[i, j, 0, 1] > indices[i, j, 1, 1]:
                        indices[i, j, 0, 0] = -1
                    else:
                        indices[i, j, 1, 0] = -1

        return indices[:, :, :, 0]
        # -------
    else:
        # Get the indices of the minimum distances
        indices = np.argmin(distances, axis=2)  # shape: (M, N)
        
        return indices

# def rounded_dilation(image, kernel_size):
#     # get the shape of the image
#     shape = image.shape
#     # create a new image with the same shape
#     new_image = np.zeros_like(image)
#     # get the kernel size
#     k = kernel_size // 2
#     # iterate over the image
#     for i in np.arange(k, shape[0]-k):
#         for j in np.arange(k, shape[1]-k):
#             # get the window
#             window = image[max(i-k, 0):min(i+k+1, shape[0]), max(j-k, 0):min(j+k+1, shape[1])]
#             if np.sum(window[0]) != 0 and np.sum(window[-1]) \
#                 or np.sum(window[:, 0]) != 0 and np.sum(window[:, -1]):
#                 new_image[i, j] = 255

#     return new_image

def convolution(image, structuring_element, kernel_size):
    # Perform the convolution
    convolved_image = convolve(image, structuring_element, mode='constant', cval=0)
    
    # Create the new image based on the condition
    new_image = np.zeros_like(image, dtype=np.uint8)
    new_image[(convolved_image > 0) & (convolved_image < kernel_size**2)] = 255

    return new_image

def rounded_dilation_topleft(image, kernel_size, iterations=1):
    # Create a structuring element
    k = kernel_size // 2
    structuring_element = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    structuring_element[0, 0] = 1
    structuring_element[0, 1] = 1
    structuring_element[1, 0] = 1
    
    for _ in range(iterations):
        image = convolution(image, structuring_element, kernel_size)
    
    return image

def rounded_dilation_bottomright(image, kernel_size, iterations=1):
    # Create a structuring element
    k = kernel_size // 2
    structuring_element = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    structuring_element[-1, -1] = 1
    structuring_element[-1, -2] = 1
    structuring_element[-2, -1] = 1
    
    for _ in range(iterations):
        image = convolution(image, structuring_element, kernel_size)
    
    return image

def add_borders(image):
    '''
    Add borders to the elements in the image.
    Detects the edges and adds a border to the detected edges.

    Parameters:
        image (np.ndarray): the image to add borders to

    Returns:
        np.ndarray: the image with borders
    '''
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect the edges
    edges = cv2.Canny(gray, 100, 200)
    # dilate the edges
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    # dilatacion solo cuando hay valores blancos en ambos lados del pixel
    iteraciones = 5
    dilated = rounded_dilation_topleft(dilated, 21, iterations=iteraciones)
    # dilated = rounded_dilation_bottomright(dilated, 16, iterations=iteraciones)
    # dilated = cv2.dilate(dilated, np.ones((5, 5), np.uint8), iterations=5)
    # dilated = cv2.resize(dilated, FINAL_RESOLUTION, interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('Edges', dilated)
    # cv2.waitKey(0)
    # exit()

    # volver a sacar los bordes
    edges = cv2.Canny(dilated, 100, 200)
    dilated = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)

    # volver a sacar los bordes
    edges = cv2.Canny(dilated, 100, 200)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # add the dilated edges to the image
    image[dilated == 255] = 0
    cv2.imshow('Edges', dilated)
    cv2.waitKey(0)
    return image

def img2pixelart(image) -> np.ndarray:
    global RESOLUTION
    global FINAL_RESOLUTION
    if KEEP_PROPORTIONS:
        # get the aspect ratio
        ratio = image.shape[1] / image.shape[0]
        # calculate the new width
        new_width = int(RESOLUTION[1] * ratio)
        # if the new width is smaller than the resolution, calculate the new height
        if new_width < RESOLUTION[0]:
            new_height = int(RESOLUTION[0] / ratio)
            new_width = RESOLUTION[0]
        else:
            new_height = RESOLUTION[1]
        RESOLUTION = (new_width, new_height)

        # also calculate the final resolution
        new_width = int(FINAL_RESOLUTION[1] * ratio)
        if new_width < FINAL_RESOLUTION[0]:
            new_height = int(FINAL_RESOLUTION[0] / ratio)
            new_width = FINAL_RESOLUTION[0]
        else:
            new_height = FINAL_RESOLUTION[1]
        FINAL_RESOLUTION = (new_width, new_height)

    res_image = image
    if BORDERS:
        # add borders to the elements in the image
        res_image = add_borders(res_image)
    res_image = cv2.resize(res_image, RESOLUTION, interpolation=cv2.INTER_AREA)

    # change resolution to 32x32


    new_image = np.zeros_like(res_image)
    indices = get_quantized_color_vectorized(res_image, USED_PALLETTE)
    # print(indices.shape)
    # new_image = USED_PALLETTE[indices]

    # ------
    if SHADOWING:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                i1, i2 = indices[i, j]
                # print(i1, i2)
                if i2 == -1:
                    new_image[i, j] = USED_PALLETTE[i1]
                elif i1 == -1:
                    new_image[i, j] = USED_PALLETTE[i2]
                else:
                    # random choice
                    new_image[i, j] = USED_PALLETTE[random.choice([i1, i2])]
    else:
        new_image = USED_PALLETTE[indices]
    # ------
    new_image = new_image.astype(np.uint8)
       
    if MARKS_OFFSET > 0:
        new_image = markZones(new_image, MARKS_OFFSET)

    # upscale the image
    new_image = cv2.resize(new_image, FINAL_RESOLUTION, interpolation=cv2.INTER_NEAREST)

    return new_image

def zoneImage(image, zone):
    # get the shape of the image
    shape = image.shape
    # get the zone size
    zone_size = shape[0] // zone
    # create a new image with the same shape
    new_image = np.zeros_like(image)
    # iterate over the image
    for i in np.arange(0, shape[0], zone_size):
        for j in np.arange(0, shape[1], zone_size):
            # get the window
            window = image[i:i+zone_size, j:j+zone_size]
            # get the color of the window
            color = np.mean(window, axis=(0, 1))
            # fill the window with the color
            new_image[i:i+zone_size, j:j+zone_size] = color

    return new_image

def markZones(image, zone_size):
    # get the shape of the image
    shape = image.shape
    height_marks = shape[0] // zone_size +1
    width_marks = shape[1] // zone_size +1
    # create a new image with the height and width increased by the zone size
    new_image = np.zeros((shape[0] + height_marks, shape[1] + width_marks, 3), dtype=np.uint8)
    # fill the new image with the original image interpersed with the marks
    extra_rows = 0
    i = 0
    while i < shape[0]:
        extra_cols = 0
        j = 0
        marked_row = False
        while j < shape[1]:
            new_image[i+extra_rows, j+extra_cols] = image[i, j]
            if i % zone_size == 0:
                new_image[i+extra_rows, j+extra_cols] = MARK_COL0R
                if not marked_row:
                    marked_row = True
                    extra_rows += 1
                new_image[i+extra_rows, j+extra_cols] = image[i, j]
            
            if j % zone_size == 0:
                new_image[i+extra_rows, j+extra_cols] = MARK_COL0R
                extra_cols += 1
                new_image[i+extra_rows, j+extra_cols] = image[i, j]
                
            j += 1
        i += 1

    return new_image

if __name__ == '__main__':
    # get the image path with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='path to the image')
    parser.add_argument('-p', '--proportions', action='store_true', help='keep proportions')
    parser.add_argument('-r', '--resolution', nargs=2, type=int, help='resolution of the final image (less priority than zone)', required=False)
    # pallette argument with different options, show the option in the help
    choices = [p.rstrip('.py') for p in os.listdir('pallettes') if p != '__init__.py' and p.endswith('.py') and p != 'pallettes.py']
    parser.add_argument('-P', '--pallette', choices=choices, default='LOSPEC500', help='color pallette to use')
    # parser.add_argument('-b', '--borders', action='store_true', help='add borders to the elements in the image')
    parser.add_argument('-s', '--shadowing', action='store_true', help='use shadowing')
    parser.add_argument('-z', '--zone-pixels', type=int, help='divide pixels by zone, the final image is a square of ZONE x ZONE pixels (same as changing resolution but doing the mean between the pixels in each zone)', default=0, required=False, nargs=1)
    parser.add_argument('-m', '--marks_offset', type=int, help='mark zones in the image, doesn\'t need --zone-pixels option', default=0, required=False, nargs=1)
    parser.add_argument('-o', '--output', help='output path, saves the image in the given path instead of showing it', required=False)

    args = parser.parse_args()

    KEEP_PROPORTIONS = args.proportions
    if args.resolution:
        RESOLUTION = tuple(args.resolution)
    SHADOWING = args.shadowing

    ZONE = int(args.zone_pixels[0]) if args.zone_pixels else 0
    MARKS_OFFSET = int(args.marks_offset[0]) if args.marks_offset else 0

    OUTPUT_PATH = args.output

    USED_PALLETTE = get_palette(args.pallette)
    # BORDERS = args.borders

    image = cv2.imread(args.image)
    img_name = os.path.basename(args.image)
    new_image = img2pixelart(image)

    if ZONE > 0:
        new_image = zoneImage(new_image, ZONE)

    if OUTPUT_PATH and os.path.exists(OUTPUT_PATH):
        cv2.imwrite(os.path.join(OUTPUT_PATH, 'new_' + img_name), new_image)
    else:
        cv2.imshow('Pixel Art', new_image)
        cv2.waitKey(0)
