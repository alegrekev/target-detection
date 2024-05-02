import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt

def recognize_dominant_color(image_path, targ_coords):
    """
    Function that runs k-means algorithm get the dominant color of the target.

    Args:
        image_path (str): The path to the input image.
        targ_coords (list): Coordinates, width, height of the target.

    Returns:
        dominant_color_hsv (tuple): HSV of the dominant color.
    """
    # Read the image
    x, y, w, h = targ_coords
    image = cv2.imread(image_path)
    
    image = image[y:y+h, x:x+w]

    # Convert the image from BGR to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Reshape the image to form a list of pixels
    pixels = image_hsv.reshape(-1, 3)
    
    # Use k-means clustering to find the most dominant color
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(pixels)
    
    # Get the HSV values of the dominant color
    dominant_color_hsv = kmeans.cluster_centers_[1]

    return dominant_color_hsv

def get_color_name(hsv_tuple):
    """
    Function that converts HSV to color name.

    Args:
        hsv_tuple (tuple): Tuple with color in HSV format.

    Returns:
        color (string): Color name of the HSV.
    """
    # Predefined colors in HSV space
    # These do not work most of the time, values given by GPT no reason behind them
    predefined_colors_hsv = {
        "blue": (120, 255, 255),
        "red": (0, 255, 255),
        "green": (60, 255, 255),
        "purple": (150, 255, 255),
        #"brown": (15, 165, 165),
        "orange": (30, 255, 255)
    }

    dmetric = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) # metric used to determine distance between colors
    distances = {k: dmetric(v, hsv_tuple) for k, v in predefined_colors_hsv.items()}
    color = min(distances, key=distances.get)

    return color