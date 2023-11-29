import numpy as np
import cv2


def detect_dominant_color(image, k=1):
    """
    Function to detect the dominant color in an image using k-means clustering.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        k (int): The number of dominant colors to detect using k-means clustering.

    Returns:
        str: The label of the dominant color in the image.
    """
    threshold_image = cv2.threshold(cropped_grayscale_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    # Sort the dominant colors by frequency
    labels, counts = np.unique(label, return_counts=True)
    sorted_labels = labels[np.argsort(-counts)]  # Sort in descending order of frequency
    sorted_colors = [get_color_label(center[label]) for label in sorted_labels]

    # Find the most frequent dominant color
    dominant_color = sorted_colors[0] if sorted_colors else "unknown"
    return dominant_color


def get_color_label(color_rgb):
    color_ranges = {
         "white": [(0, 0, 200), (179, 30, 255)],  # HSV values for white
         "black": [(0, 0, 0), (179, 30, 50)],     # HSV values for black
         "red": [(0, 100, 100), (10, 255, 255)],  # HSV values for red (lower and upper bounds)
         "blue": [(90, 50, 50), (130, 255, 255)],  # HSV values for blue (lower and upper bounds)
         "green": [(35, 100, 100), (85, 255, 255)],  # HSV values for green (lower and upper bounds)
         "purple": [(125, 100, 100), (155, 255, 255)],  # HSV values for purple (lower and upper bounds)
         "brown": [(10, 100, 100), (30, 255, 255)],  # HSV values for brown (lower and upper bounds)
         "orange": [(5, 100, 100), (20, 255, 255)],  # HSV values for orange (lower and upper bounds)
         "any": [(0, 0, 0), (179, 255, 255)]
}

def detect_colors_in_hulls(image, hulls, k=3):
    """
    Function to detect the dominant color within the specified hulls in an image.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        hulls (list): A list of convex hulls (contours) to analyze.
        k (int): The number of dominant colors to detect using k-means clustering.

    Returns:
        str: The label of the dominant color within the hulls.
    """
    detected_color = "unknown"
    # x, y, w, h = 0, 0, 0, 0
    for hull in hulls:
        # Get the bounding rectangle for the current hull
        x, y, w, h = cv2.boundingRect(hull)
        # Extract the region of interest (ROI) from the image
        roi = image[y:y + h, x:x + w]

        cv2.imwrite('target-detection/images1/b.jpg', roi)

            # Detect the dominant color within the ROI
        dominant_color_roi = detect_dominant_color(roi, k)

            # If a dominant color is found in the ROI, use it as the detected color
        if dominant_color_roi != "unknown":
            detected_color = dominant_color_roi
            break
        

    return detected_color

    

    # color_ranges = {
    #     "white": [(0, 0, 200), (179, 30, 255)],  # HSV values for white
    #     "black": [(0, 0, 0), (179, 30, 50)],     # HSV values for black
    #     "red": [(0, 100, 100), (10, 255, 255)],  # HSV values for red (lower and upper bounds)
    #     "blue": [(90, 50, 50), (130, 255, 255)],  # HSV values for blue (lower and upper bounds)
    #     "green": [(35, 100, 100), (85, 255, 255)],  # HSV values for green (lower and upper bounds)
    #     "purple": [(125, 100, 100), (155, 255, 255)],  # HSV values for purple (lower and upper bounds)
    #     "brown": [(10, 100, 100), (30, 255, 255)],  # HSV values for brown (lower and upper bounds)
    #     "orange": [(5, 100, 100), (20, 255, 255)],  # HSV values for orange (lower and upper bounds)
    #     "any": [(0, 0, 0), (179, 255, 255)]
    # }

    # def is_color_in_range(color, color_range):
    #     return all(color_range[0] <= color <= color_range[1] for color, color_range in zip(color, color_range))

    # detected_colors = []

    # for hull in hulls:
    #     # Get the bounding rectangle for the current hull
    #     x, y, w, h = cv2.boundingRect(hull)

    #     # Extract the region of interest (ROI) from the image
    #     roi = image[y:y + h, x:x + w]

    #     # Calculate the average color within the ROI
    #     average_color = np.mean(roi, axis=(0, 1)).astype(int)

    #     # Find the closest color based on the average color
    #     closest_color = None
    #     for color_label, color_range in color_ranges.items():
    #         if is_color_in_range(average_color, color_range):
    #             closest_color = color_label
    #             break

    #     if closest_color is not None:
    #         detected_colors.append(closest_color)

    # return detected_colors