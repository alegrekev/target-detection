import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

def create_mask(image, lower_hue, lower_saturation, lower_value, upper_hue, upper_saturation, upper_value):
    """
    Function to create a mask based on the given HSV values.

    Args:
        image (numpy.ndarray): The input image in HSV color space.
        lower_hue (int): The lower bound for the Hue component (0-179).
        lower_saturation (int): The lower bound for the Saturation component (0-255).
        lower_value (int): The lower bound for the Value component (0-255).
        upper_hue (int): The upper bound for the Hue component (0-179).
        upper_saturation (int): The upper bound for the Saturation component (0-255).
        upper_value (int): The upper bound for the Value component (0-255).

    Returns:
        mask (numpy.ndarray): The resulting mask with values indicating which pixels in the input image fall within the specified HSV range.
    """
    lower_bound = np.array([lower_hue, lower_saturation, lower_value])  # creates array of lower HSV values
    upper_bound = np.array([upper_hue, upper_saturation, upper_value])  # creates array of upper HSV values
    mask = cv2.inRange(image, lower_bound, upper_bound)  # creates mask of every color within bounds

    return mask

def find_letter(image):
    return pytesseract.image_to_string(image, lang = 'eng', config='--psm 7')

def find_shape(contour, rect_location_x, rect_location_y):
    """
    Function to detect shapes of targets.

    Args:
        contour (numpy.ndarray): The contour to be analyzed and for which the shape is determined.
        rect_location_x (int): The X-coordinate of the bounding rectangle's top-left corner relative to the image.
        rect_location_y (int): The Y-coordinate of the bounding rectangle's top-left corner relative to the image.

    Returns:
        shape_dict (dict): A dictionary containing the location (X, Y) of the shape in the image and the number of sides of the detected shape.
    """
    approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)  # approximates number of sides

    # gets location of the shapes in the image and adds to dictionary of shape locations & sides
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        shape_dict = {}
        shape_dict[str(x + rect_location_x) + ", " + str(y + rect_location_y)] = len(approx)
        print(shape_dict)
        return shape_dict


def get_circle_percentages(image):
    """
    Function to detect circles in an image and calculate the percentage completeness of each detected circle.

    Args:
        image (numpy.ndarray): The input image in which circles are to be detected.

    Returns:
        List[float]: A list of percentages representing the completeness of detected circles.
    """
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 0)
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows / 8, 50, param1=50, param2=10, minRadius=5,
                               maxRadius=2000)

    percentages = []

    if circles is not None:
        for circle in circles[0, :]:
            center = int(circle[0]), int(circle[1])
            radius = int(circle[2])

            inlier_count = 0
            total_points = 0

            for angle in np.arange(0, 2 * np.pi, 0.1):
                total_points += 1
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))

                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    if dist[y, x] < (radius / 25.0):
                        inlier_count += 1

            completeness_percentage = (inlier_count / total_points) * 100.0
            percentages.append(completeness_percentage)
            print(f"{completeness_percentage:.2f}% of a circle with radius {radius} detected at ({center[0]}, {center[1]})")

    return percentages


def contour_intersect(original_image, contour1, contour2):
    """
    Function to check if two contours intersect within an original image.
    https://stackoverflow.com/questions/55641425/check-if-two-contours-intersect

    Args:
        original_image (numpy.ndarray): The original image containing the contours.
        contour1 (numpy.ndarray): The first contour to check for intersection.
        contour2 (numpy.ndarray): The second contour to check for intersection.

    Returns:
        bool: True if the contours intersect; otherwise, False.
    """
    contours = [contour1, contour2]

    blank = np.zeros(original_image.shape[0:2])

    image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

    intersection = np.logical_and(image1, image2)

    return intersection.any()


def detect_dominant_color(image, k=3):
    """
    Function to detect the dominant color in an image using k-means clustering.

    Args:
        image (numpy.ndarray): The input image in BGR color space.
        k (int): The number of dominant colors to detect using k-means clustering.

    Returns:
        str: The label of the dominant color in the image.
    """
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
        # if hullx < x:
        #     x = hullx
        # if hully < y:
        #     y = hully
        # if hullw > w:
        #     w = hullw
        # if hullh > h:
        #     h = hullh
        # Extract the region of interest (ROI) from the image
        roi = image[y:y + h, x:x + w]

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


def categorize_shapes(contour):
    """
    Function to categorize shapes based on the number of sides.

    Args:
        contour (numpy.ndarray): The contour to be categorized.

    Returns:
        str: The category of the shape (e.g., 'circle', 'semicircle', 'triangle', 'rectangle', etc.).
    """
    num_sides = len(cv2.approxPolyDP(contour, 0.090 * cv2.arcLength(contour, True), True))

    if num_sides >= 13:
        return 'circle'
    elif num_sides == 7:
        return 'semicircle'
    elif num_sides == 6:
        return 'quarter circle'
    elif num_sides == 3:
        return 'triangle'
    elif num_sides == 4:
        return 'rectangle'
    elif num_sides == 5:
        return 'pentagon'
    elif num_sides == 10:
        return 'star'
    elif num_sides == 12:
        return 'cross'
    else:
        return 'unknown'


def detect_targets(image):
    """
    Function to detect and analyze targets in an image.

    Args:
        image (str): The path to the input image.

    Returns:
        tuple: A tuple containing two images - the masked result image and the image with contours and shapes overlaid.
    """
    color_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)  # reads image, converts to RGB
    color_image_hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)  # converts image to HSV

    # creates & combines masks that only show blue, yellow, orange, and purple
    blue_yellow_orange_mask = create_mask(color_image_hsv, 10, 100, 200, 200, 600, 450)
    purple_mask = create_mask(color_image_hsv, 145, 50, 145, 325, 455, 455)
    masks = blue_yellow_orange_mask | purple_mask

    # creates a result image that excludes everything in the mask and converts it to grayscale
    result_image = cv2.bitwise_and(color_image, color_image, mask=masks)
    grayscale_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # finds contours, sets a minimum and maximum area to detect letters
    contours, hierarchy = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 15
    max_contour_area = 100
    min_rect_contour_area = 100
    max_rect_contour_area = 1000

    # for loop to draw contours for each shape detected
    for cnt in contours:
        # if the contour area is between the minimum and maximum areas, it zooms out from the contour with bounded rectangles and applies thresholding
        if min_contour_area < cv2.contourArea(cnt) < max_contour_area:
            # creates bounded rectangle
            [x, y, w, h] = cv2.boundingRect(cnt)
            x -= 20
            y -= 20

            # crops image to bounded rectangle and creates a grayscale image
            cropped_image = color_image[y:y+h+45, x:x+w+45]
            cropped_grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('images1/croppedDrone.jpg', cropped_image)
            cropped_target = "images1/croppedDrone.jpg"

            print(find_letter(cropped_target))

            # applies threshold algorithm
            threshold_image = cv2.threshold(cropped_grayscale_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # finds contours in cropped image
            contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # creates convex hulls from contours
            hull = []
            for cnt in contours:
                if min_rect_contour_area < cv2.contourArea(cnt) < max_rect_contour_area:
                    for i in range(len(contours)):
                        if not contour_intersect(cropped_image, cnt, contours[i]):
                            hull.append(cv2.convexHull(cnt, False))

            for cnt in hull:
                # finds shape of contours
                shape_category = categorize_shapes(cnt)
                print(f"Detected {shape_category} at ({x}, {y})")

            color_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)
            print(detect_colors_in_hulls(cropped_image, hull))
            color_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

            # draws the convex hulls on the cropped image
            cv2.drawContours(cropped_image, hull, -1, (0, 255, 0), 3)

            # overlays cropped image onto color image
            color_image[y:y+cropped_image.shape[0], x:x+cropped_image.shape[1]] = cropped_image
        # if the contour is not within the minimum and maximum areas, then it just draws the convex hulls
        else:            
            # creates convex hulls from contours
            hull = []
            if cv2.contourArea(cnt) > min_contour_area:
                [x, y, w, h] = cv2.boundingRect(cnt)
                x -= 20
                y -= 20

                # crops image to bounded rectangle and creates a grayscale image
                cropped_image = color_image[y:y+h+25, x:x+w+25]
                cropped_grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('images1/croppedDrone.jpg', cropped_image)
                cropped_target = "images1/croppedDrone.jpg"

                for i in range(len(contours)):
                    if not contour_intersect(cropped_image, cnt, contours[i]):
                        hull.append(cv2.convexHull(cnt, False))


                print(find_letter(cropped_target))
                print("bruh")

                # finds shape of contours
                shape_category = categorize_shapes(cnt)
                print(f"Detected {shape_category} at ({x}, {y})")

                color_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)
                print(detect_colors_in_hulls(color_image, hull))
                color_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

            # draws the convex hulls
            cv2.drawContours(color_image_hsv, hull, -1, (0, 255, 0), 3)

    #color_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)
    #HSV(color_image, color_image_hsv)
    return result_image, color_image



def display_graph(original_image, result_image, final_image):
    """
    Function to display a graph showing images.

    Args:
        result_image (numpy.ndarray): The masked result image.
        final_image (numpy.ndarray): The image with contours and shapes overlaid.

    Returns:
        None: This function displays the graph using Matplotlib.
    """
    titles = ['Original Image', 'Masked', 'Contours']
    images = [cv2.cvtColor(cv2.imread(original_image), cv2.COLOR_BGR2RGB), result_image, final_image]
    for i in range(3):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


from matplotlib import colors
def HSV(image, hsv_image):
    h, s, v = cv2.split(hsv_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    nemo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(nemo)
    #plt.show()


    pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()


def main():
    """
    Main method that initializes and runs everything.
    """
    IMAGE_PATH = "images1/drone1.jpg"  # image file location
    pytesseract.pytesseract.tesseract_cmd = 'tesseract/tesseract.exe'  # your path may be different
    detected_images = detect_targets(IMAGE_PATH)
    display_graph(IMAGE_PATH, detected_images[0], detected_images[1])
    

if __name__ == "__main__":
    main()