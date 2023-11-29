import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

import ShapeDetection as sd
import LetterDetection as ld
import ColorDetection as cd


# use kmeans to get dominant colar to use for variable masks

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
            cv2.imwrite('target-detection/images1/croppedDrone.jpg', cropped_image)
            cropped_target = "target-detection/images1/croppedDrone.jpg"

            print(ld.find_letter(cropped_target))

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
                shape_category = sd.categorize_shapes(cnt)
                print(f"Detected {shape_category} at ({x}, {y})")

            color_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)
            print(cd.detect_colors_in_hulls(cropped_image, hull))
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
                [x, y, w, h] = cv2.boundingRect(cnt) #x,y start at top left
                x -= 20
                y -= 20

                # crops image to bounded rectangle and creates a grayscale image
                cropped_image = color_image[y:y+h+25, x:x+w+25]
                cropped_grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('target-detection/images1/croppedDrone.jpg', cropped_image)
                cropped_target = "target-detection/images1/croppedDrone.jpg"

                for i in range(len(contours)):
                    if not contour_intersect(cropped_image, cnt, contours[i]):
                        hull.append(cv2.convexHull(cnt, False))


                print(ld.find_letter(cropped_target))
                print("bruh")

                # finds shape of contours
                shape_category = sd.categorize_shapes(cnt)
                print(f"Detected {shape_category} at ({x}, {y})")

                color_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)
                print(cd.detect_colors_in_hulls(color_image, hull))
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
    IMAGE_PATH = "target-detection/images1/drone1.jpg"  # image file location
    pytesseract.pytesseract.tesseract_cmd = 'target-detection/tesseract/tesseract.exe'  # your path may be different
    detected_images = detect_targets(IMAGE_PATH)
    display_graph(IMAGE_PATH, detected_images[0], detected_images[1])
    

if __name__ == "__main__":
    main()