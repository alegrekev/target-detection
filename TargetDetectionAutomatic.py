import cv2
import numpy as np
from matplotlib import pyplot as plt


# -----------------Fields-----------------
IMAGE = "images1/drone1.jpg" # image file location


# -----------------Functions-----------------
def createMask(image, lowerHue, lowerSaturation, lowerValue, upperHue, upperSaturation, upperValue):
    """
    Function to create a mask based on the given HSV values.

    Args:
        image (numpy.ndarray): The input image in HSV color space.
        lowerHue (int): The lower bound for the Hue component (0-179).
        lowerSaturation (int): The lower bound for the Saturation component (0-255).
        lowerValue (int): The lower bound for the Value component (0-255).
        upperHue (int): The upper bound for the Hue component (0-179).
        upperSaturation (int): The upper bound for the Saturation component (0-255).
        upperValue (int): The upper bound for the Value component (0-255).

    Returns:
        mask (numpy.ndarray): The resulting mask with values indicating which pixels in the input image fall within the specified HSV range.
    """
    lowerBound = np.array([lowerHue, lowerSaturation, lowerValue]) # creates array of lower HSV values
    upperBound = np.array([upperHue, upperSaturation, upperValue]) # creates array of upper HSV values
    mask = cv2.inRange(image, lowerBound, upperBound) # creates mask of every color within bounds
    
    return mask


def findShape(contour, rectLocationX, rectLocationY):
    """
    Function to detect shapes of targets.

    Args:
        contour (numpy.ndarray): The contour to be analyzed and for which the shape is determined.
        rectLocationX (int): The X-coordinate of the bounding rectangle's top-left corner relative to the image.
        rectLocationY (int): The Y-coordinate of the bounding rectangle's top-left corner relative to the image.

    Returns:
        shapeDict (dict): A dictionary containing the location (X, Y) of the shape in the image and the number of sides of the detected shape.
    """
    approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True) # approximates number of sides

    # gets location of the shapes in the image and adds to dictionary of shape locations & sides
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

        shapeDict = {}
        shapeDict[str(x + rectLocationX) + ", " + str(y + rectLocationY)] = len(approx)
        print(shapeDict)
        return shapeDict


def get_circle_percentages(image):
    """
    Function to detect circles in an image. 
    https://stackoverflow.com/questions/20698613/detect-semicircle-in-opencv

    Args:
        image (numpy.ndarray): The input image in which circles are to be detected.

    Returns:
        None: This function prints the percentage completeness for each detected circle.
    """
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 0)
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows / 8, 50, param1=50, param2=10, minRadius=5, maxRadius=2000)      
    minInlierDist = 2.0

    for c in circles[0, :]:
        print(c)
        counter = 0
        inlier = 0

        center = (c[0], c[1])
        radius = c[2]

        maxInlierDist = radius/25.0

        if maxInlierDist < minInlierDist: maxInlierDist = minInlierDist

        for i in np.arange(0, 2*np.pi, 0.1):
            counter += 1
            x = center[0] + radius * np.cos(i)
            y = center[1] + radius * np.sin(i)

            if dist.item(int(y), int(x)) < maxInlierDist:
                inlier += 1
            print(str(100.0*inlier/counter) + ' percent of a circle with radius ' + str(radius) + " detected")


def contourIntersect(original_image, contour1, contour2):
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


def detectTargets(image):
    """
    Function to detect and analyze targets in an image.

    Args:
        image (str): The path to the input image.

    Returns:
        tuple: A tuple containing two images - the masked result image and the image with contours and shapes overlaid.
    """
    colorImage = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) # reads image, converts to RGB
    colorImageHSV = cv2.cvtColor(colorImage, cv2.COLOR_RGB2HSV) # converts image to HSV

    # creates & combines masks that only show blue, yellow, orange, and purple
    blueYellowOrangeMask = createMask(colorImageHSV, 10, 100, 200, 200, 600, 450)
    purpleMask = createMask(colorImageHSV, 145, 50, 145, 325, 455, 455)
    masks = blueYellowOrangeMask | purpleMask

    # creates a result image that excludes everything in the mask and converts it to grayscale
    resultImage = cv2.bitwise_and(colorImage, colorImage, mask=masks)
    grayscaleImage = cv2.cvtColor(resultImage, cv2.COLOR_BGR2GRAY)

    # finds contours, sets a minimum and maximum area to detect letters
    contours, hierarchy = cv2.findContours(grayscaleImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minContourArea = 15
    maxContourArea = 100
    minRectContourArea = 100
    maxRectContourArea = 1000

    # for loop to draw contours for each shape detected
    for cnt in contours:
        # if the contour area is between the minimum and maximum areas, it zooms out from the contour with bounded rectangles and applies thresholding
        if cv2.contourArea(cnt) > minContourArea and cv2.contourArea(cnt) < maxContourArea:
            # creates bounded rectangle
            [X, Y, W, H] = cv2.boundingRect(cnt)
            X -= 20
            Y -= 20

            # crops image to bounded rectangle and creates a grayscale image
            croppedImage = colorImage[Y:Y+H+45, X:X+W+45]
            croppedGrayscaleImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)

            # applies threshold algorithm
            thresholdImage = cv2.threshold(croppedGrayscaleImage, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # finds contours in cropped image
            contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #get_circle_percentages(thresholdImage)

            # creates convex hulls from contours
            hull = []
            for cnt in contours:
                if cv2.contourArea(cnt) > minRectContourArea and cv2.contourArea(cnt) < maxRectContourArea:
                    for i in range(len(contours)):
                        if contourIntersect(croppedImage, cnt, contours[i]) == False:
                            hull.append(cv2.convexHull(cnt, False))
                    
                            for cnt in hull:
                                # finds shape of contours
                                findShape(cnt, X, Y)

            # draws the convex hulls on the cropped image
            cv2.drawContours(croppedImage, hull, -1, (0, 255, 0), 3)

            # overlays cropped image onto color image
            colorImage[Y:Y+croppedImage.shape[0], X:X+croppedImage.shape[1]] = croppedImage
        # if the contour is not within the minimum and maximum areas, then it just draws the convex hulls
        else:
            # creates convex hulls from contours
            hull = []
            if cv2.contourArea(cnt) > minContourArea:
                for i in range(len(contours)):
                    if contourIntersect(croppedImage, cnt, contours[i]) == False:
                        hull.append(cv2.convexHull(cnt, False))

                        for cnt in hull:
                            # finds shape of contours
                            findShape(cnt, 0, 0)

            # draws the convex hulls
            cv2.drawContours(colorImage, hull, -1, (0, 255, 0), 3)
    return resultImage, colorImage


def displayGraph(originalImage, resultImage, finalImage):
    """
    Function to display a graph showing images.

    Args:
        resultImage (numpy.ndarray): The masked result image.
        finalImage (numpy.ndarray): The image with contours and shapes overlaid.

    Returns:
        None: This function displays the graph using Matplotlib.
    """
    titles = ['Original Image', 'Masked', 'Contours']
    images = [cv2.cvtColor(cv2.imread(originalImage), cv2.COLOR_BGR2RGB), resultImage, finalImage]
    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


# -----------------Run Functions-----------------
detected_images = detectTargets(IMAGE)
displayGraph(IMAGE, detected_images[0], detected_images[1])