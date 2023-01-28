import cv2
import numpy as np
from matplotlib import pyplot as plt

# states what image to use, creates 2 copies (one to keep and one to edit)
image = "images/drone1.jpg"
originalImage = cv2.imread(image)
originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
colorImage = cv2.imread(image)
colorImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)

# converts image from RGB to HSV
colorImageHSV = cv2.cvtColor(colorImage, cv2.COLOR_RGB2HSV)

#function to create a mask based on the given HSV values
def createMask(lowerHue, lowerSaturation, lowerValue, upperHue, upperSaturation, upperValue):
    lowerBound = np.array([lowerHue, lowerSaturation, lowerValue]) # 10 100 210, 145 40 145
    upperBound = np.array([upperHue, upperSaturation, upperValue]) # 200 600 450, 325 455 455
    mask = cv2.inRange(colorImageHSV, lowerBound, upperBound)

    return mask

# creates masks that only show blue, yellow, orange, and purple
blueYellowOrangeMask = createMask(10, 100, 190, 200, 600, 450)
purpleMask = createMask(145, 50, 145, 325, 455, 455)

# combines the masks
masks = blueYellowOrangeMask | purpleMask

# creates a result image that excludes everything in the mask and converts it to grayscale
resultImage = cv2.bitwise_and(colorImage, colorImage, mask=masks)
grayscaleImage = cv2.cvtColor(resultImage, cv2.COLOR_BGR2GRAY)

# finds contours and sets a minimum area
contours, hierarchy = cv2.findContours(grayscaleImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
minContourArea = 40

# zooms out from contours with bounded rectangles and applies thresholding
for cnt in contours:
    if cv2.contourArea(cnt) > minContourArea:
        # creates bounded rectangles
        [X, Y, W, H] = cv2.boundingRect(cnt)
        X -= 20
        Y -= 20

        # crops images to bounded rectangles and creates a grayscale image
        croppedColorImage = colorImage[Y:Y+H+45, X:X+W+45]
        cv2.imwrite('images/croppedDrone.jpg', croppedColorImage)
        croppedColorImage = "images/croppedDrone.jpg"
        croppedImage = cv2.imread(croppedColorImage)
        croppedGrayscaleImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)

        # detect if it a letter, and if it is, continue

        # applies threshold algorithm
        thresholdImage = cv2.threshold(croppedGrayscaleImage, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # finds contours in cropped image
        contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # finds contours and their convex hulls from threshold image
        hull = []
        for cnt in contours:
            if cv2.contourArea(cnt) > minContourArea:
                hull.append(cv2.convexHull(cnt, False))

        # draws the convex hulls for each image
        cv2.drawContours(croppedImage, hull, -1, (0, 255, 0), 3)

        # overlays cropped image onto color image
        colorImage[Y:Y+croppedImage.shape[0], X:X+croppedImage.shape[1]] = croppedImage

# plots all images
titles = ['Original Image', 'Result Image', 'Contours']
images = [originalImage, resultImage, colorImage]
for i in range(3):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
