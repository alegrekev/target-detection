import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


# states what image to use, creates 2 copies (one to keep and one to edit)
image = "images/drone1.jpg"
originalImage = cv2.imread(image)
originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
colorImage = cv2.imread(image)
colorImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)

# sets HSV boundries and erases colors within boundries from image
colorImageHSV = cv2.cvtColor(colorImage, cv2.COLOR_BGR2HSV)
lower_green = np.array([59, 57, 36])
upper_green = np.array([125, 62, 63])















"""
# converts image to grayscale, blurs it, and applies thresholding
grayscaleImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
blurredImage1 = cv2.GaussianBlur(grayscaleImage,(7,7), 7)
blurredImage2 = cv2.GaussianBlur(grayscaleImage,(7,7), 7)
blurredImageFinal = blurredImage1 + blurredImage2
#thresholdImage = cv2.threshold(blurredImageFinal, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
thresholdImage = cv2.Canny(blurredImageFinal,100,200)

# finds contours and calculates mean area of contours
areas = []
contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    areas.append(area)
meanArea = sum(areas) / len(areas)

# draws contours that are close to the mean area in image
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 0.25 * meanArea and area < 2 * meanArea:
        cv2.drawContours(colorImage, cnt, -1, (0, 255, 0), 3)

# plots all images
titles = ['Original Image', 'Grayscale Image','Blurred','Threshold', 'Contours']
images = [originalImage, grayscaleImage, blurredImageFinal, thresholdImage, colorImage]
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
"""

#blurredImage = cv2.bilateralFilter(grayscaleImage,9,75,75)
#cannyImage = cv2.Canny(blurredImageFinal,100,200)
# thresholdImage = cv2.adaptiveThreshold(blurredImage2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # applies adaptive gaussian thresholding
# blur = cv2.GaussianBlur(grayscaleImage,(5,5), 0) # maybe should use this
# _, thresholdImage = cv2.threshold(grayscaleImage, 67, 255, cv2.THRESH_TOZERO) # applies thresholding
# thresholdImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# _, thresholdImage = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# cut in 4 parts and 8 parts to make sure that cut does not overlap the target

# do a threshold that gets rid of black, white, brown, and green



#from PIL import Image, ImageEnhance
#import os


# adjusts contrast and saturation of image to make targets stand out more
#imageToEnhance = Image.open(image)
#enhancedImage1.save('enhancedImage1.png')
#enhancedImageContrast = cv2.imread("enhancedImage1.png")
#saturation = ImageEnhance.Color(imageToEnhance)
#enhancedImageSaturation = saturation.enhance(3)
#contrast = ImageEnhance.Contrast(enhancedImageSaturation)
#enhancedImageContrast = contrast.enhance(0.4)
#enhancedImageContrast.save('alteredImage.png')
#enhancedImageSaturation = cv2.imread("enhancedImage2.png")
#os.remove("enhancedImage1.png")
#os.remove("enhancedImage2.png")
#enhancedImageFinal = cv2.imread('alteredImage.png')
#plt.imshow(enhancedImageFinal)
#plt.show()







""" 
obtain image
edit contast/saturation of image (if needed)
get rid of certain colors with HSV (find best colors to exclude)
apply grayscale, blurring, thresholding (determine best algorithms)
find contours and display the necessary contours on screen (find best range)
display all images to show process
"""









#pixel_colors = colorImage.reshape((np.shape(colorImage)[0]*np.shape(colorImage)[1], 3))
#norm = colors.Normalize(vmin=-1.,vmax=1.)
#norm.autoscale(pixel_colors)
#pixel_colors = norm(pixel_colors).tolist()

#h, s, v = cv2.split(colorImageHSV)
#fig = plt.figure()
#axis = fig.add_subplot(1, 1, 1, projection="3d")

#axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
#axis.set_xlabel("Hue")
#axis.set_ylabel("Saturation")
#axis.set_zlabel("Value")
#plt.show()