import cv2
from matplotlib import pyplot as plt

# states what image to use, creates 2 copies (one to keep and one to edit)
image = "images/drone1.jpg"
originalImage = cv2.imread(image)
originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
colorImage = cv2.imread(image)
colorImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)

# converts image to grayscale, blurs it, and applies thresholding
grayscaleImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
blurredImage1 = cv2.GaussianBlur(grayscaleImage,(7,7), 7)
blurredImage2 = cv2.GaussianBlur(grayscaleImage,(7,7), 7)
blurredImageFinal = blurredImage1 + blurredImage2
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
