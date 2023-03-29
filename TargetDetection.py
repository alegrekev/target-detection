import cv2
import numpy as np
from tkinter import *
from PIL import ImageTk, Image

"""Code for target detection & image cropping"""
# states what image to use, creates 2 copies (one to keep and one to edit)
image = "images/drone1.jpg"
color_image = cv2.imread(image)
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# converts image from RGB to HSV
color_image_HSV = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)

# function to create a mask based on the given HSV values
def createMask(lower_hue, lower_saturation, lower_value, upper_hue, upper_saturation, upper_value):
    lower_bound = np.array([lower_hue, lower_saturation, lower_value])
    upper_bound = np.array([upper_hue, upper_saturation, upper_value])
    mask = cv2.inRange(color_image_HSV, lower_bound, upper_bound)

    return mask

# creates masks that only show blue, yellow, orange, and purple
blue_yellow_orange_mask = createMask(10, 100, 200, 200, 600, 450)
purple_mask = createMask(145, 50, 145, 325, 455, 455)

# combines the masks
masks = blue_yellow_orange_mask | purple_mask

# creates a result image that excludes everything in the mask and converts it to grayscale
result_image = cv2.bitwise_and(color_image, color_image, mask=masks)
grayscale_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

# finds contours, sets a minimum and maximum area to detect letters
contours, hierarchy = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = 10
max_contour_area = 300

for cnt in contours:
    # if the contour area is between the minimum and maximum areas, it zooms out from the contour with bounded rectangles and applies thresholding
    if cv2.contourArea(cnt) > min_contour_area and cv2.contourArea(cnt) < max_contour_area:       
        # creates bounded rectangle
        [X, Y, W, H] = cv2.boundingRect(cnt)
        X -= 20
        Y -= 20

        # crops image to bounded rectangle
        cropped_color_image = color_image[Y:Y+H+45, X:X+W+45]
        cropped_color_image = cv2.cvtColor(cropped_color_image, cv2.COLOR_BGR2RGB)
        cropped_color_image = cv2.resize(cropped_color_image, (300, 300))
        cv2.imwrite('images/croppedDrone.jpg', cropped_color_image)
        cropped_color_image = "images/croppedDrone.jpg"

        """Code for input GUI"""
        # creates root
        root = Tk()
        root.title("Target Information")

        # creates labels/entry for shape
        shape_label = Label(root, text="What shape is the target?: ")
        shape_label.pack()
        shape_entry = Entry(root)
        shape_entry.pack()

        # creates labels/entry for color
        color_label = Label(root, text="What color is the target?: ")
        color_label.pack()
        color_entry = Entry(root)
        color_entry.pack()

        # creates labels/entry for alphanumeric
        alphanumeric_label = Label(root, text="What alphanumeric is within the target?: ")
        alphanumeric_label.pack()
        alphanumeric_entry = Entry(root)
        alphanumeric_entry.pack()

        # creates labels/entry for alphanumeric color
        alphanumeric_color_label = Label(root, text="What color is the alphanumeric within the target?: ")
        alphanumeric_color_label.pack()
        alphanumeric_color_entry = Entry(root)
        alphanumeric_color_entry.pack()

        # defines what happens when user clicks submit button
        def submit():
            shape = shape_entry.get()
            color = color_entry.get()
            alphanumeric = alphanumeric_entry.get()
            alphanumeric_color = alphanumeric_color_entry.get()
            print("Shape:", shape)
            print("Color:", color)
            print("Alphanumeric:", alphanumeric)
            print("Alphanumeric Color:", alphanumeric_color)
            root.destroy()

        # creates submit button
        submit_button = Button(root, text="Submit", command=submit)
        submit_button.pack()

        # adds bounded rectangle image to GUI
        img = ImageTk.PhotoImage(Image.open(cropped_color_image))
        panel = Label(root, image = img)
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        
        # runs the GUI
        root.mainloop()