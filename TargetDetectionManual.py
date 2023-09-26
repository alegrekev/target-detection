import cv2
import numpy as np
from tkinter import *
from PIL import ImageTk, Image


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
    lower_bound = np.array([lower_hue, lower_saturation, lower_value])
    upper_bound = np.array([upper_hue, upper_saturation, upper_value])
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask


def process_image(image_path):
    """
    Function to process the image, detect targets, and collect target information

    Args:
        image_path (str): Image path.

    Returns:
        tuple: shapes, colors, alphanumerics, alphanumeric_colors
    """
    color_image = cv2.imread(image_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image_HSV = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)

    # Create masks that only show blue, yellow, orange, and purple
    blue_yellow_orange_mask = create_mask(color_image_HSV, 10, 100, 200, 200, 600, 450)
    purple_mask = create_mask(color_image_HSV, 145, 50, 145, 325, 455, 455)

    # Combine the masks
    masks = blue_yellow_orange_mask | purple_mask

    # Create a result image that excludes everything in the mask and converts it to grayscale
    result_image = cv2.bitwise_and(color_image, color_image, mask=masks)
    grayscale_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Find contours and set a minimum and maximum area to detect letters
    contours, _ = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 10
    max_contour_area = 300

    # Lists to store target information
    shapes = []
    colors = []
    alphanumerics = []
    alphanumeric_colors = []

    for cnt in contours:
        # If the contour area is between the minimum and maximum areas, it processes the target
        if min_contour_area < cv2.contourArea(cnt) < max_contour_area:
            [X, Y, W, H] = cv2.boundingRect(cnt)
            X -= 20
            Y -= 20

            # Crop image to bounded rectangle
            cropped_color_image = color_image[Y:Y+H+45, X:X+W+45]
            cropped_color_image = cv2.cvtColor(cropped_color_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('images2/croppedDrone.jpg', cropped_color_image)
            cropped_color_image = "images2/croppedDrone.jpg"

            # Collect target information using GUI
            info = collect_target_info(cropped_color_image)
            shapes.append(info[0].get())
            colors.append(info[1].get())
            alphanumerics.append(info[2].get())
            alphanumeric_colors.append(info[3].get())

    return shapes, colors, alphanumerics, alphanumeric_colors


def collect_target_info(image_path):
    """
    Function to display GUI and collect target information.

    Args:
        image_path (str): Image path.

    Returns:
        tuple: shape_var, color_var, alphanumeric_var, alphanumeric_color_var
    """
    def submit():
        root.destroy()

    root = Tk()
    root.title("Target Information")

    # Variables to store target information
    shape_var = StringVar()
    color_var = StringVar()
    alphanumeric_var = StringVar()
    alphanumeric_color_var = StringVar()

    # Labels and entry fields for target information
    shape_label = Label(root, text="What shape is the target?: ")
    shape_label.pack()
    shape_entry = Entry(root, textvariable=shape_var)
    shape_entry.pack()

    color_label = Label(root, text="What color is the target?: ")
    color_label.pack()
    color_entry = Entry(root, textvariable=color_var)
    color_entry.pack()

    alphanumeric_label = Label(root, text="What alphanumeric is within the target?: ")
    alphanumeric_label.pack()
    alphanumeric_entry = Entry(root, textvariable=alphanumeric_var)
    alphanumeric_entry.pack()

    alphanumeric_color_label = Label(root, text="What color is the alphanumeric within the target?: ")
    alphanumeric_color_label.pack()
    alphanumeric_color_entry = Entry(root, textvariable=alphanumeric_color_var)
    alphanumeric_color_entry.pack()

    # Load and display the cropped image with a larger size
    img = Image.open(image_path)
    img = img.resize((400, 400), Image.LANCZOS)  # Larger size with LANCZOS resampling
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img  # To prevent garbage collection
    panel.pack()

    # Create submit button
    submit_button = Button(root, text="Submit", command=submit)
    submit_button.pack()

    # Run the GUI and handle the case where the user closes the window
    try:
        root.mainloop()
    except:
        pass

    return shape_var, color_var, alphanumeric_var, alphanumeric_color_var


def main():
    """
    Main method that initializes and runs everything.
    """
    image_path = "images2/drone1.jpg"
    shapes, colors, alphanumerics, alphanumeric_colors = process_image(image_path)

    print("Shapes:", shapes)
    print("Colors:", colors)
    print("Alphanumerics:", alphanumerics)
    print("Alphanumeric Colors:", alphanumeric_colors)


if __name__ == "__main__":
    main()