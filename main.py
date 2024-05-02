import target_detection as targ
#import shape_detection as shape
import color_detection as color
import letter_detection as lett

def main():
    """
    Main method that initializes and runs everything.
    """
    # --------------------- Detect Target Locations ---------------------

    IMAGE_PATH = "input_images/drone1.jpg"  # image file location
    detected_images = targ.detect_targets(IMAGE_PATH)
    print(detected_images[0])
    targ.display_graph(detected_images[1])

    # --------------------- Detect Target Shapes, Colors, Alphanumerics ---------------------

    for targ_coords in detected_images[0]:
        print("~~~~~ Next Target ~~~~~")

        # Shape Detection
        pass

        # Color Detection
        detected_color = color.recognize_dominant_color(IMAGE_PATH, targ_coords)
        print("tuple from kmeans: ", detected_color)
        targ_color = color.get_color_name(detected_color)

        # Alphanumeric Detection
        #alphanumeric = lett.get_alphanumeric(IMAGE_PATH, targ_coords)

        print("target location: ", targ_coords)
        #print("target shape: ", targ_shape)
        print("color name: ", targ_color)
        #print("text: ", alphanumeric, "\n\n")




if __name__ == "__main__":
    main()
