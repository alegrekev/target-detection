import cv2 as cv
import pytesseract as pt
from PIL import Image

def get_alphanumeric(image_path, targ_coords):
    # Read the image
    x, y, w, h = targ_coords
    image = cv.imread(image_path)
    image = image[y:y+h, x:x+w]

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    char = pt.image_to_string(image, config="-psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if len(char) != 1:
        print("pytesseract.image_to_string returned string with length %d\n", len(char))
        return '~'
    else:
        return char



    
