import pytesseract

def find_letter(image):
    return pytesseract.image_to_string(image, lang = 'eng', config='--psm 7')