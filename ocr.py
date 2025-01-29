
from PIL import Image
from easyocr import Reader

image_path = r'Images\sample.jpg'

# image = Image.open(image_path)  # This is for viwing the image if needed
# image.show()

reader = Reader(['en'], gpu = False) 
#This a reader that uses deep learning to extract text from images - it takes some time to load
#'en' indicates english. multiple languages can be selected for reading text.

result = reader.readtext(image_path)
#the result is an array - each row contains 4 things
# 1. a list of 4 points indicating the position co ordinates of the text
# 2. The text 
# 3. Confidence score - simply means accuracy
 
for detection in result:
    print(detection[1])  #here we are printing the text from the result object





