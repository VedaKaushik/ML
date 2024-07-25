import cv2
import glob        
import numpy as np
import matplotlib.pyplot as plt

input_path = r"C:/Users/vedak/Downloads/Data_set/dataset_blur_512/*.png"

# make sure below folder already exists
out_path = 'C:/Users/vedak/Downloads/Data_set/dataset_new/'

image_paths = list(glob.glob(input_path))
for i, img in enumerate(image_paths):
    image = cv2.imread(img)
    image = cv2.resize(image, (512, 256))
    #The initial processing of the image
    image = cv2.GaussianBlur(image,(3,3),0)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(image_bw,140,255,cv2.THRESH_BINARY)

    #The declaration of CLAHE 
    #clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_img = clahe.apply(thresh1) 
    plt.imshow(final_img)
    # cv2.imwrite(out_path + f'{str(i)}.png', final_img)