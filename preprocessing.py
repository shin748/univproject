import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import multiprocessing
import os

class do_preprocess:
    ##############       전처리 함수     ########################
    def extract_bv(self, image, preprocess=True):
        if preprocess == False: return image

        # blurring image(median method)
        blured_image = cv2.medianBlur(image, 7)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced_fundus = clahe.apply(blured_image)

        # opening and closing operation
        r1 = cv2.morphologyEx(contrast_enhanced_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
        R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
        r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
        R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
        r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
        R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
        f4 = cv2.subtract(R3,contrast_enhanced_fundus)
        f5 = clahe.apply(f4)

        # contour thresholding
        ret,f6 = cv2.threshold(f5, 70, 255, cv2.THRESH_BINARY)

        return f6

    def denoise(self, image, i, idx, window_size):
        image = cv2.fastNlMeansDenoisingMulti(self.image[i:i+window_size], idx, window_size, None, 10, 7, 35)
        return image

    def proc_data(self, img, preprocess):
        process_count = os.cpu_count()

        idx = 1
        window_size = 3

        #print("denoising start")
        with multiprocessing.Pool(process_count*2) as p:
            p1 = p.starmap_async(self.denoise, ([i, idx, window_size] for i in range(len(img)-window_size)))
            img = p1.get()
            for i in range(int((window_size-1)/2)):
                img.insert(0, img[0])
                img.append(img[-1])
            if window_size%2 == 1:
                img.append(img[-1])
        #print('denoising end')

        #print('extracting blood vessel start')
        with multiprocessing.Pool(process_count*2) as p:
            p2 = p.map_async(self.extract_bv, (img, preprocess))
            img = p2.get()
        #print('extracting blood vessel end')
        return img