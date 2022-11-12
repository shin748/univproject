import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#밝기값 클래스
class dicom_intense:
    ##############       전처리 함수     ########################
    def extract_bv(self, image, preprocess):
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

   #################       옵티컬플로우 함수   #####################
    def do_intense(self, dicom_img, all_frame, preprocess):
        ################################
        h = dicom_img.shape[1]-50; w = dicom_img.shape[2]-50
        prev = None                             #이전 프레임 저장 변수
        delay=int(1000/30); i=0                 #재생관련 변수
        mat_i=[]; mat_x=[]; mat_y=[]            #분포 측정관련 변수
        print(h,w)
        ################################
        while True:
            frame = self.extract_bv(dicom_img[i][0:dicom_img.shape[1], 0:dicom_img.shape[2]-50], preprocess) #True=전처리, False=원본

            intense_sum = frame.sum()

            mat_i.append(i)
            mat_x.append(intense_sum)

            ###############################동영상 재생처리 부분
            cv2.imshow('make it', frame)

            i+=1
            if i >= all_frame: break

            key = cv2.waitKey(delay)
            if key == 27: break #ESC(27)누르면 종료
            elif key == 8: i=0 #백스페이스 누르면 벡터 제거(sparse O.F 전용)
            elif key == 32: i-=1 #스페이스바 누르면 정지 (벡터 사라짐 주의)
            #################################################
        #print(f'프레임: {mat_i}\n 벡터수: {mat_x}')
        cv2.destroyAllWindows()
        return mat_i, mat_x
    
    def peak_interpolation(self, mat_i, mat_x):
        prev_x=mat_x[0]
        incr=1
        peak_i=[]; peak_x=[]
        ###########  피크탐색  ###################
        for i in mat_i:
            i-=1         
            if prev_x >= mat_x[i]: #감소중
                if incr==1:
                    peak_i.append(i+1); peak_x.append(prev_x) #증가하다 감소시작 = 피크위치만 저장
                    incr=0 #증가 종료를 알림
                prev_x = mat_x[i]

            elif prev_x < mat_x[i]:
                incr=1 #증가하면 증가시작을 알림
                prev_x = mat_x[i]
        
        ###########  보간진행  ###################
        f = interpolate.interp1d(peak_i, peak_x, kind='linear')
        if mat_i[-1] > peak_i[-1]: mat_i = mat_i[0:peak_i[-1]]
        f_y = f(mat_i)
        return plt.plot(mat_i, f_y, '-', color='lime')




###############################################################################################################################################