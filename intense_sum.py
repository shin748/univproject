import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from preprocessing import do_preprocess

#밝기값 클래스
class dicom_intense:
   #################       옵티컬플로우 함수   #####################
    def do_intense(self, dicom_img, all_frame, preprocess):
        ################################
        h = dicom_img.shape[1]; w = dicom_img.shape[2]
        prev = None                             #이전 프레임 저장 변수
        delay=int(60/30); i=0                 #재생관련 변수
        mat_i=[]; mat_x=[]; mat_y=[]            #분포 측정관련 변수
        #print(dicom_img.shape)
        ################################
        while True:
            #frame = do_preprocess().proc_data(dicom_img[i][int(h/2):h, 0:w-50], preprocess) #True=전처리, False=원본
            frame = do_preprocess().extract_bv(dicom_img[i][:h, :w-50], preprocess) #True=전처리, False=원본

            intense_sum = frame.sum() #i번째 프레임의 intense 개수

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
###############################################################################################################################################