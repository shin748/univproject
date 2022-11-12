import pydicom
import cv2
import numpy as np
import math
import string
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#손이 있음: 1
#자료자체에 문제있음: 3, 5, 12, 23, 39, 44
#자료에 문제는 없으나 디테일한 문제가 있음(흐리거나, 이상한게 있거나): 3, 49
#이외엔 문제없음? 37(26) 정체불명의 혈관이 후반에 발생
path = 'C:/Users/USERH/Desktop/ESCARGOT/8/8(24).dcm'

dicom_data = pydicom.dcmread(path) #force=True 인자를 넣으면 인식안되는 dicom 읽을수 있음
dicom_img = dicom_data.pixel_array #dicom이미지 호출 (dicom_img[n] = n번째 프레임)
#dicom_img.shape = [389, 512, 512] -> 프레임 389장, 512x512

#########################영상 변수들
delay=int(1000/30); i=0
i=0
all_frame = dicom_img.shape[0]
#print(all_frame)
####################################

#########################특수 변수들
struct_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
struct_e2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mat_x=[]; mat_y=[]; mat_i=[]
####################################

def inversion(frame):
    mask = np.full_like(frame,255)
    return cv2.bitwise_xor(frame, mask)

    
########################################################################################
while True:
    frame = dicom_img[i][0:512, 0:512]
    h = frame.shape[0]; w = frame.shape[1]

    
    frame = cv2.adaptiveThreshold(frame, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 243, 15)
    #frame = cv2.GaussianBlur(frame,(7,7),0)
    frame = cv2.medianBlur(frame,7)
    #frame = cv2.blur(frame,(7,7))
    #frame = inversion(frame)

    #frame = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, struct_e)
    #frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, struct_e2)
    #frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, struct_e)
    cv2.imshow('ESC TO OFF', frame)

    ########동영상 재생처리 부분
    i+=1
    if i >= all_frame: break

    key = cv2.waitKey(delay)
    if key == 27: break #ESC(27)누르면 종료
    elif key == 8: i=0 #백스페이스 누르면 벡터 제거(sparse O.F 전용)
    elif key == 32: i-=1 #스페이스바 누르면 정지 (벡터 사라짐 주의)
    #############################
#plt.figure().gca(projection='3d').scatter(mat_x,mat_y,mat_i)
#plt.scatter(mat_x, mat_y)
#plt.show()
cv2.destroyAllWindows()