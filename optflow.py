import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#옵티컬 플로우 클래스
class dicom_optflow:
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
    def do_optflow(self, dicom_img, all_frame, preprocess):
        ################################
        h = dicom_img.shape[1]-50; w = dicom_img.shape[2]-50
        prev = None                             #이전 프레임 저장 변수
        delay=int(1000/30); i=0                 #재생관련 변수
        mat_i=[0]; mat_x=[0]; mat_y=[]          #분포 측정관련 변수 (0을 넣는 이유는 첫 프레임은 값이 없어 저장되지 않기 때문)

        ################################
        while True:
            frame = self.extract_bv(dicom_img[i][0:dicom_img.shape[1], 0:dicom_img.shape[2]-50], preprocess) #True=전처리, False=원본
            flow_frame = frame.copy() #옵티컬 플로우 시각화를 위한 점과 벡터
            h = frame.shape[0]; w = frame.shape[1]
            max = 0
            if prev is None: prev = frame
            else:
                flow = cv2.calcOpticalFlowFarneback(prev,frame,None,0.5,3, 10, 3,5,1.5,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                ######################################옵티컬 플로우 시각화 파트
                cnt_vec=0; step=16 #step수에 따라 파악하는 (x,y)개수가 달라짐
                idx_y,idx_x = np.mgrid[step/2:h:step , step/2:w:step].astype(np.int)
                grid = np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2) #그리드 점 찍기

                for x,y in grid:
                    cv2.circle(flow_frame, (x,y), 1, (255,255,0), 1) #점 찍기
                    dx,dy = flow[y, x].astype(np.int)
                    ################################################
                    '''
                    dx=0이면 y축으로만 움직이니 수직벡터, dy=0이면 x축으로만 움직이니 수평벡터
                    dx>0이면 우측, dy>0이면 아래로 뻗음.

                    수평벡터는 무시하는게 좋아보임.

                    하향(위에서 아래)벡터: dy>0 -> (dx==0, dy>0) or (dx>0, dy>0) or (dx<0, dy>0)
                    상향(아래에서 위)벡터: dy<0 -> (dx==0, dy<0) or (dx>0, dy<0) or (dx<0, dy<0)
                    '''
                    ###############################################
                    if dx>=0 and dy>3: #(x,y) -> (x+dx, y+dy)
                        #if(same_vec[y][x]>0): continue
                        #same_vec[y][x]+=1
                        cnt_vec+=1 #벡터 카운팅
                        cv2.line(flow_frame, (x,y), (x+dx, y+dy), (255,255, 0),2)

                mat_i.append(i); mat_x.append(cnt_vec) #i=프레임, x=벡터수

                prev = frame
            ####################### else 종료 #################################

            ###############################동영상 재생처리 부분
            cv2.imshow('make it', flow_frame)

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