import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optflow import dicom_optflow
from intense_sum import dicom_intense
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import sympy as sp
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from interpolation import interp
from data_analysis import analysis
from sklearn.metrics import mean_squared_error
from math import sqrt


#엑셀파일로 경로를 파악하기 위한 클래스 (+벡터수 저장/확인용 함수)
class excel:
    df = pd.read_excel('ESCARGOT/ESCARGOT.xlsx')

    def __init__(self):
        self.drop_df = self.df.drop([0,2,4,9,11,12,15,22,27,29,34,36,38,41,43,49], axis=0, inplace=False)
        #self.drop_df = self.df.drop([0,2,4,9,11,12,22,34,38,41,43,49], axis=0, inplace=False)
        self.df.drop(['비고'], axis=1, inplace=True) #불필요한 열 제거
        self.df.dropna(axis=1); self.df.fillna(0, inplace=True) #혹시나 불필요한 열 더 제거
        self.idx = self.df.index
        self.didx = self.drop_df.index #드랍하고 남은 인덱스행 파악

    def load_excel(self, pat_num=-1): #엑셀로부터 파일의 경로를 추출해온다.
        paths = []
        if pat_num == -1: #모든 경로를 추출하고 싶을 때
            for i in self.didx:
                num = self.drop_df['Img_bas_n'][i].astype(int)
                paths.append('ESCARGOT/' + f'{i+1}/{i+1}({num}).dcm')
        else: #특정 경로만 추출하고 싶을 때
            num = self.df['Img_bas_n'][self.idx[pat_num]].astype(int)
            paths = 'ESCARGOT/' + f'{pat_num+1}/{pat_num+1}({num}).dcm'

        return paths
    #####################################################

    def set_all_plot(self, save=False, preprocess=False, interpolate=False, cycle=1, opt=False):
        paths = self.load_excel()
        plen = len(paths)

        tfcs_pred=[]; tfce_pred=[]; ccfcs_pred=[]; ccfce_pred=[]
        rtfcs=[]; rtfce=[]; rccfcs=[]; rccfce=[]
        for i in range(plen):
            dicom_data = pydicom.dcmread(paths[i]) #force=True 인자를 넣으면 인식안되는 dicom 읽을수 있음
            dicom_img = dicom_data.pixel_array #dicom이미지 호출 (dicom_img[n] = n번째 프레임)

            k = self.didx[i]
            dicom_img = np.load('DENOISED/'+str(k+1)+'(b).npy')

            all_frame = dicom_img.shape[0] #dicom_img.shape = [389, 512, 512] -> 프레임 389장, 512x512

            ##옵티컬플로우 or 밝기값
            if opt == True: mat_i, mat_x = dicom_optflow().do_optflow(dicom_img, all_frame, preprocess)
            else: mat_i, mat_x = dicom_intense().do_intense(dicom_img, all_frame, preprocess)

            #dicom_img = dicom_intense().proc_data(dicom_img)

            ##보간사용/미사용
            plt.plot(mat_i, mat_x, color='black')
            colors = ['lime', 'yellow', 'magenta', 'gold', 'grey']

            if interpolate == False: plt.plot(mat_i, mat_x, color='lime')
            else:
                for i in range(cycle):
                    interp_i, mat_x1 = interp().peak_interpolation(mat_i, mat_x) #1차 보간
                    TFCS, TFCE, CCFCS, CCFCE = analysis().get_TFC_CCFC(np.array(interp_i), np.array(mat_x1), isgmm=True, draw=True) #클러스터링(n=2)기반 극대극소를 발견하는 TFC

            ##############################결과출력
            rtfc_s, rtfc_e = self.df['TFC_start_b'][k].astype(int), self.df['TFC_end_b'][k].astype(int)
            rccfc_s, rccfc_e = self.df['CCFC_start_b'][k].astype(int), self.df['CCFC_end_b'][k].astype(int)

            tfcs_pred.append(TFCS); tfce_pred.append(TFCE); ccfcs_pred.append(CCFCS); ccfce_pred.append(CCFCE)
            rtfcs.append(rtfc_s); rtfce.append(rtfc_e); rccfcs.append(rccfc_s); rccfce.append(rccfc_e)

            #########################

            print(f'[{k+1}] TFC: [{rtfc_s}, {rtfc_e}], predict: [{TFCS}, {TFCE}]')
            print(f'CCFC: [{rccfc_s}, {rccfc_e}], predict: [{CCFCS}, {CCFCE}]')
            plt.title(f'[{int(k+1)}] TFC: [{rtfcs}, {rtfce}], predict: [{TFCS}, {TFCE}]\n CCFC: [{rccfcs}, {rccfce}], predict: [{CCFCS}, {CCFCE}]', fontsize=15)
            #####################################

            ##엑셀속 frame count 표시
            plt.axvspan(self.drop_df['TFC_start_b'][k].astype(int), self.drop_df['TFC_end_b'][k].astype(int), facecolor='red')
            plt.axvspan(self.drop_df['CCFC_start_b'][k].astype(int), self.drop_df['CCFC_end_b'][k].astype(int), facecolor='blue')
            plt.title(f'[{int(k+1)}] data: [{rtfcs}, {rtfce}], predict: [{TFCS}, {TFCE}]', fontsize=20)
            if save == True: plt.savefig('all_plot/'+f'{k+1}.png')
            plt.clf()
            
        ###오차 평가
        tfcs_rms = sqrt(mean_squared_error(rtfcs, tfcs_pred))
        tfce_rms = sqrt(mean_squared_error(rtfce, tfce_pred))
        print(f'tfcs score={tfcs_rms}, tfce score={tfce_rms}')

        ccfcs_rms = sqrt(mean_squared_error(rccfcs, ccfcs_pred))
        ccfce_rms = sqrt(mean_squared_error(rccfce, ccfce_pred))
        print(f'ccfcs score={ccfcs_rms}, ccfce score={ccfce_rms}')
        #################
    ####################################################



    def show_plot(self, pat_num, preprocess=False, interpolate=False, cycle=1, opt=False):  #특정파일에 대한 하강벡터수 확인
        pat_num-=1
        path = self.load_excel(pat_num)
        dicom_data = pydicom.dcmread(path) #force=True 인자를 넣으면 인식안되는 dicom 읽을수 있음
        dicom_img = dicom_data.pixel_array #dicom이미지 호출 (dicom_img[n] = n번째 프레임)
        
        dicom_img = np.load('DENOISED/'+str(pat_num+1)+'(b).npy')
        all_frame = dicom_img.shape[0] #dicom_img.shape = [389, 512, 512] -> 프레임 389장, 512x512
        ##옵티컬플로우 or 밝기값
        if opt == True: mat_i, mat_x = dicom_optflow().do_optflow(dicom_img, all_frame, preprocess)
        else: mat_i, mat_x = dicom_intense().do_intense(dicom_img, all_frame, preprocess)
        
        ##보간사용/미사용
        k = pat_num; interp_i=[]
        plt.plot(mat_i, mat_x, color='black')
        if interpolate == False: plt.plot(mat_i, mat_x, color='black')
        else:
            for i in range(cycle):
                interp_i, mat_x1 = interp().peak_interpolation(mat_i, mat_x) #1차 보간
                TFCS, TFCE, CCFCS, CCFCE = analysis().get_TFC_CCFC(np.array(interp_i), np.array(mat_x1), isgmm=True, draw=True) #클러스터링(n=2)기반 극대극소를 발견하는 TFC

        rtfcs, rtfce = self.df['TFC_start_b'][k].astype(int), self.df['TFC_end_b'][k].astype(int)
        rccfcs, rccfce = self.df['CCFC_start_b'][k].astype(int), self.df['CCFC_end_b'][k].astype(int)

        print(f'TFC: [{rtfcs}, {rtfce}], predict: [{TFCS}, {TFCE}]')
        print(f'CCFC: [{rccfcs}, {rccfce}], predict: [{CCFCS}, {CCFCE}]')
        plt.title(f'[{int(k+1)}] TFC: [{rtfcs}, {rtfce}], predict: [{TFCS}, {TFCE}]\n CCFC: [{rccfcs}, {rccfce}], predict: [{CCFCS}, {CCFCE}]', fontsize=15)

        ##엑셀속 frame count 표시
        plt.axvspan(self.df['TFC_start_b'][k].astype(int), self.df['TFC_end_b'][k].astype(int), facecolor='red')
        plt.axvspan(self.df['CCFC_start_b'][k].astype(int), self.df['CCFC_end_b'][k].astype(int), facecolor='blue')
        plt.show()
    #######################################################

