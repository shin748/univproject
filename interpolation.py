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

class interp:
    def peak_interpolation(self, mat_i, mat_x): 
        prev_x=mat_x[0]
        peak_i=[0]; peak_x=[mat_x[0]]
        if prev_x < mat_x[1]: incr=1
        else: incr=0

        xlen = len(mat_x)
        #print(f'mat_i = {mat_i}({len(mat_i)}개),\nmat_x = {mat_x}({len(mat_x)}개)')
        ###########  피크탐색  ###################
        for i in range(1, xlen):
            if prev_x >= mat_x[i]: #감소중
                if incr==1: #증가하다가 감소시작한거면 = 피크
                    peak_i.append(i); peak_x.append(prev_x)
                    incr=0 #감소시작을 알림
            elif prev_x < mat_x[i]: incr=1 #증가하면 증가시작을 알림    
            prev_x = mat_x[i]
        
        ###########  보간진행  ###################
        #print(f'peak_i = {peak_i}({len(peak_i)}개), peak_x = {peak_x}({len(peak_x)}개)')
        f = interpolate.interp1d(peak_i, peak_x, kind='quadratic')
        interp_i = np.array(mat_i[peak_i[0]:peak_i[-1]+1])
        f_y = f(interp_i)
        return interp_i, f_y


    #################polynomial regression
    def data_interpolation(self, poly_num, mat_i, interp_i, mat_x, deriv=True):
        #학습데이터: (interp_i, mat_x). 피크보간값을 다항식으로 fitting시키고 싶기 때문
        #예측데이터: (mat_i, ?)
        #fit_transform: 훈련집합이 가진 평균,분산의 분포에 맞게 정규화하는 것
        #trasnform: fit_transform으로 얻은 정규화값들에 맞게끔 테스트집합의 입력값을 스케일링(맞추는 것)
        poly = PolynomialFeatures(degree=poly_num)
        new_i = poly.fit_transform(interp_i.reshape(-1,1)) #interp_i를 degree차 함수에 맞는 x값으로 스케일링=변형(reshape와 같은 형태여야 하나봄)
        
        r = LinearRegression()
        r.fit(new_i, mat_x) #degree차 함수에 맞게 변형된 훈련집합의 x값과 y값인 (x,y)로 회귀모델 학습(=예측값 계산)

        mat_i2 = np.array(mat_i).reshape(-1,1)
        ypred = r.predict(poly.transform(mat_i2)) #예측용 x좌표도 이전 다항식에 맞게 변형시키고, 학습한 모델(r)로 y값 예측

        #print(poly.get_feature_names()) #차수 확인
        #print(r.coef_) #계수 확인
        
        ############################
        deg = poly_num #deg차 다항식
        coef =  r.coef_#다항식의 계수
        #print(coef)

        if deriv == True:  return mat_i2, ypred, self.deriv(deg, coef) #극값도 반환
        else: return mat_i, ypred #(x,y)반환
    ##############################################

    def deriv(self, deg, coef): #계수로 식을 제작하여 미분
        x = sp.symbols('x'); sstr = ''
        #print(np.round(coef,2))
        for i in range(deg+1):
            if coef[i]<0: sstr=sstr.rstrip('+') #음수인 경우만 특수처리
            sstr+=str(coef[i]) + '*x**' + str(i) + '+' #차수와 계수기반으로 다항식 작성
        sstr=sstr[:-1] #마지막 +는 제거
        #print(sstr)

        ############## ↑ 다항식 제작 완료 ↑ ###############
        
        fx = sp.sympify(sstr)
    
        f1 = sp.Derivative(fx, x).doit() #도함수
        f2 = sp.Derivative(f1,x).doit() #이계도함수
        
        ex = sp.solve(f1) #극값의 x좌표구하기

        #print(ex)
        return ex
