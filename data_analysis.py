import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from interpolation import interp

class analysis:
    #일반 클러스터링
    def clustering(self, num, isgmm, draw, xdata, ydata, color_start=0):
        colors = ['lime', 'yellow', 'magenta', 'cyan', 'grey']
        #colors2 = ['mediumslateblue', 'olive', 'lightcoral', 'darkcyan', 'grey']

        if isgmm == True: cluster = GaussianMixture(n_components=num, random_state=0)   
        else: cluster = KMeans(n_clusters=num, random_state=5)
        cluster.fit(ydata.reshape(-1,1))
        c_list = cluster.predict(ydata.reshape(-1,1)) #클러스터 리스트

        gsize = c_list.shape[0]
        prej=0; gcnt=0; group=[c_list[0]]
        
        for j in range(gsize):
            prev=c_list[prej]; curr=c_list[j]
            if prev != curr:
                if draw==True: plt.plot(xdata[prej:j+1], ydata[prej:j+1], color=colors[c_list[prej]+color_start])
                if gcnt < num: group.append(curr); gcnt+=1 #gcnt < n_components-1
                prej=j
        if draw==True: plt.plot(xdata[prej:j+1], ydata[prej:j+1], color=colors[c_list[prej]+color_start])
        
        return c_list, group #클러스터링 결과와 군집 등장순서(group[0~x])를 알려줌




    def get_ex(self, ex, start, end, degree): #함수에 출력된 극값중 가장 작은 값을 반환
        #print(f'ex={ex}')
        a=0; b=0
 
        if degree==2:
            a = ex[0]
            if str(a)[-1] == 'I': a=abs(complex(ex[0]))

            if a<start or a<0 or a>end: 
                print('there is no extream point'); a=0
            return a

        if degree==3:
            a=ex[0]; b=ex[1]
            swap=1; find_ex=0
            if str(a)[-1] == 'I': a=abs(complex(ex[0]))
            if str(b)[-1] == 'I': b=abs(complex(ex[1]))

            if a<start or a<0 or a>end:
                find_ex=b; swap=0
            if b<start or b<0 or b>end:
                find_ex=a; swap=0

            if swap==1:
                if a<b: find_ex = b
                else: find_ex = a
            
            return find_ex
    


    def get_gsection(self, c_list, group): #그룹1->그룹2->그룹1 형태로 클러스터링됐다 가정, (그룹1->그룹2)와 (그룹2->그룹1)으로 변환되는 지점을 반환
        gsize = c_list.shape[0]; chk=0
        group1_end=0; group2_end=0
        for i in range(1,gsize): 
            if c_list[i]==group[1]:
                if group1_end == 0: group1_end=i #구간1 끝부분
                chk=1
            elif c_list[i]!=group[1] and chk==1: #상위구간 이탈
                group2_end=i; break #구간2 끝부분
        
        return group1_end, group2_end




    def localminmax(self, ydata, c_list, group): #TFC, CCFC 계산
        gsize = c_list.shape[0]; chk=0; group1_end=0; group2_end=0

        ###구간1, 2 파악######################
        group1_end, group2_end = self.get_gsection(c_list, group)
        for i in range(group1_end, gsize): c_list[i] = group[0]


        ########################################
        ####구간1 미분##############
        mat_i2, ypred, ex = interp().data_interpolation(2, np.array(range(0,group1_end+1)), np.array(range(0,group1_end+1)), ydata[0:group1_end+1], True)
        plt.plot(mat_i2, ypred, color='orange')
        
        ex1 = self.get_ex(ex, 0, group1_end, 2)


        ####구간2 미분##############
        mat_i2, ypred, ex = interp().data_interpolation(2, np.array(range(group1_end, group2_end+1)), np.array(range(group1_end, group2_end+1)), ydata[group1_end:group2_end+1], True)
        plt.plot(mat_i2, ypred, color='orange')

        ex2 = self.get_ex(ex, group1_end, group2_end, 2)


        #####구간2 to 3 통째로 미분####
        #극값이 올바르게 출력되지 않음. 범위의 문제??
        mat_i2, ypred, ex = interp().data_interpolation(3, np.array(range(group1_end, gsize)), np.array(range(group1_end, gsize)), ydata[group1_end:gsize], True)
        plt.plot(mat_i2, ypred, color='orange')
        #############################################



        ######## polynomial regression으로 fitting된 함수에 대해 클러스터링############
        c_list2, group2 = self.clustering(2, True, True, mat_i2, ypred,2)
        group1_end2, group2_end2 = self.get_gsection(c_list2, group2)
        group1_end2 += group1_end; group2_end2 += group1_end
        
        #mat_i2, ypred, ex = interp().data_interpolation(2, np.array(range(group1_end2, gsize)), np.array(range(group1_end2, gsize)), ydata[group1_end2:gsize], True)
        #plt.plot(mat_i2, ypred, color='crimson')

        c_list3, group3 = self.clustering(2, True, True, mat_i2[group1_end:group1_end2], ypred[group1_end:group1_end2], color_start=2)
        group1_end3, group2_end3 = self.get_gsection(c_list3, group3)
        group1_end3+=group1_end; group2_end3+=group1_end #부정확함. 수정 필요
        print(f'극대점쪽 {group1_end3}, {group2_end3}')
        max_ex = int((group1_end3 + group2_end3)/2)

        c_list4, group4 = self.clustering(2, True, True, mat_i2[group1_end2:gsize], ypred[group1_end2:gsize], color_start=2)
        group1_end4, group2_end4 = self.get_gsection(c_list4, group4)
        group1_end4+=group1_end2; group2_end4=group1_end2 #부정확함. 수정 필요
        print(f'극소점쪽 {group1_end4}, {group2_end4}')
        min_ex = int((group1_end4+group2_end4)/2)

        print(f'max-ex={max_ex}, min-ex={min_ex}')
        ################################################################################

        ##### 프레임 계산: 아직 임시값이므로 정확한 값 아님 #################################################
        
        TFCS = int((group1_end/2) + (ex2-(group1_end/2))*0.2)
        TFCE = int((group1_end/2) + (ex2-(group1_end/2))*0.6)

        CCFCS = int(ex2 + ( ((gsize+group2_end)/2)-ex2)*0.05)
        #CCFCE = int(ex2 + ( ((gsize+group2_end)/2)-ex2)*0.8)
        CCFCE = group2_end 

        ''' 이하 또한 임시 tuning값이므로 사용을 권장하진 않음..
        TFCS = int(ex1*0.8 + ex2*0.2)
        TFCE = int(ex1*0.4 + ex2*0.6)

        CCFCS = int(ex2+((ex3-ex2)*0.005))
        CCFCE = int(ex2+(ex3-ex2)*0.5) 
        '''

        return TFCS, TFCE, CCFCS, CCFCE

    ####################################################
    def get_TFC_CCFC(self, xdata, ydata, isgmm, draw): #TFC, CCFC 출력부
        c_list, group = self.clustering(2, isgmm, draw, xdata, ydata)
        return self.localminmax(ydata, c_list, group)
    ####################################################
