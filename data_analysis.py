import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from interpolation import interp

class analysis:
    def clustering(self, isgmm, draw, num, xdata, ydata):
        '''
        #n_splits: 분할개수
        #shuffle: 데이터를 랜덤하게 섞어놓고 시작
        cluster = GaussianMixture(n_components=3, random_state=0)  
        vald = KFold(n_splits=3, shuffle=True, random_state=1)
        #vald = StratifiedKFold(n_splits=5)
        for train_i, test_i in vald.split(interp_i1):
            train = interp_i1[train_i].reshape(-1,1)
            test = interp_i1[test_i].reshape(-1,1)
            cluster.fit(train); cluster.fit(test)
            cgroup = cluster.predict(interp_i1.reshape(-1,1)
        Cross validation등을 사용하면 노이즈를 더 넣는 격이 된다.
        '''

        colors = ['lime', 'yellow', 'magenta', 'cyan', 'grey']
        #colors2 = ['mediumslateblue', 'olive', 'lightcoral', 'darkcyan', 'grey']

        if isgmm == True: cluster = GaussianMixture(n_components=num, random_state=0)   
        else: cluster = KMeans(n_clusters=2, random_state=5)
        cluster.fit(ydata.reshape(-1,1))
        cluster.fit(np.array(ydata).reshape(-1,1))
        c_res = cluster.predict(ydata.reshape(-1,1))

        gsize = c_res.shape[0]
        prej=0; gcnt=0; group=[c_res[0]]
        
        for j in range(gsize):
            prev=c_res[prej]; curr=c_res[j]
            if prev != curr:
                if draw==True: plt.plot(xdata[prej:j+1], ydata[prej:j+1], color=colors[c_res[prej]])
                if gcnt < num: group.append(curr); gcnt+=1 #gcnt < n_components-1
                prej=j
        if draw==True: plt.plot(xdata[prej:j+1], ydata[prej:j+1], color=colors[c_res[prej]])
        

        return c_res, group #클러스터링 결과와 군집 등장순서(group[0~x])를 알려줌
    
    ####################################################
    def localminmax(self, ydata, c_res, group): #좌측=TFC를 위한 극대극소
        gsize = c_res.shape[0]; chk=0; group0_end=0; group1_end=0
        '''
        highpeak_i=[]
        maxj=0; maxy = max(ydata)
        incr=1; stop=0
        for j in range(1,gsize): 
            if ydata[j] > ydata[maxj]: maxj=j
            ############################################
            if c_res[j]==group[1]: #상위구간(그룹1) 진입
            #if c_res[j]==group[1]: #상위구간(그룹1) 진입
                if group0_end == 0: group0_end=j

                if ydata[j] == maxy:
                    highpeak_i.append(j)
                    stop=1
                    #break
                
                ####################
                if ydata[j-1] >= ydata[j]: #감소중이면
                    #print(f'{j-1}피크진입')
                    chk=1
                    if incr==1: #근데 증가하다가 감소시작한거면 = 피크
                        if stop==0: highpeak_i.append(j-1) #상위구간 피크만 추출
                        #print(f'{j-1}추가')
                        incr=0 #감소시작을 알림
                elif ydata[j-1] < ydata[j]: incr=1 #증가하면 증가시작을 알림   
                #####################
                        
            elif c_res[j]!=group[1] and chk==1: group1_end=j; break #상위구간 이탈   
            ############################################
        
        avg_highpeak_i=0
        h_len = len(highpeak_i)
        #avg_highpeak_i = np.mean(np.array(highpeak_i))
        
        for j in range(h_len): avg_highpeak_i+=(1/highpeak_i[j])
        avg_highpeak_i /= h_len
        avg_highpeak_i = int(1/avg_highpeak_i) #조화평균
        '''

        for j in range(1,gsize): 
            if c_res[j]==group[1]: chk=1
            elif c_res[j]!=group[1] and chk==1: group1_end=j; break #상위구간 이탈   
                
        Llocalmin=0
        for j in range(group0_end, 0, -1):
            if ydata[j] <= ydata[j-1]: #감소하다가 다시 증가 = 아마도 그룹1의 마지막 극소점 (이후 상승하다 그룹2로 이동)
                Llocalmin=j; break

        TFCS = int(Llocalmin*0.5 + group0_end*0.5) #TFCS = 구간1 마지막 극솟점 to 구간(1,2)값

        mat_i2, ypred, ex = interp().data_interpolation(2, np.array(range(group0_end, group1_end+1)), np.array(range(group0_end, group1_end+1)), ydata[group0_end:group1_end+1], True)
        plt.plot(mat_i2, ypred, color='orange')
        TFCE = int(group0_end*0.5 + ex[0]*0.5) #TFCE = 구간(1,2)값 to 구간2 극대점

        CCFCS, CCFCE = self.R_localminmax(ydata, c_res, group, ex[0])

        return TFCS, TFCE, CCFCS, CCFCE

    ####################################################
    def get_TFC_CCFC(self, xdata, ydata, isgmm, draw):
        c_res, group = self.clustering(2, isgmm, draw, xdata, ydata)
        return self.localminmax(ydata, c_res, group)
    ####################################################

    def R_localminmax(self, ydata, c_res, group, ex):
        gsize = len(c_res)-1
        group0_start=0
        for i in range(gsize, 0, -1):
            if c_res[i] != group[0]: group0_start = i+1; break
        
        Rlocalmin=-1
        for i in range(group0_start, len(ydata)):
            if ydata[i-1] < ydata[i]: Rlocalmin=i-1
        

        CCFCS = int(ex*0.8 + group0_start*0.2) #CCFCS = 구간2 극대점 to 구간(2,3)값
        CCFCE = int(group0_start*0.9 + Rlocalmin*0.1) #CCFCE = 구간(2,3)값 to 구간3 최초극소점

        return CCFCS, CCFCE




    






        
    '''
    #튜닝된 평균값보다 큰 피크점 기준으로 구간을 나눠서 다항식 피팅해보는 파트
    avgx = np.mean(mat_x2) + (0.5*(max(mat_x1)-np.mean(mat_x1)))
    prev_x=mat_x1[0]
    high_peak=-1
    if prev_x < mat_x1[1]: incr=1
    else: incr=0
    for j in interp_i1[1:]:
        #print(f'prev_x={prev_x}, max={max(mat_x1)}, avgx={avgx}')
        if prev_x >= mat_x1[j]:
            if incr==1 and prev_x >=avgx: #피크점인데 평균값보다 큰 경우면 우리가 찾는 높은 피크점
                high_peak = j; break
            elif incr==1: incr=0
        elif prev_x < mat_x1[j]: incr=1 #증가하면 증가시작을 알림    
        prev_x = mat_x1[j]
    '''
    '''
    cluster = GaussianMixture(n_components=2, random_state=0)  
    cluster.fit(mat_x1[:high_peak].reshape(-1,1))
    cluster.fit(np.array(mat_x[:high_peak]).reshape(-1,1))
    c_res = cluster.predict(mat_x1[:high_peak].reshape(-1,1))
    gsize = c_res.shape[0]
    prej=0; gcnt=0; group=[]
    #######그룹1,2,3 파악 및 그래프 생성
    for j in range(gsize):
        prev=c_res[prej]; curr=c_res[j]
        if prev != curr:
            plt.plot(interp_i1[prej:j+1], mat_x1[prej:j+1], color=colors[c_res[prej]])
            if gcnt<2: group.append(prev); gcnt+=1
            prej=j
    plt.plot(interp_i1[prej:j+1], mat_x1[prej:j+1], color=colors[c_res[prej]])
    ################################################################
    cluster2 = GaussianMixture(n_components=5, random_state=1)
    cluster2.fit(mat_x1[high_peak:].reshape(-1,1))
    cluster2.fit(np.array(mat_x[high_peak:]).reshape(-1,1))
    c_res2 = cluster2.predict(mat_x1[high_peak:].reshape(-1,1))
    high_peak-=1
    gsize2 = c_res2.shape[0]
    prej=0; gcnt2=0; group2=[]
    #######그룹1,2,3 파악 및 그래프 생성
    #print(f'[0, {len(interp_i1)}] but c_res2 is [0, {len(c_res2)}]')                
    for j in range(gsize2):
        prev=c_res2[prej]; curr=c_res2[j]
        if prev != curr:
            #print(f'{high_peak+prej} to {high_peak+j+1}')
            plt.plot(interp_i1[high_peak+prej:high_peak+j+1], mat_x1[high_peak+prej:high_peak+j+1], color=colors2[c_res2[prej]])
            if gcnt2<2: group.append(prev); gcnt2+=1
            prej=j
    plt.plot(interp_i1[high_peak+prej:high_peak+j+1], mat_x1[high_peak+prej:high_peak+j+1], color=colors2[c_res2[prej]])
    plt.axvspan(high_peak-1, high_peak+1, facecolor='orange')
    '''
    '''
    #print(f'high_peak={high_peak}')
    plt.plot(interp_i1, mat_x1, '-', color=colors[i])
    p1, p2 = self.data_interpolation(5, mat_i[:high_peak], interp_i1[:high_peak], mat_x1[:high_peak]) #0~구한 피크점까지, 1차 보간에 regression모델 적용
    plt.plot(p1, p2, '-', color=colors[i+1])
    p3, p4 = self.data_interpolation(5, mat_i[high_peak:], interp_i1[high_peak:], mat_x1[high_peak:]) #구한 피크점~끝까지, 1차 보간에 regression모델 적용
    plt.plot(p3, p4, '-', color=colors[i+2])
    '''   