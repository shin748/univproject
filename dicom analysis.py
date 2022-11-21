from excel import excel

#excel().show_optflow_plot(): 엑셀의 이름 열에 있는 n을 적어야함
#excel().save_all_optflow_plot(): 엑셀의 모든 대상의 하강벡터를 저장 (약 10분 소요)

#6, 24, 27, 45
def main():
    excel().show_plot(31, preprocess=True, interpolate=True, cycle=1, opt=False)
    #excel().set_all_plot(save=False, preprocess=True, interpolate=True, cycle=1, opt=False)
    #preprocess: 전처리 사용/미사용
    #interpolate: 피크 보간법 사용/미사용
    #cycle: 보간 횟수(적정은 2회일듯)
    #opt: 옵티컬플로우 사용 / 밝기값 사용
    
if __name__ == '__main__': main()