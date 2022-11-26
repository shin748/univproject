from excel import excel
#py baseline

#6, 24, 27, 45
#9, 21, 41
#33 파일 이상함??
def main():
    excel().show_plot(33, bas=True, preprocess=True, interpolate=True, cycle=1, opt=False)
    #excel().set_all_plot(save=True, bas=True, preprocess=True, interpolate=True, cycle=1, opt=False)
    #bas: bas=미투여, hyp=투여
    #preprocess: 전처리 사용/미사용
    #interpolate: 피크 보간법 사용/미사용 
    #cycle: 보간 횟수(적정은 1회일듯)
    #opt: 옵티컬플로우 사용 / 밝기값 사용
    
if __name__ == '__main__': main()
