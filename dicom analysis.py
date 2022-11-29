from excel import excel
#py baseline

#15, 21 , 26(h)
def main():
    #excel().show_plot(2, bas=False, preprocess=True, interpolate=True, opt=False)
    #excel().show_plot(43, bas=True, preprocess=True, interpolate=True, opt=False)
    excel().set_all_plot(save=False, bas=False, preprocess=True, interpolate=True, opt=False)
    #bas: bas=미투여, hyp=투여
    #preprocess: 전처리 사용/미사용
    #interpolate: 피크 보간법 사용/미사용 
    #opt: 옵티컬플로우 사용 / 밝기값 사용
    
if __name__ == '__main__': main()
