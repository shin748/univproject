from excel import excel

#excel().show_optflow_plot(): 엑셀의 이름 열에 있는 n을 적어야함
#excel().save_all_optflow_plot(): 엑셀의 모든 대상의 하강벡터를 저장 (약 10분 소요)

def main():
    excel().show_optflow_plot(17, preprocess=True, interpolate=True)
    #preprocess: 전처리 사용/미사용
    #interpolate: 피크 보간법 사용/미사용
    
if __name__ == '__main__': main()