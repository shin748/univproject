# 1. dicom with optical flow
 argv를 사용하지 않는 파일임<br/>
 가급적 ```excel().show_plot(pat_num, preprocess, interpolate, cycle, opt)```만 사용하길 바람 <br/>
 pat_num: 환자번호 <br/>
 preprocess: 전처리 True/False <br/>
 interpolate: 피크보간 True/False <br/>
 cycle: 피크보간 횟수 !!!!!!!! 이 버전에선 1이외에 기입하지 말 것!!!!!!!!!!!<br/>
 opt: 옵티컬 플로우 사용 True/False (False시 intensity) <br/>

# 2. excel
 경로를 직접 코드에 지정해야함.
 파일명은 필히 **'경로/환자번호/환자번호(파일번호).dcm'** 형태여야함
 
# 3. optflow
 70번째줄 ```if dx>=0 and dy>3:```의 dx, dy값을 수정하여 벡터길이를 조작
