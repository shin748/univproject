# 1. dicom analysis
 argv를 사용하지 않는 파일임<br/>
 가급적 ```excel().show_plot(환자번호, preprocess, interpolate, cycle=1, opt=False)```만 사용하길 바람 <br/>
 pat_num: 환자번호 <br/>
 preprocess: 전처리 True/False <br/>
 interpolate: 피크보간 True/False <br/>
 cycle=1: 피크보간 횟수 (현 시점에선 1로 사용할 것)<br/>
 opt: 옵티컬 플로우 사용 True/False (False시 intensity. False를 사용할 것) <br/>

# 2. excel
 경로를 직접 코드에 지정해야함.
 파일명은 필히 **'경로/환자번호/환자번호(파일번호).dcm'** 형태여야함
  ```별도의 denoise된 npy를 지정하므로 denoise(5번 참조)가 적용된 파일이 필요```

# 3. data_analysis
 1) 클러스터링을 통해 낮은 값과 높은 값을 분리한 다음
 2) 높은 값 그룹을 polynomial regression을 통해 2차함수로 피팅하여 미분
 3) 도출된 극댓값을 기준으로 좌측의 낮은 값 그룹의 끝 프레임을 이용해 TFC를 출력
 4) 우측의 낮은 값 그룹의 시작 프레임을 이용해 CCFC 출력 (튜닝값 수정중)

# 4. interpolation
 노이즈 제거 차원에서의 피크점만을 잇는 스무딩을 수행함.
 polynomial regression과 미분에 대한 메소드 또한 존재

# 5. intense_sum, optflow, preprocess
 intense_sum은 intensity counting
 optflow는  ```if dx>=0 and dy>3:```의 dx, dy값을 수정하여 벡터길이를 조작
 preprocess는 전처리(denoise를 수행할 경우 별도의 사전저장)