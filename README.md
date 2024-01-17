
## Get started

### Environment Setting
- 실험 환경
  - Linux
  - Python 3.10

모든 패키지 및 라이브러리를 설치하려면, 터미널 창에서 `pip install -r requreiment.txt` 을 실행하세요

### Dataset
- `data` 폴더 내에 전처리가 완료된 파일이 존재합니다(용량 문제 때문에 학습 데이터셋의 약 1/5만 업로드)
- 전체 데이터셋으로 학습을 원한다면, [이 링크](https://dacon.io/competitions/official/235985/overview/description)에서 다운로드 가능합니다.
  - 다운로드 후 `workout/data_cleansing.py`를 실행해주세요.

### Start-Super Easy Way!
```bash
cd workout
mkdir results
# if you preprocess the origin data, please uncomment the following 2 lines
# python data_cleansing.py
# python preprocess.py
bash start.sh
```

## PR test + 데이터셋 후보군 구축 (11/24)


주차수요 예측 AI 경진대회
https://dacon.io/competitions/official/235745/overview/description
data : 2,900

대구 교통사고 피해 예측 AI 경진대회
https://dacon.io/competitions/official/236193/overview/description

**제주도 도로 교통량 예측 AI 경진대회 (결정)**
https://dacon.io/competitions/official/235985/overview/description
data : 4,701,217

## EDA (11/29)
### 도로명
- 도로명은 일반국도, 지방도, -로 구성되어 있음 -> 이들을 1,2,3으로 인코딩해서 분석
  -> -로가 국도, 지방도에 비해 평균속도가 낮았다 : -로는 시내도로이기 때문인 것으로 예상

### 날짜(월, 계절, 요일)
- 요일별은 거의 차이가 없음: 주말과 주중이 차이 없는것으로 보아, 평균속도가 관광객의 영향이 많이 받는 것으로 보임
- 계절별로 월을 나누어 확인해 본 결과, 여름(6-8월)이 가장 평균 속도가 낮음, 나머진 비슷
- 월별로 확인해 본 결과, 성수기인 7월이 가장 평균 속도가 낮았다.(7월: 35 < 나머지: 42): 계절보다는 성수기/비성수기가 더 결정요인이 큰것같음
  +) 데이터가 21.9 - 22.7의 데이터기 때문에, 8월 데이터는 확인할 수 없었음+) 테스트 데이터가 22.8월 데이터임.. <- 아마도 성수기로 봐야할 것 같음

### 시간
- 10-15시가 새벽-밤 대비 평균적으로 낮은 속도를 보였고, 특히 18시에 제일 낮은 평균속도를 보임: 18시는 관광객 + 퇴근 인구 때문에 가장 낮은 평균속도를 보였을 것으로 예상: 출근시간은 크게 막히지 않는 모습을 보임: 평균속도를 결정하는 것은 주민 < 관광객

### 차선 수
- 평균 속도: 2(44.98)>1(43.27)>>>>>3(34.91): 차선이 많은 도로일수록 오히려 많은 관광객이 몰리는 도로일 확률이 높다.

### 최고속도제한
- 40km 제한인 도로가 가장 평균 속도가 높음: 제한속도 40km 인 도로는 1차선에만 존재: 제한속도 40km 도로는 일반국도 12, 95 - 특히 일반국도 95가 평균속도(70)가 높음-> 잘 안쓰는 도로일 것으로 예상됨 
- 95번 도로는 산업도로, 12번 도로는 일주도로 -> 그래서 95번이 유독 빨랐을 것으로 예상

### 호선
- 일반국도/지방도는 호선마다 평균 속도가 다름:
  - 일반국도) 11(39.99)<99<12<16<95(70.44) 순으로 차이남
  - 지방도) 1120 이 가장 낮음(36)

### 위도/경도
- 더 잘해야할 것 같음. 데이터 부족시 추가조사 필요
- 초기실험에서는 제외하는것이 좋을 것 같음(minmax scaling 필요)
- 출발지-도착지 이름도 있지만, 같은 이름이라면 위도/경도가 같기 때문에 사용이 필요할지 생각해봐야할 것 같음

## git 관리 관련 참고할만한 사이트
- [A visual git reference](https://marklodato.github.io/visual-git-guide/index-ko.html)
- [간단하게 정리한 git 사용법](https://gin-girin-grim.tistory.com/10)
