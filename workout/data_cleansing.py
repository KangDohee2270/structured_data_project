import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class RoadDataEncoding:
    def __init__(self, data):
        self.data = data
        self.start_latitude_encoder = LabelEncoder()
        self.start_latitude_encoder.fit(train_data["start_latitude"])

        self.end_latitude_encoder = LabelEncoder()
        self.end_latitude_encoder.fit(train_data["end_latitude"])
        
        self.weight_restricted_encoder = LabelEncoder()
        self.weight_restricted_encoder.fit(train_data["weight_restricted"])

        self.enc_list = {"start_latitude": self.start_latitude_encoder, 
                         "end_latitude": self.end_latitude_encoder, 
                         "weight_restricted": self.weight_restricted_encoder}
    
    def le_transform(self, column: str):
        encoder = self.enc_list[column]
        self.data[column + "_enc"] = encoder.transform(self.data[column])
    
    # '-' → NaN 값 변경
    # def change_rough_replace_NaN(self): 
    #     self.data = self.data.replace('-', np.NaN)


    def replace_road_name(self):
        data = self.data
        data.loc[(data["road_rating"] == 107) & (data["weight_restricted"] == 32400.0) & (data["road_name"] == "-"), "road_name"] = "산서로"
        data.loc[(data["road_rating"] == 107) & (data["weight_restricted"] == 43200.0) & (data["road_name"] == "-"), "road_name"] = "중문로"

        data.loc[(data["start_node_name"] == "송목교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_node_name"] == "남수교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_node_name"] == "하귀입구") & (data["road_name"] == "-"), "road_name"] = "일반국도12호선"
        data.loc[(data["start_node_name"] == "양계장") & (data["road_name"] == "-"), "road_name"] = "일반국도12호선"
        data.loc[(data["start_node_name"] == "난산입구") & (data["road_name"] == "-"), "road_name"] = "지방도1119호선"
        data.loc[(data["start_node_name"] == "영주교") & (data["road_name"] == "-"), "road_name"] = "일반국도11호선"
        data.loc[(data["start_node_name"] == "서중2교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_node_name"] == "천제이교") & (data["road_name"] == "-"), "road_name"] = "산서로"
        data.loc[(data["start_node_name"] == "하나로교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_node_name"] == "신하교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_node_name"] == "야영장") & (data["road_name"] == "-"), "road_name"] = "관광단지1로"
        data.loc[(data["start_node_name"] == "월계교") & (data["road_name"] == "-"), "road_name"] = "산서로"
        data.loc[(data["start_node_name"] == "서울이용원") & (data["road_name"] == "-"), "road_name"] = "태평로"
        data.loc[(data["start_node_name"] == "김녕교차로") & (data["road_name"] == "-"), "road_name"] = "일반국도12호선"
        data.loc[(data["start_node_name"] == "어도초등교") & (data["road_name"] == "-"), "road_name"] = "일반국도16호선"
        data.loc[(data["start_node_name"] == "광삼교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_node_name"] == "오렌지농원") & (data["road_name"] == "-"), "road_name"] = "일반국도11호선"
        data.loc[(data["start_node_name"] == "우사") & (data["road_name"] == "-"), "road_name"] = "일반국도16호선"
        data.loc[(data["start_node_name"] == "서귀포시산림조합") & (data["road_name"] == "-"), "road_name"] = "지방도1136호선"
        data.loc[(data["start_node_name"] == "성읍삼거리") & (data["road_name"] == "-"), "road_name"] = "일반국도16호선"

        data.loc[(data["end_node_name"] == "남수교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["end_node_name"] == "농협주유소") & (data["road_name"] == "-"), "road_name"] = "월계교"
        data.loc[(data["end_node_name"] == "난산입구") & (data["road_name"] == "-"), "road_name"] = "지방도1119호선"
        data.loc[(data["end_node_name"] == "성읍삼거리") & (data["road_name"] == "-"), "road_name"] = "일반국도16호선"
        data.loc[(data["end_node_name"] == "김녕교차로") & (data["road_name"] == "-"), "road_name"] = "일반국도12호선"
        data.loc[(data["end_node_name"] == "한남교차로") & (data["road_name"] == "-"), "road_name"] = "서중2교"
        data.loc[(data["end_node_name"] == "서울이용원") & (data["road_name"] == "-"), "road_name"] = "태평로"
        data.loc[(data["end_node_name"] == "하귀입구") & (data["road_name"] == "-"), "road_name"] = "일반국도12호선"
        data.loc[(data["end_node_name"] == "어도초등교") & (data["road_name"] == "-"), "road_name"] = "일반국도16호선"
        data.loc[(data["end_node_name"] == "월계교") & (data["road_name"] == "-"), "road_name"] = "산서로"
        data.loc[(data["end_node_name"] == "양계장") & (data["road_name"] == "-"), "road_name"] = "일반국도12호선"
        data.loc[(data["end_node_name"] == "하나로교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["end_node_name"] == "광삼교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["end_node_name"] == "수간교차로") & (data["road_name"] == "-"), "road_name"] = "양계장"
        data.loc[(data["end_node_name"] == "난산사거리") & (data["road_name"] == "-"), "road_name"] = "난산입구"
        data.loc[(data["end_node_name"] == "서중2교") & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["end_node_name"] == "서귀포시산림조합") & (data["road_name"] == "-"), "road_name"] = "지방도1136호선"
        data.loc[(data["end_node_name"] == "옹포사거리") & (data["road_name"] == "-"), "road_name"] = "월계교"
        data.loc[(data["end_node_name"] == "진은교차로") & (data["road_name"] == "-"), "road_name"] = "하나로교"

        # 7번째자리에서 반올림 할 경우 data에서의 고윳값 갯수가 변하지 않습니다
        data[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]] = data[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]].apply(lambda x: round(x, 6))

        data.loc[(data["start_latitude"] == 33.409416) & (data["road_name"] == "-"), "road_name"] = "월계교"
        data.loc[(data["start_latitude"] == 33.402546) & (data["road_name"] == "-"), "road_name"] = "지방도1119호선"
        data.loc[(data["start_latitude"] == 33.471164) & (data["road_name"] == "-"), "road_name"] = "일반국도12호선"
        data.loc[(data["start_latitude"] == 33.411255) & (data["road_name"] == "-"), "road_name"] = "월계교"
        data.loc[(data["start_latitude"] == 33.405319) & (data["road_name"] == "-"), "road_name"] = "산서로"
        data.loc[(data["start_latitude"] == 33.322018) & (data["road_name"] == "-"), "road_name"] = "서중2교"
        data.loc[(data["start_latitude"] == 33.325096) & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_latitude"] == 33.408431) & (data["road_name"] == "-"), "road_name"] = "산서로"
        data.loc[(data["start_latitude"] == 33.284189) & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_latitude"] == 33.47339) & (data["road_name"] == "-"), "road_name"] = "양계장"


        data.loc[(data["end_latitude"] == 33.47339) & (data["road_name"] == "-"), "road_name"] = "일반국도12호선"
        data.loc[(data["end_latitude"] == 33.358358) & (data["road_name"] == "-"), "road_name"] = "일반국도16호선"
        data.loc[(data["end_latitude"] == 33.412573) & (data["road_name"] == "-"), "road_name"] = "월계교"
        data.loc[(data["end_latitude"] == 33.244882) & (data["road_name"] == "-"), "road_name"] = "산서로"
        data.loc[(data["end_latitude"] == 33.322018) & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_longitude"] == 126.259693) & (data["road_name"] == "-"), "road_name"] = "월계교"
        data.loc[(data["end_longitude"] == 126.261797) & (data["road_name"] == "-"), "road_name"] = "월계교"

        data.loc[(data["end_longitude"] == 126.414236) & (data["end_latitude"] == 33.255215) & (data["lane_count"] == 2) & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["end_longitude"] == 126.456384) & (data["end_latitude"] == 33.465863) & (data["lane_count"] == 2) & (data["road_name"] == "-"), "road_name"] = "애조로"

        data.loc[(data["start_longitude"] == 126.262739) & (data["start_latitude"] == 33.415854) & (data["lane_count"] == 2) & (data["road_name"] == "-"), "road_name"] = "월계교"
        data.loc[(data["start_longitude"] == 126.413687) & (data["start_latitude"] == 33.255431) & (data["lane_count"] == 2) & (data["road_name"] == "-"), "road_name"] = "중문로"
        data.loc[(data["start_longitude"] == 126.454583) & (data["start_latitude"] == 33.466433) & (data["lane_count"] == 2) & (data["road_name"] == "-"), "road_name"] = "애조로"
        data.loc[(data["start_longitude"] == 126.456384) & (data["start_latitude"] == 33.465863) & (data["lane_count"] == 2) & (data["road_name"] == "-"), "road_name"] = "애조로"

        self.data = data



    def change_rough_road_name(self, x):
        if '일반국도' in x:
            return 1
        elif '지방도' in x:
            return 2
        else:
            return 3

    def change_base_date(self):
        self.data["base_date"] = (self.data["base_date"] - 20000000) % 10000
        
    def preprocess_rough_road_name(self):
        self.data['rough_road_name'] = self.data['road_name'].apply(self.change_rough_road_name)


    def change_season(self, x):
        month = int(x / 100)
        
        if month in [12, 1, 2]:
            return 1
        elif month in [3, 4, 5]:
            return 2
        elif month in [6, 7, 8]:
            return 3
        else:
            return 4

    def preprocess_season(self):
        self.data['season'] = self.data['base_date'].apply(self.change_season)

    def change_month(self, x):
        month = int(x / 100)
        return month

    def preprocess_month(self):
        self.data['month'] = self.data['base_date'].apply(self.change_month)

    def change_peak_season(self, x):
        month = int(x / 100)
        return month in [7, 8]

    def preprocess_peak_season(self):
        self.data['peak_season'] = self.data['base_date'].apply(self.change_peak_season)

    def change_peak_hour(self, x):
        base_hour = x
        return base_hour not in [0, 1, 2, 3, 4, 5, 6, 7, 22, 23]

    def preprocess_peak_hour(self):
        self.data['peak_hour'] = self.data['base_hour'].apply(self.change_peak_hour)
    
    def preprocess_line_number(self):
        self.data['line_number'] = self.data['road_name'].str.extract('(\d+)', expand=False).fillna(0).astype(int)
    
    def preprocess_all(self):
        self.preprocess_rough_road_name()
        self.preprocess_line_number()
        self.change_base_date()
        self.preprocess_season()
        self.preprocess_month()
        self.preprocess_peak_season()
        self.preprocess_peak_hour()

        # 인코딩 컬럼 추가
        self.le_transform("start_latitude")
        self.le_transform("end_latitude")
        self.le_transform("weight_restricted")

if __name__ == "__main__":
    # Load the data
    print("Load the data...")
    test_data = pd.read_csv('/home/data/test_origin.csv')
    train_data = pd.read_csv('/home/data/test_origin.csv')

    train_data[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]] = train_data[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]].apply(lambda x: round(x, 6))
    test_data[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]] = test_data[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]].apply(lambda x: round(x, 6))
    
    print("Data Loading Complete. Preprocess train data first")
    train_data_encoding = RoadDataEncoding(train_data)
    # train_data_encoding.change_rough_replace_NaN() # '-' → NaN 값 변경
    # 결측치 처리
    train_data_encoding.replace_road_name() 
    print("Complete Fill NA")
    # 데이터 전처리 / 컬럼 추가
    train_data_encoding.preprocess_all()
    print("Complete Initial preprocessing")
    print("Preprocess test data ...")
    # test_data에 대한 전처리 수행 및 저장
    test_data_encoding = RoadDataEncoding(test_data)
    # 결측치 처리
    test_data_encoding.replace_road_name() 
    print("Complete Fill NA")
    # test_data_encoding.change_rough_replace_NaN()
    test_data_encoding.preprocess_all()
    print("Complete Initial preprocessing. Save 2 files")
    train_data_encoding.data.to_csv('./data/train_encoded.csv', index=False)
    test_data_encoding.data.to_csv('./data/test_encoded.csv', index=False)