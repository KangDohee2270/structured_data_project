from typing import Literal, List
from dataclasses import dataclass

import pandas as pd

@dataclass
class HomeData:
  file_trn: str = '/home/data/train_tmp.csv'               # 훈련 데이터 CSV 파일의 기본 경로
  file_tst: str = './home/data/test.csv'
  target_col: str = 'target'
  features = ['day_of_week', 'base_hour', 'lane_count', 'maximum_speed_limit', 
        'month', 'rough_road_name', 'line_number', 'season', 'peak_season']                    
  # fill_num_strategy: Literal['mean', 'min', 'max'] = 'min'  # 누락된 숫자 값을 채우는 전략

  def _read_df(self, split:Literal['train', 'test']='train'):
    """
    특정 분할('train' 또는 'test')에 따라 CSV 파일을 읽고 전처리합니다.
    해당 분할에 대한 DataFrame을 반환합니다.
    """
    if split == 'train':
      df = pd.read_csv(self.file_trn)
      df_X = df[self.features]
      target = df[self.target_col]
      return df_X, target
    elif split == 'test':
      df = pd.read_csv(self.file_tst)
      df = df[self.features]
      return df
    raise ValueError(f'"{split}"은(는) 허용되지 않습니다.')

  def preprocess(self):

    X_trn, y_trn = self._read_df(split="train")
    X_tst = self._read_df(split="test")

    return X_trn, y_trn, X_tst

def get_args_parser(add_help=True):
  """
  명령행 인수 구문 분석을 위한 ArgumentParser 반환
  """
  import argparse

  parser = argparse.ArgumentParser(description="데이터 전처리", add_help=add_help)
  # 입력
  parser.add_argument("--train-csv", default="/home/data/train_eda.csv", type=str, help="훈련 데이터 CSV 파일")
  parser.add_argument("--test-csv", default="/home/data/test_eda.csv", type=str, help="테스트 데이터 CSV 파일")
  # 출력
  parser.add_argument("--output-train-feas-csv", default="./data/trn_X.csv", type=str, help="출력 훈련 특성")
  parser.add_argument("--output-test-feas-csv", default="./data/tst_X.csv", type=str, help="출력 테스트 특성")
  parser.add_argument("--output-train-target-csv", default="./data/trn_y.csv", type=str, help="출력 훈련 타겟")
  parser.add_argument("--output-test-target-csv", default="./data/tst_y.csv", type=str, help="출력 테스트 타겟")
  
  # 옵션
  parser.add_argument("--target-col", default="target", type=str, help="타겟 열")
  parser.add_argument("--fill-num-strategy", default="min", type=str, help="숫자 열 채우기 전략 (mean, min, max)")
  parser.add_argument("--features", default=['day_of_week', 'base_hour', 'lane_count', 'maximum_speed_limit', 
       'target', 'month', 'rough_road_name', 'line_number', 'season', 'peak_season'], type=list, help="사용할 features")
  return parser

if __name__ == "__main__":
  # 명령행 인수 파싱
  args = get_args_parser().parse_args()
  # 제공된 인수로 HomeData 인스턴스 생성
  home_data = HomeData(
    file_trn = args.train_csv,
    file_tst = args.test_csv,
    target_col=args.target_col,
    # args.fill_num_strategy
  )
  # 데이터 전처리 수행
  trn_X, trn_y, tst_X = home_data.preprocess()
  
  # 전처리된 데이터를 CSV 파일로 저장
  trn_X.to_csv(args.output_train_feas_csv)
  tst_X.to_csv(args.output_test_feas_csv)
  trn_y.to_csv(args.output_train_target_csv)
  