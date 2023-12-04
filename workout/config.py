import torch
import torch.nn.functional as F
import torchmetrics
from nn import ANN

config = {

   'preprocess' : {
      "features":  ['lane_count', 'road_rating', 'maximum_speed_limit',
                  'weight_restricted', 'month', 'rough_road_name', 
                  'line_number', 'start_latitude', 'start_longitude',
                  'end_latitude', 'end_longitude','end_turn_restricted','start_turn_restricted',
                  "base_hour", "peak_season", 'multi_linked', 'connect_code', 'peak_hour'],
      "train-csv": "/home/data/train.csv",
      "test-csv" : "/home/data/test.csv",
      "output-train-feas-csv" : "./data/trn_X_1205.csv",
      "output-test-feas-csv" : "./data/tst_X_1205.csv", 
      "output-train-target-csv" : "./data/trn_y_1205.csv", 
      "output-test-target-csv" : "./data/tst_y_1205.csv", 
      "encoding-columns": ['start_turn_restricted', 'end_turn_restricted'],
      "scale-columns" : ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude'], 
      "target-col" : "target",
      "scaler" : "minmax"
  },
  'wandb_runname': "test",
  'files': {
    'X_csv': '/home/data/trn_X_1205.csv',
    'y_csv': '/home/data/trn_y_1205.csv',
    'output': './model.pth',
    'output_csv': './results/five_fold.csv',
  },

  'model': ANN,
  'model_params': {
    'input_dim': 'auto', # Always will be determined by the data shape
    'hidden_dim': [128, 128, 64, 32],
    'activation': "relu",
    'use_dropout': False,
    'drop_ratio': 0.3,
  },

  'train_params': {
    'data_loader_params': {
      'batch_size': 128,
      'shuffle': True,
    },
    'loss': F.mse_loss,
    'optim': torch.optim.Adam,
    'optim_params': {
      'lr': 0.001,
    },
    'metric': torchmetrics.MeanSquaredError(squared=False),
    'device': 'cuda',
    'epochs': 10,
  },

  'cv_params':{
    'n_split': 5,
  },

}
