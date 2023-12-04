import torch
import torch.nn.functional as F
import torchmetrics
from nn import ANN

config = {

   'preprocess' : {
  

    "train-csv": "/home/data/train.csv",
    "test-csv" : "/home/data/test_eda.csv",
    "output-train-feas-csv" : "./data/trn_X.csv",
    "output-test-feas-csv" : "./data/tst_X.csv", 
    "output-train-target-csv" : "./data/trn_y.csv", 
    "output-test-target-csv" : "./data/tst_y.csv", 
    "scale-columns" : ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude'], 
    "target-col" : "target",
    "scaler" : "minmax"
},
  'wandb_runname': "test",
  'files': {
    'X_csv': './data/trn_X.csv',
    'y_csv': './data/trn_y.csv',
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
      'batch_size': 32,
      'shuffle': True,
    },
    'loss': F.mse_loss,
    'optim': torch.optim.Adam,
    'optim_params': {
      'lr': 0.0001,
    },
    'metric': torchmetrics.MeanSquaredError(squared=False),
    'device': 'cpu',
    'epochs': 10,
  },

  'cv_params':{
    'n_split': 5,
  },

}