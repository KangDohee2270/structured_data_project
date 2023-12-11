import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import os
import joblib


if __name__ == "__main__":
  exec(open("./config.py").read())
  cfg = config
  files = cfg.get('files')
  
  X_df = pd.read_csv(files.get('X_test_csv'), index_col=0).to_numpy(dtype=np.float32)
  Model = cfg.get('model')

  if not issubclass(Model, nn.Module):
    model = joblib.load(files.get('ml_output_model')) 
    result = model.predict(X_df).tolist()

  else:
    X_tst = torch.tensor(X_df.to_numpy(dtype=np.float32))
    ds = TensorDataset(X_tst)
    dl = DataLoader(ds)

    # model
    train_params = cfg.get('train_params')
    device = torch.device(train_params.get('device'))
    model_params = cfg.get('model_params')
    model_params['input_dim'] = X_tst.shape[-1]
    model = Model(**model_params).to(device)

    model.load_state_dict(torch.load(files.get("output")))
    model.eval()
    result = []
    print("A prediction is being made. Please wait a moment...")
    with torch.inference_mode():
      for X in dl:
        X = X[0].to(device)
        output = float((model(X)).squeeze())
        # print(output)
        result.append(output)

  test_origin = pd.read_csv("/home/data/test_origin.csv", index_col=0)
  test_id = test_origin.index.to_list()

  col_name = ["id", "target"]
  list_df = pd.DataFrame(zip(test_id, result), columns=col_name)
  model_name = os.path.splitext(os.path.basename(files.get("output")))[0]
  list_df.to_csv(files.get("submission_csv"), index=False)
  print('Save submission file completely')