import dill
import pandas as pd
import os
import json
from datetime import datetime
import logging
from os.path import isfile, join
from os import listdir


# path = os.environ.get('PROJECT_PATH', '..')
path = os.path.expanduser('~/airflow_hw')


# def version():
#     return model['metadata']


# def predict(model):
#     data = []
#     mypath = f'{path}/data/test/'
#     onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#     for _ in range(len(onlyfiles)):
#         with open(f'{mypath}{onlyfiles[_]}') as file1:
#             content = json.load(file1)
#         data.append(content)
#     df = pd.DataFrame.from_dict(data, orient='columns')
#     onlyfiles_ids = []
#     for _ in range(len(onlyfiles)):
#         onlyfiles_ids.append(onlyfiles[_].split('.')[0])
#     result_of_prediction = []
#     for _ in range(len(onlyfiles)):
#         y = model.predict(df.loc[_:_])
#         result_of_prediction.append(y[0])
#     result_df = pd.DataFrame(
#         {'car_id': onlyfiles_ids,
#          'pred': result_of_prediction
#          })
#     preds_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
#     result_df.to_csv(preds_filename, index=False)
#     logging.info(f'Predictions are saved as {preds_filename}')
def predict():
    last_model = sorted(os.listdir(f"{path}/data/models"))[-1]
    with open(f"{path}/data/models/{last_model}", 'rb') as file:
        model = dill.load(file)
    test_cars = os.listdir(f"{path}/data/test")
    preds = {'card_id': [], 'pred': []}

    for car_id in test_cars:
        with open(f"{path}/data/test/{car_id}", 'rb') as file_2:
            car = json.load(file_2)
        df = pd.DataFrame(car, index=[0])
        y = model.predict(df)
        preds['card_id'].append(car_id.split('.')[0])
        preds['pred'].append(y[0])

    df_preds = pd.DataFrame(preds)
    now = datetime.now().strftime("%Y%m%d%H%M")
    df_preds.to_csv(f'{path}/data/predictions/{now}.csv', index=False)


if __name__ == '__main__':
    predict()
    # latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    # with open(f'{path}/data/models/{latest_model}', 'rb') as file:
    #     model = dill.load(file)
    # predict(model)
