import json
import requests
import pandas as pd

import mlflow
port = 1234

if __name__ == '__main__':
    # Load test set
    df = pd.read_csv ('heart.csv')
    X = df.drop(['target'], axis=1)
    D=X.iloc[:2]
    input_data = D.to_json(orient="split")
    endpoint = "http://20.62.252.142:{}/invocations".format(port)
    headers = {"Content-type": "application/json; format=pandas-records"}
    prediction = requests.post(endpoint, json=json.loads(input_data))
    print(prediction.text)
