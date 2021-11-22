import json
import requests
import pandas as pd

import mlflow
port = 1234

if __name__ == '__main__':
    # Load test set
   
    df = pd.read_csv ('heart.csv')
    X = df.drop(['target'], axis=1)
    print(X)
    D=X.iloc[:2]
    print(D)

    input_data = D.to_json(orient="split")
    print(input_data)
    endpoint = "http://20.62.252.142:{}/invocations".format(port)
    headers = {"Content-type": "application/json; format=pandas-records"}
    print(headers)
    prediction = requests.post(endpoint, json=json.loads(input_data))
    print(prediction.text)

