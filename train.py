import os
import warnings
import sys
from mlflow.types import schema

import pandas as pd
import numpy as np
from pandas.core.algorithms import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,f1_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import mlflow
import mlflow.sklearn

from urllib.parse import parse_qsl, urljoin, urlparse

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    df = pd.read_csv ('heart.csv')
    X = df.drop(['target'], axis=1)
    y = df.target
    X_std = StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=433)

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    #set_tracking_uri()
    c=int(sys.argv[1]) if len(sys.argv) > 1 else 100

    #mlflow.create_experiment("Mlflow Practice Experiments")
    with mlflow.start_run(run_name="Mlflow Practice") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        model=RandomForestClassifier(n_estimators=c)
   
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        accuracy = model.score(x_test,y_test)
        recall=recall_score(y_test, y_pred)
        f1=f1_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        print(accuracy)

        mlflow.log_param("n_estimators", c)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1 score", f1)
        mlflow.log_metric("precision", precision)
        print(run_id,experiment_id)


        mlflow.sklearn.log_model(model, "model")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
  

