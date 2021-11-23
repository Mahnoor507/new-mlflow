import os
import warnings
import sys
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
from mlflow.tracking import MlflowClient
def foo():
  print("-------------------------------------------------------------------------------------------------")
  print("------------------------------------------ Previous Model details -------------------------------")
  print("-------------------------------------------------------------------------------------------------")
def foo2():
  print("-------------------------------------------------------------------------------------------------")
  print("------------------------------------------ New Model details ------------------------------------")
  print("-------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    client = MlflowClient()
    np.random.seed(40)
    df = pd.read_csv ('heart.csv')
    X = df.drop(['target'], axis=1)
    y = df.target
    X_std = StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=433)

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    #set_tracking_uri()
    input=int(sys.argv[1]) if len(sys.argv[1]) >= 1 else 100
    #mlflow.create_experiment("Mlflow Practice Experiments")
    with mlflow.start_run(run_name="Mlflow Practice") as run:
      flag=False
      model=RandomForestClassifier(n_estimators=input)
      model.fit(x_train,y_train)
      y_pred = model.predict(x_test)
      accuracy = model.score(x_test,y_test)
      recall=recall_score(y_test, y_pred)
      f1=f1_score(y_test,y_pred)
      precision=precision_score(y_test,y_pred)
      print(accuracy)
      mlflow.log_param("n_estimators", input)
      mlflow.log_metric("accuracy", accuracy)
      mlflow.log_metric("recall", recall)
      mlflow.log_metric("f1 score", f1)
      mlflow.log_metric("precision", precision)     
      mlflow.sklearn.log_model(model, "model")
      filter_string="name = 'random-forest-model'"
      tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
      new=client.search_registered_models(filter_string=filter_string)
      if(new):
        models=new[0]._latest_version
        id=models[0].run_id  
        ok=client.get_metric_history(run_id=id,key="accuracy")
        version_accuracy=float(ok[0].value)
        if(accuracy>version_accuracy):
          mlflow.sklearn.log_model(model, "model", registered_model_name="random-forest-model")
          foo()
          print("Model name: ", models[0].name)
          print("Model Version: ",models[0].version)
          print("Model's Accuracy: ", version_accuracy)
          foo2()
          new=client.search_registered_models(filter_string=filter_string)
          print("New Registered Model's name: ", new[0].latest_versions[0].name)
          print("New Registered Model's Version: ",new[0].latest_versions[0].version)
          print("New Registered Model's Accuracy: ",accuracy)
        else:print("Newly trained model not registered.")
      else:
        mlflow.sklearn.log_model(model, "model", registered_model_name="random-forest-model") 
        print("No registered Model found registering as version 1")
       
      

