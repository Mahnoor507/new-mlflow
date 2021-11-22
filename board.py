import time
import dash
from dash import html
from dash import dcc
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
from mlflow.tracking import MlflowClient
app = dash.Dash()   #initialising dash app
client = MlflowClient()
model_name="random-forest-model"
filter_string = "name='{}'".format(model_name)
results = client.search_model_versions(filter_string)
l1=[]
l2=[]
l3=[]
print("-" * 80)
for res in results:
    c=time.ctime(int(res._last_updated_timestamp/1000))
    dt_obj = datetime.fromtimestamp(res._last_updated_timestamp/1000).strftime("%B %d, %Y")
    l1.append(dt_obj)
    ok=client.get_metric_history(run_id=res.run_id,key="accuracy")
    l2.append(float(ok[0].value)*100)
    l3.append("version: {}".format(res.version))
def stock_prices():
    fig = go.Figure([go.Scatter(x = l1, y =l2,\
                     line = dict(color = 'firebrick', width = 4), text=l3)])
    fig.update_layout(title = 'Accuracy over time',
                      xaxis_title = 'Date-time',
                      yaxis_title = 'Accuracy'
                      )
    fig.update_traces(textposition='top right',mode='lines+markers+text', marker=dict(color='#5D69B1', size=8))
    return fig        

 
app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'Model comparison', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),

        
        dcc.Graph(id = 'line_plot', figure = stock_prices())    
    ]
                     )
if __name__ == '__main__': 
    app.run_server()    