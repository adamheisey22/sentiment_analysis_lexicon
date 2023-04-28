import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import date
from sklearn.metrics import accuracy_score,classification_report

def results_graph(df, data_column1, data_column2):
    trial_name = date.today()
    
    fig = make_subplots(rows=1, cols=2, specs=[[{"type" : "pie"}, {"type": "pie"}]], subplot_titles=("VADER", "other source"))

    values = df[data_column1].value_counts()
    #labels = df['Rating'].unique().tolist()

    values1 = df[data_column2].value_counts()
    #labels1 = df['Overall Sentiment'].unique().tolist()

    labels = ['Positive', 'Negative']

    fig.add_trace(go.Pie(
        values=values1,
        labels=labels,
        hole=.3),
        row=1, col=1
    )

    fig.add_trace(go.Pie(
        values=values,
        labels=labels,
        hole=.3),
        row=1, col=2
    )

    #data = [trace1, trace2]
    #fig = go.Figure(data=data, layout=layout)
    fig.update_layout(height=600, width=800, title_text="VADER ADJ vs. other source")
    fig.show()

def results_matrix(df, data_column1, data_column2):
    trial_name = date.today()
    
    fig = make_subplots(rows=1, cols=2, specs=[[{"type" : "pie"}, {"type": "pie"}]], subplot_titles=("VADER", "other source"))

    values = df[data_column1].value_counts()
    #labels = df['Rating'].unique().tolist()

    values1 = df[data_column2].value_counts()
    #labels1 = df['Overall Sentiment'].unique().tolist()

    labels = ['Positive', 'Negative']

    fig.add_trace(go.Pie(
        values=values1,
        labels=labels,
        hole=.3),
        row=1, col=1
    )

    fig.add_trace(go.Pie(
        values=values,
        labels=labels,
        hole=.3),
        row=1, col=2
    )

    #data = [trace1, trace2]
    #fig = go.Figure(data=data, layout=layout)
    fig.update_layout(height=600, width=800, title_text="VADER ADJ vs. other source")
    print(classification_report(df[data_column1],df[data_column2]))
    print(accuracy_score(df[data_column1],df[data_column2]))
    with open(f'eval/{trial_name}.txt',"w") as f:
        print(f'other source: {values}', file=f)
        print("  ")
        print(" --------------------------------- ", file=f)
        print("  ")
        print(f'VADER : {values1}', file=f)
        print("  ")
        print(" --------------------------------- ", file=f)
        print("  ")
        a_score = (accuracy_score(df[data_column1],df[data_column2]))
        print(f'The Accuracy Score is: {a_score}', file=f)
        print("  ")
        print(" --------------------------------- ", file=f)
        print("  ")
        print(classification_report(df[data_column1],df[data_column2]), file=f)
        print("  ")
        print(" --------------------------------- ", file=f)
   