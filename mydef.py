from curses import raw
from sqlite3 import DateFromTicks
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def find_percent_missing_data(dataframe):
    for col in dataframe.columns:
        missing_data = dataframe[col].isna().sum()
        missing_percent = (missing_data  / len(dataframe))* 100
        print(f"{col}: {missing_percent}% ")

def draw_bar_chart(x_axis,y_axis,x_name= None, y_name = None,x_axis_rotation = None):
    plt.bar(x = x_axis, height = y_axis, color = 'green')
    plt.xticks(x_axis,rotation = x_axis_rotation)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

def draw_pie(data,labels):
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:10]
    #create pie chart
    plt.pie(
        data, labels = labels, 
        colors = colors, autopct='%.0f%%', 
        radius= 2)
    # x_axis_legend = -0,5
    # y_axis_legend = 0.2
    plt.legend(loc="lower right", bbox_to_anchor=(-0.5, 0.2))
    plt.show()

def add(x,y):
    return x+y