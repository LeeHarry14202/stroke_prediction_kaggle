from curses import raw
from sqlite3 import DateFromTicks
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()


def draw_missing_data_table(dataframe):
    total_missing_data = dataframe.isnull().sum()
    total_row_dataframe = dataframe.isnull().count()
    percent_missing_data = (total_missing_data / total_row_dataframe)
    missing_data_table = pd.concat([total_missing_data , percent_missing_data], axis=1, keys=['Total', 'Percent'])
    return missing_data_table


def draw_bar_chart(x_axis,y_axis,x_name= None, y_name = None,x_axis_rotation = None):
    plt.bar(x = x_axis, height = y_axis, color = 'green')
    plt.xticks(x_axis,rotation = x_axis_rotation)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def draw_pie(data, labels, name =None):
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:10]
    #create pie chart
    plt.pie(
        data, labels = labels, 
        colors = colors, autopct='%1.2f%%', 
        radius= 1.5)
    # x_axis_legend = -0,5
    # y_axis_legend = 0.2
    plt.legend(loc="lower right", bbox_to_anchor=(-0.5, 0.2))
    plt.title(
        name, 
        fontsize = 20 , 
        loc='left',
        x = -0.75)
    plt.show()
