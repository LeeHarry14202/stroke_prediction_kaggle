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


def draw_pie(df, name =None):
    df = df.value_counts()
    #define Seaborn color palette to use
    colors = sns.color_palette('winter')[0:10]
    #create pie chart
    plt.pie(
        x = df, 
        colors = colors, 
        labels=list(df.index),
        autopct='%1.2f%%', 
        
        radius= 1.5,
        shadow =True,
        )
    # x_axis_legend = -0,5
    # y_axis_legend = 0.2
    plt.legend(loc="lower right", bbox_to_anchor=(-0.5, 0.2))
    plt.title(
        name, 
        fontsize = 20 , 
        loc='left',
        x = -0.75)
    plt.show()

def draw_head_map(df, vmin =None, vmax = None):
    fig, ax = plt.subplots(figsize = (10,8))

    first_row =1
    last_column = -1
    numerical_df = df[(df.select_dtypes(include=['float64']).columns)]
    # ones_like can build a matrix of booleans with the same shape of our data
    numerical_corr = numerical_df.corr()
    # ones_like can build a matrix of booleans with the same shape of our data
    ones_corr = np.ones_like(numerical_corr, dtype=bool)
    # remvove first row and last column 
    numerical_corr = numerical_corr.iloc[first_row:, :last_column]
    # np.triu: retun only upper triangle matrix
    mask = np.triu(ones_corr)[first_row:, :last_column]

    sns.heatmap(
        data =numerical_corr ,
        mask = mask,
        # Show number 
        annot = True,
        # Round number
        fmt = ".2f",
        # Set color
        cmap ='winter_r',
        # Set limitation of color bar (right)
        vmin = vmin, vmax = vmax,
        # Color of the lines that will divide each cell.
        linecolor = 'white',
        # Width of the lines that will divide each cell.  
        linewidths = 0.5);
    yticks = [i.upper () for i in numerical_corr.index]
    xticks = [i.upper () for i in numerical_corr.columns]

    ax.set_yticklabels(yticks, rotation = 0, fontsize =8);
    ax.set_xticklabels(xticks, rotation = 0, fontsize =8);

    title = 'HEADMAP OF NUMERICAL VARIABLES'
    ax.set_title(title, loc ='left', fontsize = 20);

def draw_multiple_categorical_chart(df, hue = None):
    row_of_chart =1
    col_of_chart = 2
    list_categorical_column = list(df.select_dtypes(include=['object', 'int64']).columns)
    index = 0
    while index < len(list_categorical_column):
        fig, (ax1, ax2) = plt.subplots(row_of_chart, col_of_chart, figsize=(15,5))
        sns.countplot(data = df, x = list_categorical_column[index], hue = hue, palette='winter_r',ax =ax1)
        index +=1
        sns.countplot(data = df, x = list_categorical_column[index], hue = hue, palette='winter_r',ax =ax2)
        index +=1


def score_module_classifier(x_train, y_train, x_test, y_test):
    # Import ML Libraries
    from sklearn.metrics import accuracy_score, recall_score, precision_score , confusion_matrix
    from xgboost import XGBClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    classifiers = [
        [XGBClassifier(random_state =1, use_label_encoder=False),'XGB Classifier'], [RandomForestClassifier(random_state =1),'Random Forest'], 
        [LGBMClassifier(random_state =1),'LGBM Classifier'], [KNeighborsClassifier(), 'K-Nearest Neighbours'], 
        [SGDClassifier(random_state =1),'SGD Classifier'], [SVC(random_state =1),'SVC'],
        [GaussianNB(),'GaussianNB'],[DecisionTreeClassifier(random_state =1),'Decision Tree Classifier']
    ];

    for cls in classifiers:
        model = cls[0]
        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_test)

        accuracy =  round(accuracy_score(y_test, y_pred), 2) *  100
        recall = round(recall_score(y_test, y_pred), 2) *  100
        precision = round(precision_score(y_test, y_pred), 2) *  100

        print(f"{cls[1]}")
        print ('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print("Accuracy : ", round(accuracy_score(y_test, y_pred), 2) *  100)
        print("Recall : ", round(recall_score(y_test, y_pred), 2) *  100)
        print("Precision : ", round(precision_score(y_test, y_pred), 2) *  100)
        print("---------------------------------")