import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os

# pip install -r requirements.txt 
warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)

df = pd.read_csv('./dataset/Mall_Customers.csv')
# print(df.head())

# print(df.describe())

# print(df.isnull().sum())


def histograms():
    plt.style.use('fivethirtyeight')
    n = 0
    for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(1 , 3 , n)
        plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
        sns.distplot(df[x] , bins = 20)
        plt.title('Distplot of {}'.format(x))
    plt.show()

def genders():
    plt.figure(1, figsize=(15,5))
    sns.countplot(y = "Gender", data=df)
    plt.show()

def relationships(relationship_array):
    plt.figure(1, figsize=(15, 7))
    n = 0
    for x in relationship_array:
        for y in relationship_array:
            n += 1
            plt.subplot(3,3, n)
            plt.subplots_adjust(hspace= 0.5, wspace=0.5)
            sns.regplot(x = x, y = y, data=df)
            plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)
    plt.show()
def age_vs_annual_gender():
    plt.figure(1, figsize = (15, 6))
    for gender in ['Male','Female']:
        plt.scatter(x = 'Age', y = 'Annual Income (k$)', data = df[df['Gender'] == gender], s = 200, alpha= 0.5, label = gender)
    plt.xlabel('Age'), plt.ylabel('Annual Income (k$)')
    plt.title('Age vs annual Income sorted by gender')
    plt.legend()
    plt.show()
def k_means_test():
    X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
    inertia = []
    for n in range(1 , 11):
        #using the SKLearn cluster kMeans algo
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
        algorithm.fit(X1)
        inertia.append(algorithm.inertia_)
        print(inertia)
# histograms()
# genders()
# relationships(['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)'])
# age_vs_annual_gender()
k_means_test()