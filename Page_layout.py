
import streamlit as st
from PIL import Image
from sklearn import datasets
import pandas as pd


class main_page(object):
    def __init__(self):
        

        st.set_page_config(layout="wide")
        #####Main Page s
        self.image=Image.open('media/bedo2.png')
        st.image(self.image,width=250)
        st.title('Bedo AI Virtual labs')

        #sidebar Content
        st.sidebar.image('media/bedo2.png',width=50)
        st.sidebar.header("What do you want to learn today")
        self.categories=st.sidebar.selectbox("Ml Category",options=['Supervised Learning','Unsupervised Learning'])
        self.Algotypes=st.sidebar.selectbox("Agorithm Type",options=['Regression','Classification'])

        self.classification_algorithms=['Logistic Regression','K Nearset Neighpors KNN','K Means','Support Vector Machine','Decession Tree','Naive Bayes','Benchmark Algorithms']
        self.Regression_algorithms=['Logistic Regression','Support Vector Machine','Decession Tree','Naive Bayes','Benchmark Algorithms']

        self.algorithm=st.sidebar.radio('Algorithm',options=self.classification_algorithms if self.Algotypes=='Classification'else self.Regression_algorithms )
        self.learntype=st.sidebar.radio("learning enviornment",options=['Tutorials','Algorithm in Action'])
        new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">'+ self.categories + " 👉" + self.Algotypes +" 👉" + self.algorithm + '</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        
 
 
    def gen_new_data():
        Feature,Target = datasets.make_classification(
        n_samples=100,  # row number
        n_features=3, # feature numbers
        n_classes = 2, # The number of classes 
        n_redundant=0,
        n_clusters_per_class=1,#The number of clusters per class
        weights=[0.5,0.5] )    
        df=pd.DataFrame(Feature,Target)
        df=df.drop(columns=2)
        df[2]=df.index
        df=df.reset_index(drop=True)
        #scaler=MinMaxScaler()
        #df[[0, 1]] = scaler.fit_transform(df[[0, 1]])
        df.to_csv('data/new.txt',sep=',',index=False)
        return df
    def alignh(lines,colname):
       for i in range (1 , lines):
         colname.markdown("#")
