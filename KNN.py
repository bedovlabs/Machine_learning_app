from mimetypes import knownfiles
from socket import gaierror
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from Page_layout import main_page
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


class Supervised_knn(object):
    
    
    def knn_tut(self):      

        #Animation functiondef logitic_tut(self):      

        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
            Try_col1,Try_col2=Try_algorithm.columns((5,5))
            #self.criterion=Try_col1.selectbox("Criterion",options=['gini', 'entropy', 'log_loss'])
            self.Kvalue=Try_col1.slider("K value ",1,10,1)
            
            Go_model_BTN=Try_col1.button("Go")
            #self.max_features=Try_col2.selectbox("Max Features",options=['None', 'auto', 'sqrt', 'log2'])
            #self.min_samples_split=Try_col2.slider("Minimum Samples Split",2,10,1)
            self.cluster_std=Try_col2.slider("Cluster standerd deviation",.5,10.0,.5)
            self.nsamples=Try_col2.slider("No of samples",100,1000,50)
            Generate_dataBTN =Try_col2.button("Generate New Data")

        
            if Go_model_BTN:
               fig=self.load_and_viusalize()
               st.plotly_chart(fig)
            
            
            if Generate_dataBTN:
                 graph=self.generate_data(self.Kvalue,self.nsamples,self.cluster_std)
                 st.pyplot(graph)

                
        with How_it_work:
            how_col1,how_col2=How_it_work.columns((8,1))
            how_col1.subheader("Algorithm structure")
            how_col1.image("media/knearst1.png")
            how_col1.subheader("Data piplining")
            how_col1.image("media/knearst2.png")
            how_col1.subheader("Application")
            how_col1.image("media/knearst3.png")
            how_col1.subheader("Advantages and Disadvantages")
            how_col1.image("media/knearst4.png")

    
        #Animation function


    def knn_inaction(self):      
        col1, col2 = st.columns((3,4))
        col1.subheader("Select Dataset")
        selected_dataset=col1.selectbox("Datasets",options=['Smart Irrigation','Iris','Smoke detection',
                                                          'Diabetes','Diseses','Image classifier',
                                                          'Article classifier','Digit recognition'])
        col2.subheader("Set KNN  Parameter")
        self.algorithm=col2.selectbox("Algorithm",options=['auto', 'ball_tree', 'kd_tree', 'brute'])
        self.n_neighbors=col2.slider("No of neighbors",3,15,1)
        self.leaf_size=col2.slider("leaf size",10,100,5)
       


        main_page.alignh(5,col1)
        Go =col1.button("Go")

        if Go :
            if selected_dataset=='Smart Irrigation':

                tab_irrigData,tab_irrigGraph=st.tabs(("Data","Graphs"))
                with tab_irrigData :
                    from smart_irrigation import knn_irr
                    irrigation_knn=knn_irr()
                    df=irrigation_knn.load_data()
                    tab_irrigData.subheader("Sample Data")
                    tab_irrigData.write(df.sample(10))
                    tab_irrigData.subheader("Irrigation data Statistics")
                    tab_irrigData.write(df.describe().transpose())
                    irr_col1,irr_col2=tab_irrigData.columns((6,6))
                    irr_col2.subheader("Missing Values ")
                    irr_col2.write(df.isnull().sum())
                    irr_col2.subheader(" Classes Representation")
                    irr_col2.write( df['Irrigation'].value_counts())
                    corr=df.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    with sns.axes_style("dark"):
                        f, ax = plt.subplots(figsize=(6, 6))
                        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                    irr_col1.subheader("Heat Map")
                    irr_col1.pyplot(f)
                    xtrain,xtest,ytrain,ytest=irrigation_knn.train_test(df)

                    irrigation_knn.fit_model(xtrain,ytrain, self.n_neighbors,self.algorithm,self.leaf_size)
                    acc=irrigation_knn.test_model(xtest,ytest)
                    irr_col2.subheader("Acurracy ="+acc)
                    irr_col1.subheader("Predict Irrigation Status")
                    userinput=irrigation_knn.user_input_smart_irr(irr_col1,irr_col2)
                    prediction=irrigation_knn.predict_input(userinput)
                    main_page.alignh(10,irr_col2)
                    irr_col2.subheader("The predicted irrigation Status is"+ str(prediction))

                with tab_irrigGraph:
                    graphcol1,graphcol2=tab_irrigGraph.columns((5,5))
                    x=df.iloc[:,:5]
                    x.hist()
                    plt.show()
                    graphcol1.header("Features Histogram")
                    graphcol1.pyplot(plt)
                    #fig, ax = plt.subplots()
                    #ax.hist(x, bins=20)
                # tabirisGraphs.pyplot(fig)
                    x.plot()
                    plt.show()
                    graphcol2.header("Features Plot")
                    graphcol2.pyplot(plt)

                    #redvize
                    pd.plotting.radviz(df, 'Irrigation')
                    graphcol1.header("Radviz Plot")
                    graphcol1.pyplot(plt)
                    #Andrews_curves plot
                    pd.plotting.andrews_curves(df, 'Irrigation')
                    graphcol2.header("Andrews Curves Plot")
                    graphcol2.pyplot(plt)
                    #Scatter Matrix plot
                    axes=pd.plotting.scatter_matrix(df, alpha=0.2)
                    graphcol1.header("Scatter Matrix")
                    for ax in axes.flatten():
                        ax.xaxis.label.set_rotation(90)
                        ax.yaxis.label.set_rotation(0)
                        ax.yaxis.label.set_ha('right')
                    graphcol1.pyplot(plt)
                    #feature ploting
                    x=df.iloc[:100,:]
                    y=np.arange(0,100)
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    x['y']=y
                    ax1 = x.plot(kind='scatter', y='CropType', x='y', color=color[0],rot=90)    
                    ax2 = x.plot(kind='scatter', y='CropDays', x='y', color=color[1], ax=ax1)    
                    ax3 = x.plot(kind='scatter', y='SoilMoisture', x='y', color=color[2], ax=ax1)
                    ax4 = x.plot(kind='scatter', y='temperature', x='y', color=color[3],ax=ax1)    
                    ax5 = x.plot(kind='scatter', y='Humidity', x='y', color=color[4],ax=ax1)  

                    graphcol2.header("Feature Scatter Matrix")
                    plt.xlabel("Features")
                    plt.ylabel("Irrigation")
                    graphcol2.pyplot(plt)

 #**************************************#********************************************************           
            elif selected_dataset=='Diabetes':
                tab_diabetesData,tab_diabetesGraphs=st.tabs(("Data","Graphs"))
                with tab_diabetesData:
                    from diabetes import diabetes_knn
                    diabet_classifier=diabetes_knn()
                    df=diabet_classifier.load_data()
                    tab_diabetesData.header("Sample Data")
                    tab_diabetesData.write(df.head(10))
                    tab_diabetesData.header("Data Statistics")
                    tab_diabetesData.write(df.describe())
                    x_train,x_test,y_train,y_test= diabet_classifier.train_test(df)
                    #main_page.alignh(6,diabetescol2)
                    tab_diabetesData.subheader("scaling features to optimize performance")
                    tab_diabetesData.write(x_train)
                    diabetescol1,diabetescol2=tab_diabetesData.columns((6,6))
                    diabetescol2.header("Missing values ")
                    diabetescol2.write(df.isnull().sum())
                    main_page.alignh(3,diabetescol2)
                    diabetescol2.subheader("classes representation")
                    diabetescol2.write( df['Outcome'].value_counts())
                    diabetescol1.subheader("Correlation Heat Map")
                    corr=df.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    with sns.axes_style("white"):
                        f, ax = plt.subplots(figsize=(7, 5))
                        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                    diabetescol1.pyplot(f)
                    diabet_classifier.fit_model(x_train,y_train,self.n_neighbors,self.algorithm,self.leaf_size)
                    accuracy=diabet_classifier.test_model(x_test,y_test)
                # main_page.alignh(5,diabetescol2)
                    diabetescol2.subheader(" Model Accuracy  \t "+ accuracy + "")
                    
                    diabetescol1.subheader('predict of Diabetes')
                    test_inputs= diabet_classifier.user_input_features_diabetes(df,diabetescol1)
                #diabetescol1.subheader('User Input parameters')
                    main_page.alignh(6,diabetescol2)
                    diabetescol2.subheader('The diabetes prdiction is ')
                    diabetescol2.subheader("Diabetes positive" if diabet_classifier.predict_input(test_inputs)==1 else "Diabetest negative")
                
            
                with tab_diabetesGraphs:
                    graphcol1,graphcol2=tab_diabetesGraphs.columns((5,5))
                    x=df.iloc[:,:8]
                    x.hist()
                    plt.show()
                    graphcol1.header("Features Histogram")
                    graphcol1.pyplot(plt)
                    #fig, ax = plt.subplots()
                    #ax.hist(x, bins=20)
                # tabirisGraphs.pyplot(fig)
                    x.plot()
                    plt.show()
                    graphcol2.header("Features plot")
                    graphcol2.pyplot(plt)

                    #redvize
                    pd.plotting.radviz(df, 'Outcome')
                    graphcol1.header("Radviz plot")
                    graphcol1.pyplot(plt)
                    #Andrews_curves plot
                    pd.plotting.andrews_curves(df, 'Outcome')
                    graphcol2.header("Andrews_curves plot")
                    graphcol2.pyplot(plt)
                    #Scatter Matrix plot
                    axes=pd.plotting.scatter_matrix(df, alpha=0.2)
                    graphcol1.header("Scatter Matrix")
                    for ax in axes.flatten():
                        ax.xaxis.label.set_rotation(90)
                        ax.yaxis.label.set_rotation(0)
                        ax.yaxis.label.set_ha('right')
                    graphcol1.pyplot(plt)

                    #feature ploting
                    x=df.iloc[:100,:]
                    y=np.arange(0,100)
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    x['y']=y
                    ax1 = x.plot(kind='scatter', y='Pregnancies', x='y', color=color[0],rot=90)    
                    ax2 = x.plot(kind='scatter', y='Glucose', x='y', color=color[1], ax=ax1)    
                    ax3 = x.plot(kind='scatter', y='BloodPressure', x='y', color=color[2], ax=ax1)
                    ax4 = x.plot(kind='scatter', y='SkinThickness', x='y', color=color[3],ax=ax1)    
                    ax5 = x.plot(kind='scatter', y='Insulin', x='y', color=color[4], ax=ax1)    
                    ax6 = x.plot(kind='scatter', y='Age', x='y', color=color[5], ax=ax1)
                    ax7 = x.plot(kind='scatter', y='BMI', x='y', color=color[6],ax=ax1)    
                    ax8 = x.plot(kind='scatter', y='DiabetesPedigreeFunction', x='y', color=color[7], ax=ax1)   
                    plt.xlabel("Features")
                    plt.ylabel("Diabetes")
                    graphcol2.header("feature Scatter Matrix")
                    graphcol2.pyplot(plt)
 #**************************************#********************************************************           
              
       
 #**************************************#********************************************************           
            elif selected_dataset=='Article classifier':
                articlesData,articlesGraph=st.tabs(("Data","Graphs"))
                with articlesData :   
                    text_col1,text_col2=articlesData.columns((6,6))
                    from Articles_classification import Knn_text
                    articles_classifier=Knn_text()
                    df=articles_classifier.load_articles()
                    articlesData.subheader("Sample Data")
                    articlesData.write(df.sample(10))
                    text_col1,text_col2=articlesData.columns((6,6))
                    text_col1.subheader("Missing values ")
                    text_col1.write(df.isnull().sum())
                    text_col2.subheader("Classes Representation")
                    text_col2.write( df['category'].value_counts())
                    X_train,y_train,x_test,y_test=articles_classifier.train_test(df)
                    articles_classifier.fit_model(X_train,y_train,self.n_neighbors,self.algorithm,self.leaf_size)
                    text_col1.subheader("Accuracy =  " + articles_classifier.test_model(x_test,y_test))
                    text_col1.subheader("predict article  Category ")
                    article_to_predict=text_col2.text_area("Put Your Text Here",height=200)
                    text_col1.write(articles_classifier.predict_input(article_to_predict))
                
    def load_and_viusalize(self):
        
        import numpy as np
        import pandas as pd
        import plotly.io as pio
        import plotly.express as px
        import plotly.offline as py
        import random

        # Create the data
        np.random.seed(20)
        df = pd.DataFrame({'X1': np.random.randint(1, 20, 20),
                        'X2': np.random.randint(1, 20, 20),
                        'Y': np.random.choice(['Class 2', 'Class 1'], size=20)})
        df.loc[len(df)] = [12,6, 'Unknown'] # query point
        df['Distance'] = ((df[['X1', 'X2']] - df.iloc[-1, :2]) ** 2).sum(axis=1) # distances from query point
        df = df.sort_values(by='Distance')
        
        # random item from list
        
        df['Predicted Class'] = np.random.choice(['Class 2', 'Class 1','Unknown'], size=21)

        # Plot with plotly
        color_dict = {"Class 1": "#636EFA", "Class 2": "#EF553B", "Unknown": "#7F7F7F"}
        fig = px.scatter(df, x="X1", y="X2", color='Y', color_discrete_map=color_dict,
                        range_x=[0, 30], range_y=[0, 30],
                        width=650, height=520,title="K Nearst Neighbors ")
        fig.update_traces(marker=dict(size=20,
                                    line=dict(width=1)))

        # Add lines
        shape_dict = {} # create a dictionary
        for k in range(0, len(df)):
            shape_dict[k] = [dict(type="line", xref="x", yref="y",x0=x, y0=y, x1=6, y1=3, layer='below',
                                    line=dict(color="Black", width=2)) for x, y in df.iloc[1:k+1, :2].to_numpy()]
            if k != 0:
                shape_dict[k].append(dict(type="circle", xref="x", yref="y",x0=5.75, y0=2.75, x1=6.25, y1=3.25,
                                            fillcolor=color_dict[df.iloc[k, 4]]))

        # Add dropdown
        fig.update_layout(
        updatemenus=[dict(buttons=[dict(args=[{"shapes": shape_dict[k]}],
                                        label=str(k),
                                        method="relayout") for k in range(0, len(df))],
                            direction="down", showactive=True,
                            x=0.115, xanchor="left", y=1.14, yanchor="top")])

        # Add dropdown label
        fig.update_layout(annotations=[dict(text="k = ",
                                        x=0, xref="paper", y=1.13, yref="paper",
                                        align="left", showarrow=False)],
                        font=dict(size=20))
        return fig

    def generate_data(self,k,n_samples,cluster_std):

        import matplotlib.pyplot as plt

        from sklearn.datasets import make_blobs

        X, y = make_blobs(n_samples = n_samples, n_features = 2, centers = 4,cluster_std = cluster_std, random_state = 4)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
        knn = KNeighborsClassifier(n_neighbors = k)

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        fig=plt.figure(figsize = (10,5))
        plt.subplot(1,1,1)
        plt.scatter(X[:,0], X[:,1], c=y, marker= '*',s=100,edgecolors='blue')
        plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, marker= '*', s=100,edgecolors='red')

        plt.title("KNN prediction - original Data in blue predected in red ", fontsize=20)

        return fig
        
