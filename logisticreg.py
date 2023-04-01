from Page_layout import main_page
import streamlit as st
import streamlit.components.v1 as components
from regression12 import Reganim
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np





class supervised_logistic(object):
    def __init__(self):

 
        global train_test_param,learning_rate_parameter,No_of_iteration_parameter,Frame_interval_parameter
   
   
    def logitic_tut(self):      

        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
            Try_col1,Try_col2=Try_algorithm.columns((5,5))

            self. train_test_param=Try_col1.slider("Train-Test Split value",.5,.9,.1)
            self.learning_rate_parameter=Try_col1.slider("Learning Rate",min_value=.001,max_value=.1,step=0.001)
            Go_model_BTN=Try_col1.button("Go")

            #alignh(1,col2)
            #col2.markdown("#####")
            self.No_of_iteration_parameter=Try_col2.slider("Iterations",1,150,1)
            self.Frame_interval_parameter=Try_col2.slider("Interval",100,500,100)
            Generate_dataBTN =Try_col2.button("Generate New Data")

        
            if Go_model_BTN:
                #st.write(self.learning_rate_parameter,self.No_of_iteration_parameter,self.Frame_interval_parameter,self.train_test_param)
                graph= self.animat('data/logreg.txt',self.learning_rate_parameter,self.No_of_iteration_parameter,self.Frame_interval_parameter,self.train_test_param)
                components.html(graph.to_jshtml(), height=600,width=600,scrolling=True)
            
            if Generate_dataBTN:
                main_page.gen_new_data()
                graph= self.animat('data/new.txt',self.learning_rate_parameter,self.No_of_iteration_parameter,self.Frame_interval_parameter,self.train_test_param)
        
                components.html(graph.to_jshtml(), height=600,width=600,scrolling=True)
        with How_it_work:
            how_col1,how_col2=How_it_work.columns((8,1))
            how_col1.image("media/hirarchy.png")
            how_col1.subheader("Data piplining")
            how_col1.image("media/propagation.png")
            how_col1.subheader("Feature propagation")
            how_col1.image("media/themath.png")

    
        #Animation function

#
    def animat(self,file,lrate,iter,interv,trts):
        logreg=Reganim(file,lr=lrate,iterations=iter,intervals=interv,tr_ts=trts)
        anim=FuncAnimation(logreg.fig,logreg.animation, frames=iter, interval=interv, repeat_delay=500)
        #components.html(logreg.anim.to_jshtml(), height=600,width=600,scrolling=True)
        return anim
    
   

    

    def user_input_features_diabetes(self,df,colname1):
        Pregnancies = colname1.slider('Pregnancies', 0, 17, 1)/17
        Glucose = colname1.slider('Glucose', 0, 200, 1)
        Glucose=(Glucose - df['Glucose'].mean())/df['Glucose'].std()
        BloodPressure = colname1.slider('BloodPressure', 0, 150, 1)
        BloodPressure=(BloodPressure - df['BloodPressure'].mean())/df['BloodPressure'].std()
        SkinThickness = colname1.slider('SkinThickness',0,100, 1)
        SkinThickness=(SkinThickness - df['SkinThickness'].mean())/df['SkinThickness'].std()
        Insulin = colname1.slider('Insulin', 0, 900, 1)
        SkinThickness=(Insulin - df['Insulin'].mean())/df['Insulin'].std()
        BMI = colname1.slider('BMI', 0.0, 80.0, 0.01)
        BMI=(BMI - df['BMI'].mean())/df['BMI'].std()

        DiabetesPedigreeFunction = colname1.slider('DiabetesPedigreeFunction', 0.005, 3.0, 0.001)
        DiabetesPedigreeFunction=(DiabetesPedigreeFunction - df['DiabetesPedigreeFunction'].mean())/df['DiabetesPedigreeFunction'].std()

        Age = colname1.slider('Age', 20, 85, 1)
        Age=(BMI - df['Age'].mean())/df['Age'].std()


        data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age      
            }
        features = pd.DataFrame(data, index=[0])
        return features

    def scaledata(self,df):
        mms = MinMaxScaler() # Normalization
        ss = StandardScaler() # Standardization
        scaled_df=df.copy()
        scaled_df['Pregnancies'] = mms.fit_transform(df[['Pregnancies']])
        scaled_df['Insulin'] = ss.fit_transform(df[['Insulin']])
        scaled_df['DiabetesPedigreeFunction'] = ss.fit_transform(df[['DiabetesPedigreeFunction']])
        scaled_df['Age'] = ss.fit_transform(df[['Age']])
        scaled_df['BloodPressure'] = ss.fit_transform(df[['BloodPressure']])
        scaled_df['SkinThickness'] = ss.fit_transform(df[['SkinThickness']])
        scaled_df['Glucose'] = ss.fit_transform(df[['Glucose']])
        scaled_df['BMI'] = ss.fit_transform(df[['BMI']])
        
        return scaled_df
###*********************************************************************************************************
###*********************************************************************************************************
   
   
    def logiticaction(self):   
       
        

        col1, col2 = st.columns((3,3))
        col1.subheader("Select Dataset")

        selected_dataset=col1.selectbox("Datasets",options=['Smart Irrigation','Iris','Smoke detection',
                                                          'Diabetes','Diseses','Image classifier',
                                                          'Article classifier','Digit recognition'])
        col2.subheader("Set Logistic Regression  Parameter")
        fulllist=['l1', 'l2', 'elasticnet', 'none']
        clfoption1=[ 'l2']
        clfoption2=[ 'l2','none']
        Solver=col2.selectbox("Solver",options=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        #col2.markdown('###')
        if Solver=='newton-cg':
            Clf=col2.selectbox("Penalty",options=clfoption2)
        elif Solver=='lbfgs':
            Clf=col2.selectbox("Penalty",options=clfoption2)
        elif Solver=='liblinear':
            Clf=col2.selectbox("Penalty",options=clfoption1)
        elif Solver=='sag':
            Clf=col2.selectbox("Penalty",options=clfoption2)
        elif Solver=='saga':
            Clf=col2.selectbox("Penalty",options=fulllist)
        main_page.alignh(2,col1)
        Go =col1.button("Go")

        if Go :
            if selected_dataset=='Smart Irrigation':
                tab_irrigData,tab_irrigGraph=st.tabs(("Data","Graphs"))
                with tab_irrigData :
                    from smart_irrigation import logistic_irrg
                    logistic_irrg=logistic_irrg()
                    df=logistic_irrg.load_data()
                
                    tab_irrigData.subheader("Sample Data")
                    tab_irrigData.write(df.sample(10))
                    tab_irrigData.subheader("Irrigation Statistics")
                    tab_irrigData.write(df.describe())
                    irr_col1,irr_col2=tab_irrigData.columns((6,6))
                    corr=df.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    with sns.axes_style("dark"):
                        f, ax = plt.subplots(figsize=(6, 6))
                        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                    irr_col1.pyplot(f)
                    irr_col2.subheader("Missing values ")
                    irr_col2.write(df.isnull().sum())
                    irr_col2.subheader("Classes representation")
                    irr_col2.write( df['Irrigation'].value_counts())
                    
                    xtrain,xtest,ytrain,ytest=logistic_irrg.train_test(df)
                    #irrigation_svm.fit_model(xtrain,xtest,ytrain,ytest)
                    accuracy=logistic_irrg.test_model(xtest,ytest)
                    irr_col2.subheader(" Model Accuracy  \t "+ str(accuracy)+ "")
                    irr_col1.subheader("Predict Irrigation State")
                    user_input=logistic_irrg.user_input_smart_irr(irr_col1,irr_col2)
                    prediction=logistic_irrg.predict_input(user_input)
                    irr_col1.subheader("predicted irrigation state is   "+ str(prediction))



                with tab_irrigGraph:
                    graphcol1,graphcol2=tab_irrigGraph.columns((5,5))
                    x=df.iloc[:,:5]
                    x.hist( xlabelsize=5)
                    plt.show()
                    graphcol1.header("Features Histogram")
                    graphcol1.pyplot(plt)
                    x.plot()
                    plt.show()
                    graphcol2.header("Features plot")
                    graphcol2.pyplot(plt)

                    #redvize
                    pd.plotting.radviz(df, 'Irrigation')
                    graphcol1.header("Radviz plot")
                    graphcol1.pyplot(plt)
                    #Andrews_curves plot
                    pd.plotting.andrews_curves(df, 'Irrigation')
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
                    ax1 = x.plot(kind='scatter', y='CropType', x='y', color=color[0],rot=90)    
                    ax2 = x.plot(kind='scatter', y='CropDays', x='y', color=color[1], ax=ax1)    
                    ax3 = x.plot(kind='scatter', y='SoilMoisture', x='y', color=color[2], ax=ax1)
                    ax4 = x.plot(kind='scatter', y='temperature', x='y', color=color[3],ax=ax1)    
                    ax5 = x.plot(kind='scatter', y='Humidity', x='y', color=color[4],ax=ax1)    
                    plt.xlabel("Features")
                    plt.ylabel("Irrigation")
                    graphcol2.header("feature Scatter Matrix")
                    graphcol2.pyplot(plt)
                    #CropType,CropDays,SoilMoisture,temperature,Humidity,Irrigation

        
       
                
            elif selected_dataset=='Iris':
                #tabiris=st.tabs(("Iris flower Classification"))
                tabirisData,tabirisGraphs=st.tabs(("Data","Graphs"))
                with tabirisData:
                    iriscol1,iriscol2=tabirisData.columns((6,6))
                    from iris import logistic_iris
                    iris_classifier=logistic_iris()

                    df = iris_classifier.load_data()

                    iriscol1.subheader("Sample Data")
                    iriscol1.write(df.sample(10))
                    iriscol2.subheader("Data Statistics")
                    iriscol2.write(df.describe())
                    iriscol2.header("Classes Representation")
                    iriscol2.write( df['Flower_type'].value_counts())
                # main_page.alignh(2,iriscol2)
                    iriscol2.header("Missing values ")
                    iriscol2.write(df.isnull().sum())

                    iriscol1.header("Correlation Heat Map")
                    corr=df.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    with sns.axes_style("white"):
                        f, ax = plt.subplots(figsize=(7, 5))
                        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                    iriscol1.pyplot(f)
                    # feature and Target Separation
                    x_train,x_test,y_train,y_test=iris_classifier.train_test(df)
                    iris_classifier.fit_model(x_train,y_train,Clf,Solver)
                    accuracy=iris_classifier.test_model(x_test,y_test)
                    iriscol2.subheader(" Model Accuracy  \t "+ str(accuracy)+ "")

                    test_inputs=iris_classifier.user_input_features_iris(iriscol1,iriscol2)
                    image_type= str(iris_classifier.predict_input(test_inputs))
                    characters_to_remove = "'[]'"
                    for character in characters_to_remove:
                        image_type = image_type.replace(character, "")
                    
                    iriscol2.subheader("The predicted flower is")
                    iriscol2.image(str("media/"+image_type + ".png"))
                    iriscol2.subheader(image_type)
            
            
                with tabirisGraphs:
                    graphcol1,graphcol2=tabirisGraphs.columns((5,5))
                    df.hist(xlabelsize=5)
                    plt.show()
                    graphcol1.header("Features Histogram")
                    graphcol1.pyplot(plt)
                    #fig, ax = plt.subplots()
                    #ax.hist(x, bins=20)
                # tabirisGraphs.pyplot(fig)
                    df.plot()
                    plt.show()
                    graphcol2.header("Features plot")
                    graphcol2.pyplot(plt)

                    #redvize
                    pd.plotting.radviz(df, 'Flower_type')
                    graphcol1.header("Radviz plot")
                    graphcol1.pyplot(plt)
                    #Andrews_curves plot
                    pd.plotting.andrews_curves(df, 'Flower_type')
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
                    ax1 = x.plot(kind='scatter', y='sepal_length', x='y', color=color[0],rot=90)    
                    ax2 = x.plot(kind='scatter', y='sepal_width', x='y', color=color[1], ax=ax1)    
                    ax3 = x.plot(kind='scatter', y='sepal_width', x='y', color=color[2], ax=ax1)
                    ax4 = x.plot(kind='scatter', y='petal_width', x='y', color=color[3],ax=ax1)    
                    plt.xlabel("Features")
                    plt.ylabel("Flower Type")
                    graphcol2.header("feature Scatter Matrix")
                    graphcol2.pyplot(plt)


            elif selected_dataset=='Smoke detection':
                smoke=st.tabs(("Smoke detection"))
                pass

            elif selected_dataset=='Diabetes':
                tabdiabetData,tabdiabetGraph=st.tabs(("Data","Graphs"))
                with tabdiabetData:
                    diabetescol1,diabetescol2=tabdiabetData.columns((4,4))
                    df = pd.read_csv('data/diabetes.csv')
                    diabetescol1.header("Sample Data")
                    diabetescol1.write(df.head(10))
                    diabetescol1.header("Data Statistics")
                    diabetescol1.write(df.describe())
                    diabetescol2.header("Missing values ")
                    diabetescol2.write(df.isnull().sum())
                    main_page.alignh(3,diabetescol2)
                    diabetescol2.subheader(" Equal class representation")
                    diabetescol2.write( df['Outcome'].value_counts())
                    diabetescol1.subheader("Correlation Heat Map")
                    corr=df.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    with sns.axes_style("white"):
                        f, ax = plt.subplots(figsize=(7, 5))
                        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                    diabetescol1.pyplot(f)
                    #scale Features
                    df1= self.scaledata(df)
                    x=df1.iloc[:,:8]
                    y=df1.iloc[:,8]
                    main_page.alignh(6,diabetescol2)
                    diabetescol2.subheader("scaling features to optimize performance")
                    diabetescol2.write(x)
                    # feature and Target Separation
                    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
                    model=LogisticRegression(penalty=Clf,solver=Solver)
                    model.fit(x_train,y_train)
                    y_pred=model.predict(x_test)
                    diabetescol1.subheader("Confusion Matrix")
                    diabetescol1.write(confusion_matrix(y_test,y_pred))
                    accuracy=accuracy_score(y_test,y_pred)*100
                    main_page.alignh(5,diabetescol2)
                    diabetescol2.subheader(" Model Accuracy  \t "+ str(accuracy)+ "")
                    main_page.alignh(6,diabetescol2)
                    diabetescol1.subheader('predict of Diabetes')
                    test_inputs= self.user_input_features_diabetes(df,diabetescol1)
                #diabetescol1.subheader('User Input parameters')
                    diabetescol2.subheader('The diabetes prdiction is ')
                    diabetescol2.subheader("Diabetes positive" if model.predict(test_inputs)==1 else "Diabetest negative")
            
                with tabdiabetGraph:
                    graphcol1,graphcol2=tabdiabetGraph.columns((5,5))
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

                
                pass
            elif selected_dataset=='Diseses':
                tabdiseses=st.tabs(("Diseses Prediction"))
                pass
                
            elif selected_dataset=='Image classifier':
                images=st.tabs(('Image classifier'))
                pass
                
                pass
            elif selected_dataset=='Article classifier':
                Article=st.tabs(('Article classifier'))
                pass
            elif selected_dataset=='Digit recognition':
                Digit=st.tabs(('Digit recognition'))
                pass


     