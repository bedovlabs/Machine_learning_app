import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regex import X
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from dtreeviz.trees import *
import streamlit as st
from Page_layout import main_page
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


class Supervised_decession_tree(object):
    
    
    def dec_tree_tut(self):      

        #Animation functiondef logitic_tut(self):      

        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
            Try_col1,Try_col2=Try_algorithm.columns((5,5))
            self.criterion=Try_col1.selectbox("Criterion",options=['gini', 'entropy', 'log_loss'])
            self.max_depth=Try_col1.slider("Max Depth",1,10,1)
            
            Go_model_BTN=Try_col1.button("Go")
            self.max_features=Try_col2.selectbox("Max Features",options=['None', 'auto', 'sqrt', 'log2'])
            self.min_samples_split=Try_col2.slider("Minimum Samples Split",2,10,1)
            
            Generate_dataBTN =Try_col2.button("Generate New Data")

        
            if Go_model_BTN:
                df=load_iris()
                fig1,fig2=self.load_data_and_visualize(df)
                Try_algorithm.pyplot(fig1)
                Try_algorithm.plotly_chart(fig2,use_container_width=True)    
              
            
            if Generate_dataBTN:
                df=main_page.gen_new_data()
                fig1,fig2=self.load_data_and_visualize(df)
                Try_algorithm.pyplot(fig1)
                Try_algorithm.plotly_chart(fig2,use_container_width=True)    
                
                
        with How_it_work:
            how_col1,how_col2=How_it_work.columns((8,1))
            how_col1.image("media/decession tree1.png")
            how_col1.subheader("Data piplining")
            how_col1.image("media/decession tree2.png")
            how_col1.subheader("Information Gain")
            how_col1.image("media/decession tree3.png")
            how_col1.subheader("Attribute selection Measure propagation")
            how_col1.image("media/decession tree4.png")

    
        #Animation function


    def dec_tree_inaction(self):      


        col1, col2 = st.columns((3,4))
        col1.subheader("Select Dataset")
        selected_dataset=col1.selectbox("Datasets",options=['Smart Irrigation','Iris','Smoke detection',
                                                          'Diabetes','Diseses','Image classifier',
                                                          'Article classifier','Digit recognition'])
        col2.subheader("Set Decession Tree Parameter")
        self.criterion=col2.selectbox("Criterion",options=['gini', 'entropy', 'log_loss'])
        self.max_features=col2.selectbox("Max Features",options=[ 'auto', 'sqrt', 'log2'])
        self.max_depth=col2.slider("Max Depth",1,10,1)
        self.min_samples_split=col2.slider("Minimum Samples Split",3,10,1)
        if self.min_samples_split<2:
            self.min_samples_split=2

        main_page.alignh(6,col1)
        Go =col1.button("Go")

        if Go :
            if selected_dataset=='Iris':
                tabirisData,tabirisGraphs=st.tabs(("Data","Graphs"))
                with tabirisData:
                    iriscol1,iriscol2=tabirisData.columns((6,6))
                    df = pd.read_csv('data/iris.csv')
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
                    x=df.iloc[:,:-1]
                    y=df.iloc[:,-1]
                    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
                    model = DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth,max_features=self.max_features ,
                                                    min_samples_split=self.min_samples_split ).fit(x_train,y_train)
                    y_pred=model.predict(x_test)
                #main_page.alignh(7,iriscol3)
                    
                    iriscol2.header("Confusion Matrix")
                    #main_page.alignh(3,iriscol3)
                    iriscol2.write(confusion_matrix(y_test,y_pred))
                    accuracy=accuracy_score(y_test,y_pred)*100
                    iriscol2.header(" Model Accuracy  \t "+ str(accuracy)+ "")
                    test_inputs=self.user_input_features_iris(iriscol1,iriscol2)
                    image_type= str(model.predict(test_inputs))
                    characters_to_remove = "'[]'"
                    for character in characters_to_remove:
                        image_type = image_type.replace(character, "")
                    
                    iriscol2.subheader("The predicted flower is")
                    iriscol2.image(str("media/"+image_type + ".png"))
                    iriscol2.subheader(image_type)
            
            
                with tabirisGraphs:
                    graphcol1,graphcol2=tabirisGraphs.columns((5,5))
                    x.hist(xlabelsize=5)
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
        
 #**************************************#********************************************************           

            elif selected_dataset=='Smart Irrigation':
                tab_irrigData,tab_irrigGraph=st.tabs(("Data","Graphs"))
                with tab_irrigData :
                    from smart_irrigation import dec_tree_irr
                    irrigation_dectree=dec_tree_irr()
                    df=irrigation_dectree.load_data()
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
                    xtrain,xtest,ytrain,ytest=irrigation_dectree.train_test(df)

                    irrigation_dectree.fit_model(xtrain,ytrain,self.criterion,self.max_depth,self.max_features,self.min_samples_split)
                    acc=irrigation_dectree.test_model(xtest,ytest)
                    irr_col2.subheader("Acurracy ="+acc)
                    irr_col1.subheader("Predict Irrigation Status")
                    userinput=irrigation_dectree.user_input_smart_irr(irr_col1,irr_col2)
                    prediction=irrigation_dectree.predict_input(userinput)
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
            elif selected_dataset=='Image classifier':
                tab_imagesData,tab_imagesGraph=st.tabs(("Data","Graphs"))

                with tab_imagesData :   

                    from PIL import Image
                    from skimage.transform import resize
                    from skimage.io import imread
                    from image_recognition import image_dectree
                    import os
                
                    tab_imagesData.subheader("Sample Data ")
                    img_col1,img_col2=tab_imagesData.columns((6,6))
                    cat_dog_classifier=image_dectree()
                    x,y,df=cat_dog_classifier.load_data()
                    datadir='media/animals/' 
                    for i in ['Cats','Dogs']:
                        path=os.path.join(datadir,i)
                        for img,j in zip(os.listdir(path),range(1,4)):
                            if i=='Cats':
                                image = Image.open(str(path+"/"+img))
                                new_image = image.resize((200, 200))
                                img_col1.image(new_image,width=150)
                            else:
                                image = Image.open(str(path+"/"+img))
                                new_image = image.resize((200, 200))
                                img_col2.image(new_image, width=150)

                    img_col1.subheader("Classes Representation")
                    img_col1.write(df['Target'].value_counts())
                    cat_dog_classifier.fit_model(x,y,self.criterion,self.max_depth,self.max_features,self.min_samples_split)
                    img_col1.subheader("Predict Animal Type")
                    
                    bottom_image = st.file_uploader('', type='jpg')
                    if bottom_image is not None:
                        image = Image.open(bottom_image)
                        new_image = image.resize((150, 150))
                        st.image(new_image)
                        img_array = np.array(image)
                        img_resized=resize(img_array,(250,270,3))
                        img=img_resized.transpose(2,0,1).reshape(3,-1)
                        prediction=cat_dog_classifier.predict_input(img)
                        st.write("Dog" if np.bincount(prediction).argmax()==1 else "Cat")



 #**************************************#********************************************************
          



        elif selected_dataset=='Diabetes':      
            tab_diabetesData,tab_diabetesGraphs=st.tabs(("Data","Graphs"))
            with tab_diabetesData:
                from diabetes import diabetes_dctree
                diabet_classifier=diabetes_dctree()
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
                diabet_classifier.fit_model(x_train,y_train,self.criterion,self.max_depth,self.max_features,self.min_samples_split)
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



    def load_data_and_visualize(self,df):    
        data = df
        if isinstance(data, pd.DataFrame):
            X = data.iloc[:,:-1]
            y= data.iloc[:,-1]
            feature_names = X.columns
            #feature_names = data['feature_names']
            model = DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth,max_features=self.max_features ,
                                            min_samples_split=self.min_samples_split ).fit(X,y)
            fig1= plt.figure(figsize=(12, 4), dpi=200)

            plot_tree(model, feature_names=X.columns, filled=True)
            #fig1.show()
            viz = dtreeviz(model,
                        X,
                        y,
                        target_name='Flower name',  # this name will be displayed at the leaf node
                        feature_names=X.columns,
                        title="Iris data",
                        fontname="Arial",
                        title_fontsize=16,
                        colors = {"title":"purple"}
                        )
            viz


        else:
            X, y = data['data'], data['target']
            feature_names = data['feature_names']
            #feature_names = data['feature_names']
            model = DecisionTreeClassifier(criterion='entropy').fit(X,y)
            fig1= plt.figure(figsize=(12, 4), dpi=200)
            plot_tree(model, feature_names=feature_names, filled=True)
            #fig1.show()
            viz = dtreeviz(model,
                        X,
                        y,
                        target_name='Flower name',  # this name will be displayed at the leaf node
                        feature_names=data.feature_names,
                        title="Iris data",
                        fontname="Arial",
                        title_fontsize=16,
                        colors = {"title":"purple"}
                        )
            viz
        import plotly.graph_objects as go
        labels = [''] * model.tree_.node_count
        parents = [''] * model.tree_.node_count
        labels[0] = 'root'
        for i, (f, t, l, r) in enumerate(zip(
            model.tree_.feature,
            model.tree_.threshold,
            model.tree_.children_left,
            model.tree_.children_right,
        )):
            if l != r:
                labels[l] = f'{feature_names[f]} <= {t:g}'
                labels[r] = f'{feature_names[f]} > {t:g}'
                parents[l] = parents[r] = labels[i]
        fig2 = go.Figure(go.Treemap(
            branchvalues='total',
            labels=labels,
            parents=parents,
            values=model.tree_.n_node_samples,
            textinfo='label+value+percent root',
            marker=dict(colors=model.tree_.impurity),
            customdata=list(map(str, model.tree_.value)),
            hovertemplate='''
        <b>%{label}</b><br>
        impurity: %{color}<br>
        samples: %{value} (%{percentRoot:%.2f})<br>
        value: %{customdata}'''
        ))
       # fig2.show()
        return fig1,fig2

    def user_input_features_iris(self,colname1,colname2):
        main_page.alignh(3,colname1)
        colname1.header("Prediction of flower type")
        sepal_length = colname1.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = colname1.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = colname1.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = colname1.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features