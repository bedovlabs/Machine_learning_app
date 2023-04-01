

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.close("all")

from sklearn.datasets import load_iris
import numpy as np
from sklearn.svm import SVC
from celluloid import Camera
import streamlit.components.v1 as components
from Page_layout import main_page
import streamlit as st
import seaborn as sns


class svm(object):
    def __init__(self,Kernel,C,Gammas,df):
        self.kernel=Kernel
        self.C=C
        self.Gammas=Gammas
        df=df
         #generate an SVM model
        features = ["sepal width (cm)", "petal length (cm)"]
        flowers = [0,1]
        target_names=['Iris-virginica','Iris-setosa']
        SVM_model1 = SVC(kernel=self.kernel,C=self.C,gamma=self.Gammas)

        
       
        init_rows = 2
        X = df.iloc[:init_rows,:2]
        y = df["Targets"][:init_rows]
        SVM_model1.fit(X,y)

        classes = df.Targets.value_counts().index
        scores1 = [SVM_model1.score(X,y)]
      
        fig = plt.figure(figsize=(8,5))
        self.camera = Camera(fig)
        ax1 = fig.add_subplot(111)
       
        for i in range(init_rows+1,df.shape[0],3):
            
      
            X = df.iloc[:i,:2]
            y = df["Targets"][:i]
            SVM_model1.fit(X,y)
           # SVM_model2.fit(X,y)
            
            f1 = df[features[0]][:i]
            f2 = df[features[1]][:i]
            t = df["Targets"][:i]
            
            f11 = ax1.scatter(f1[t==classes[0]],f2[t==classes[0]],c="cornflowerblue",marker="o")
            f12 = ax1.scatter(f1[t==classes[1]],f2[t==classes[1]],c="sandybrown",marker="^")

           
            
            #calculate the model accuracy
            scores1.append(SVM_model1.score(X,y))
            #scores2.append(SVM_model2.score(X,y))
            
           
            #prepare data for decision boundary plotting
            x_min = X.iloc[:,0].min()
            x_max = X.iloc[:,0].max()
            y_min = X.iloc[:,1].min()
            y_max = X.iloc[:,1].max()
            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z1 = SVM_model1.decision_function(np.c_[XX.ravel(), YY.ravel()])
            Z1 = Z1.reshape(XX.shape)
            #Z2 = SVM_model2.decision_function(np.c_[XX.ravel(), YY.ravel()])
           # Z2 = Z2.reshape(XX.shape)

            #plot the decision boundary
            ax1.contour(XX, YY, Z1, colors=['darkgrey','dimgrey','darkgrey'],
                        linestyles=[':', '--', ':'], levels=[-.5, 0, .5])
       
            ax1.text(0.25, 1.03, f"SVC {self.kernel}  Training Accuracy: %.2f"%(SVM_model1.score(X,y)), 
                    fontweight="bold", transform=ax1.transAxes)
            ax1.set_xlim([-3,3])
            ax1.set_ylim([-3,3])
            
         
            ax1.set_xlabel(features[0])
            ax1.set_ylabel(features[1])
       
                
            ax1.legend([f11,f12],target_names,fontsize=9)
    
        
            #take a snapshot of the figure
            self.camera.snap()
        #animation=self.anim()
        return 
    def anim(self):#create animation
        anim =self.camera.animate()
     
        return anim
# %%

class supervised_svm():
    def __init__(self):
        global timer


    def svm_tut(self):
        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
            kernel = Try_algorithm.selectbox("Kernel",options=["linear","rbf","poly"])
            C = Try_algorithm.selectbox("C",options=[0.1, 1, 10, 100, 1000])
            gammas= Try_algorithm.selectbox("Gammas",options=[0.1, 1, 10, 100])
            Try_col1,Try_col2=Try_algorithm.columns((5,5))
            
            Go_model_BTN=Try_col1.button("Go")

            Generate_dataBTN =Try_col2.button("Generate New Data")
        
            if Go_model_BTN:
                from sklearn.datasets import load_iris
                from sklearn import preprocessing

                iris = load_iris()

        #features to use
                features = ["sepal width (cm)", "petal length (cm)"]

                #only versicolor and virginica are the targets
                flowers = [1,2]
                target_cond = (iris.target == flowers[0]) | (iris.target == flowers[1])

                #construct a dataframe with the conditions
                df_features = pd.DataFrame(preprocessing.scale(iris.data[target_cond,:]),
                                        columns = iris.feature_names)
                df_features = df_features[features]
                df_targets = pd.DataFrame(iris.target[target_cond],columns=["Targets"])
                df = pd.concat([df_features,df_targets],axis=1)

                #shuffle the dataset
                df = df.reindex(np.random.RandomState(seed=2).permutation(df.index))
                df = df.reset_index(drop=True)
            #%% train SVM dynamically by adding new data in the loop
                svm1=svm(kernel,float(C),float(gammas),df)
                svmanim=svm1.anim()
                components.html(svmanim.to_jshtml(), height=600,width=1000,scrolling=True)
              
            if Generate_dataBTN:
                df=main_page.gen_new_data()
                cols=["sepal width (cm)", "petal length (cm)","Targets"]
                df.columns=cols
                df = df.reindex(np.random.RandomState(seed=2).permutation(df.index))
                df = df.reset_index(drop=True)
                
                svm1=svm(kernel,float(C),float(gammas),df)
                svmanim=svm1.anim()
                components.html(svmanim.to_jshtml(), height=600,width=1000,scrolling=True)



                
                df.columns=cols
        with How_it_work:
            how_col1,how_col2=How_it_work.columns((8,1))
            how_col1.subheader("Support vector Machine Component")
            how_col1.image("media/svm2.png")
            how_col1.subheader("Support vector Machine General Model structure")
            how_col1.image("media/svm general model.png")
            how_col1.subheader("Support vector Machine Data piplining")
            how_col1.image("media/svmboundry.png")



#****************#***#****************#***#****************#***#****************#***#****************#***
#****************#***#****************#***#****************#***#****************#***#****************#***

    def svm_inaction(self):
        col1,col2=st.columns((5,5))
        col1.subheader("Select Dataset")

        selected_dataset=col1.selectbox("Datasets",options=['Smart Irrigation','Iris','Smoke detection',
                                                          'Diabetes','Diseses','Image classifier',
                                                          'Article classifier','Digit recognition'])
        
        col2.subheader("Set Support Vector  Parameter")
       
        naiev_algorithm=col2.selectbox("Algorithm",options=['GaussianNB', 'MultinomialNB', 'ComplementNB','BernoulliNB'])

        Alpha=col2.slider("Alpha",0.1,1.0,0.1)

        main_page.alignh(2,col1)
        Go =col1.button("Go")
       
        if Go :
            if selected_dataset=='Smart Irrigation':
                import numpy as np

                tab_irrigData,tab_irrigGraph=st.tabs(("Data","Graphs"))

                with tab_irrigData :
                    from smart_irrigation import svm_irr
                    irrigation_svm=svm_irr()
                    df=irrigation_svm.load_data()
                
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
                    irr_col1.subheader("Heat Map")
                    irr_col1.pyplot(f)
                    irr_col2.subheader("Missing Values ")
                    irr_col2.write(df.isnull().sum())
                    irr_col2.subheader(" Classes Representation")
                    irr_col2.write( df['Irrigation'].value_counts())
                
                    xtrain,xtest,ytrain,ytest=irrigation_svm.train_test(df)
                    #irrigation_svm.fit_model(xtrain,xtest,ytrain,ytest)
                    accuracy=irrigation_svm.test_model(xtest,ytest)
                    irr_col2.subheader(" Model Accuracy  \t "+ str(accuracy)+ "")
                    irr_col1.subheader("Predict Irrigation State")
                    user_input=irrigation_svm.user_input_smart_irr(irr_col1,irr_col2)
                    prediction=irrigation_svm.predict_input(user_input)
                    main_page.alignh(10,irr_col2)
                    irr_col2.subheader("   predicted irrigation state is   "+ str(prediction))


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
                    #CropType,CropDays,SoilMoisture,temperature,Humidity,Irriga
#****************#***#****************#***#****************#***#****************#***#****************#***

            elif selected_dataset=='Diseses':
                   
                tab_DiseaseData,tab_DiseaseGraph=st.tabs(("Data","Graphs"))

                with tab_DiseaseData:
                    from Diseses import svm_disese
                    Disease_classifier=svm_disese()
                    df,X,y,max_feat=Disease_classifier.load_data()
                    tab_DiseaseData.subheader("Sample Data")
                    tab_DiseaseData.write(df.sample(10))
                    heart_col1,heart_col2=tab_DiseaseData.columns((6,6))
                    heart_col2.subheader("Missing Values ")
                    heart_col2.write(df.isnull().sum())
                    heart_col1.subheader(" Classes Representation")
                    heart_col1.write( df['Disease'].value_counts())
                
                    xtrain,xtest,ytrain,ytest=Disease_classifier.train_test(X,y)
                    
                    
                    Disease_classifier.fit_model(xtrain,ytrain)
                    accuracy=Disease_classifier.test_model(xtest,ytest)
                    heart_col2.subheader(" Model Accuracy  \t "+ str(accuracy)+ "")
                    heart_col1.subheader("Predict Disease")
                            
                    xs=heart_col1.multiselect("select Symptoms",options= max_feat)
                    prediction = Disease_classifier.predict_input(xs)
                    heart_col2.subheader("The predicted Disease is "+ prediction)
   
                with tab_DiseaseGraph:
                 
                    df1=Disease_classifier.visualize_word_freq(df['symptoms'],50)
                    fig,ax=plt.subplots()
                    ax=df1.plot(kind='bar',title="TITLE")
                    st.write(df1)
                    st.pyplot(fig)
                    
#****************#***#****************#***#****************#***#****************#***#****************#***

            elif selected_dataset=='Article classifier':
                tab_textData,tab_textGraphs=st.tabs(("Data","Graphs"))
     
                with tab_textData:
                    
                    from Articles_classification import svc_text
                    text_classifioer=svc_text()    
                    df=text_classifioer.load_articles()
                    tab_textData.subheader("Sample Data")
                    tab_textData.write(df.sample(10))
                    text_col1,text_col2=tab_textData.columns((6,6))
                    text_col1.subheader("Missing values ")
                    text_col1.write(df.isnull().sum())
                    text_col2.subheader(" Equal Class representation")
                    text_col2.write( df['category'].value_counts())
                    text_col1.subheader("predict article  Category ")
                    article_to_predict=text_col2.text_area("Put Your Text Here",height=200)
                    text_col1.write(text_classifioer.predict_input(article_to_predict))
            
                with tab_textGraphs:
                    graphcol1,graphcol2=tab_textGraphs.columns((6,6))    
                    fig, ax = plt.subplots()  
                    ax = df['category'].value_counts().plot(kind='bar',
                                            figsize=(6,4),
                                            title="Ctegories distribution")
                    ax.set_xlabel("Category Names")
                    ax.set_ylabel("Frequency")           
                
                    graphcol1.pyplot(fig)


                    count = df['body'].str.split().str.len()
                    count.index = count.index.astype(str) + ' words:'
                    count.sort_index(inplace=True)

                    fig, ax = plt.subplots()  
                    ax = count.describe().plot(kind='line',
                                            figsize=(6,4),
                                            title="Article statistics")
                    ax.set_xlabel("Metric")
                    ax.set_ylabel("No of words")           

                    graphcol2.pyplot(fig)

#****************#***#****************#***#****************#***#****************#***#****************#***
            elif selected_dataset=='Image classifier':
                
                tab_imagesData,tab_imagesGraph=st.tabs(("Data","Graphs"))
    
                with tab_imagesData :   

                    from PIL import Image
                    from skimage.transform import resize
                    from skimage.io import imread
                    from image_recognition import svm_image
                    import os
                
                    tab_imagesData.subheader("Sample Data ")
                    img_col1,img_col2=tab_imagesData.columns((6,6))
                    cat_dog_classifier=svm_image()
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
                
#****************#***#****************#***#****************#***#****************#***#****************#***
            elif selected_dataset=='Iris':
                import numpy as np
                tabirisData,tabirisGraphs=st.tabs(("Data","Graphs"))
                with tabirisData:
                    iriscol1,iriscol2=tabirisData.columns((6,6))
                    from iris import svm_iris
                    iris_classifier=svm_iris()
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
                    iris_classifier.fit_model(x_train,y_train)
                    acc= iris_classifier.test_model( x_test ,y_test)
                    iriscol2.header(" Model Accuracy  \t "+ str(acc)+ "")
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

            elif selected_dataset==   'Smoke detection' :
                tab_smokeData,tab_smokeGraphs=st.tabs(("Data","Graphs"))
                with tab_smokeData:
                    from smoke_detection import smoke_svm
                    import numpy as np

                    smoke_detector=smoke_svm()    

                    df=smoke_detector.load_data()
                    tab_smokeData.subheader("Sample Data")
                    tab_smokeData.write(df.sample(10))

                    tab_smokeData.subheader("Smoke data Statistics")

                    tab_smokeData.write(df.describe().transpose())
                    smoke_col1,smoke_col2=tab_smokeData.columns((6,6))
                    smoke_col2.subheader("Missing values ")
                    smoke_col2.write(df.isnull().sum())
                    corr=df.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    with sns.axes_style("dark"):
                        f, ax = plt.subplots(figsize=(6, 6))
                        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
                    smoke_col1.subheader("Corelation Heat Map")
                    smoke_col1.pyplot(f)
                    smoke_col2.subheader("Classes representation")
                    smoke_col2.write( df["Fire Alarm"].value_counts())
                    xtrain,xtest,ytrain,ytest=smoke_detector.train_test(df)
                    smoke_detector.fit_model(xtrain,ytrain)
                    acc=smoke_detector.test_model(xtest,ytest)
                    smoke_col2.subheader("Accuracy ="+ str(acc)) 
                    smoke_col1.subheader("predict Smoke Fire Alarm ")
                    userinput=smoke_detector.user_input_smoke(smoke_col1,smoke_col2,df)
                    #smoke_col2.write(userinput)
                    
                    prediction=smoke_detector.predict_input(userinput)
                    smoke_col1.subheader("The predicted fire alarm state is "+ str(prediction))
                    
                with tab_smokeGraphs:
                    graphcol1,graphcol2=tab_smokeGraphs.columns((5,5))
                    x=df.iloc[:100,1:14]
                    x.hist()
                    plt.show()
                    graphcol1.header("Features Histogram")
                    graphcol1.pyplot(plt)
                    x.plot()
                    plt.show()

                    graphcol2.header("Features Plot")
                    graphcol2.pyplot(plt)
                    #Corelation degree
                    #fig,a = plt.subplots(figsize=(6, 6))
                    fig, ax = plt.subplots() 
                    ax=df.corr()['Fire Alarm'].sort_values().plot(kind='bar')   
                    graphcol1.subheader("Sorted Features-Fire Alarm Corelation")
                    graphcol1.pyplot(fig)
        
                    #a=df.plot(kind='bar', y='Temperature[C]', x='Fire Alarm', color=color[0],rot=90)    
                            
                    #feature ploting
                    x=df.iloc[:100,:]
                    y=np.arange(0,100)
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    x['y']=y
                    ax1 = x.plot(kind='scatter', y='Temperature[C]', x='y', color=color[0],rot=90)    
                    ax2 = x.plot(kind='scatter', y='Humidity[%]', x='y', color=color[1], ax=ax1)    
                    ax3 = x.plot(kind='scatter', y='TVOC[ppb]', x='y', color=color[2], ax=ax1)
                    ax4 = x.plot(kind='scatter', y='eCO2[ppm]', x='y', color=color[3],ax=ax1)    
                    ax5 = x.plot(kind='scatter', y='Raw H2', x='y', color=color[4],ax=ax1)  
                    ax6 = x.plot(kind='scatter', y='Raw Ethanol', x='y', color=color[3],ax=ax1)    
                    ax7 = x.plot(kind='scatter', y='Pressure[hPa]', x='y', color=color[4],ax=ax1)  

                    graphcol2.header("Feature Scatter Matrix")
                    plt.xlabel("Features")
                    plt.ylabel("Fire Alarm")
                    graphcol2.pyplot(plt)

            elif selected_dataset=='Digit recognition':
                tab_digitData,tab_digitGraph=st.tabs(("Data","Graphs"))

                with tab_digitData:
                    from streamlit_drawable_canvas import st_canvas
                    import cv2
                    from digit_recognition import digit_svm
                    import random
                    from PIL import Image

                    tab_digitData.subheader("Sample Data")
                    
                    digit_col1,digit_col2=tab_digitData.columns((6,6))
                    digit_recognizer=digit_svm()    
                    x_train, y_train, x_test, y_test=digit_recognizer.load_and_split_data_()
                    
                    #fig = plt.figure(figsize=(1,1))
                    #for j in range(1,3,1):
                    #   i=random.randint(1,10) 
                    #  plt.subplot(1,3,j)
                    # plt.imshow(df.images[i], cmap='binary')
                        #plt.title(df.target[i])
                        #plt.axis('off')
                    #digit_col1.pyplot(fig)
                    digit_col2.write(x_train.head())
                    from sklearn.preprocessing import   StandardScaler
                    mxscaler=StandardScaler()
                    x_train=mxscaler.fit_transform(x_train)
                    x_test=mxscaler.transform(x_test)

                    digit_col2.subheader("Classes representation")
                    digit_col2.write(y_train.value_counts())
                    
                    #xtrain,xtest,ytrain,ytest=digit_recognizer.train_test(df)
                
                    digit_recognizer.fit_model(x_train,y_train)
                    digit_col2.subheader("Accuracy =")

                    acc=digit_recognizer.test_model(x_test, y_test)
                    digit_col2.write(acc)


                    from skimage.transform import resize
                    from PIL import Image # note that this is using 'Pillow'

                    #display_img = (Image.fromarray((img_2d * 255).astype(np.uint8)).resize(img_2d.shape))

                    canvas_result = st_canvas(stroke_width=20,  background_color="White", width = 140, height= 140)
                    if canvas_result.json_data is not None :

                    
                        from PIL import  Image

                        
                        im = Image.fromarray(canvas_result.image_data)
                        newimg=im.resize((28,28),Image.LANCZOS).convert("L")
                        topredictimg=np.array(newimg)
                        topredictimg=mxscaler.transform(topredictimg.reshape(1,-1))
                        predicted = digit_recognizer.predict_input(topredictimg)

                        st.write(predicted)
                    
                with tab_digitGraph:

                           pass

            elif selected_dataset=="Diabetes":
                import numpy as np
                tab_diabetesData,tab_diabetesGraphs=st.tabs(("Data","Graphs"))
                with tab_diabetesData:
                    from diabetes import diabetes_svm
                    diabet_classifier=diabetes_svm()
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
                    diabet_classifier.fit_model(x_train,y_train)
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
                    tab_diabetesGraphs.pyplot(plt)

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

    def svm_applied(self):
        pass
# %%
