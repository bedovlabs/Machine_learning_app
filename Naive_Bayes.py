import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from Page_layout import main_page
import streamlit as st
import seaborn as sns
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
#from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


class supervised_naive(object):

    def __init__(self):
        self.ax = plt.figure()

    def naive_tut(self):
        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
            Try_col1,Try_col2=Try_algorithm.columns((5,5))
            self.train_test_param=Try_col1.slider("Train-Test Split value",.5,.9,.1)
            self.learning_rate_parameter=Try_col1.slider("Learning Rate",min_value=.001,max_value=.1,step=0.001)
            Go_model_BTN=Try_col1.button("Go")

            #alignh(1,col2)
            #col2.markdown("#####")
            No_of_iteration=Try_col2.slider("Iterations",10,200,1)
            Frame_interva=Try_col2.slider("Interval",50,150,10)
            Generate_dataBTN =Try_col2.button("Generate New Data")

            if Go_model_BTN:
                animation=self.draw(Frame_interva,No_of_iteration,nsamples=120)
                components.html(animation.to_jshtml(), height=600,width=600,scrolling=True)
 
            if Generate_dataBTN:
                main_page.gen_new_data()
                animation= self.draw(Frame_interva,No_of_iteration)
                components.html(animation.to_jshtml(), height=600,width=600,scrolling=True)
       
        with How_it_work:
            how_col1,how_col2=How_it_work.columns((8,1))
            how_col1.subheader("Naive Bayes Component")
            how_col1.image("media/naive_classifier1.png")
            how_col1.subheader("Naive Bayes Data piplining")
            how_col1.image("media/naive_classifier3.png")
            how_col1.subheader("Naive Bayes Structure ")
            how_col1.image("media/naive_classifier2.png")

    
    
    def naive_inaction(self):

        col1,col2=st.columns((5,5))
        col1.subheader("Select Dataset")

        selected_dataset=col1.selectbox("Datasets",options=['Smart Irrigation','Iris','Smoke detection',
                                                          'Diabetes','Diseses','Image classifier',
                                                          'Article classifier','Digit recognition'])
        
        col2.subheader("Set Naive Bayes Parameter")
       
        naiev_algorithm=col2.selectbox("Algorithm",options=['GaussianNB', 'MultinomialNB', 'ComplementNB','BernoulliNB'])

        Alpha=col2.slider("Alpha",0.1,1.0,0.1)

        main_page.alignh(2,col1)
        Go =col1.button("Go")

        if Go :
            if selected_dataset=='Smart Irrigation':
                import numpy as np

                tab_irrigData,tab_irrigGraph=st.tabs(("Data","Graphs"))

                with tab_irrigData :
                    from smart_irrigation import naive_irr
                    irrigation_naive=naive_irr()
                    df=irrigation_naive.load_data()
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
                    xtrain,xtest,ytrain,ytest=irrigation_naive.train_test(df)
                    irrigation_naive.fit_model(xtrain,ytrain,naiev_algorithm,Alpha)
                    acc=irrigation_naive.test_model(xtest,ytest)
                    irr_col2.subheader("Acurracy ="+acc)
                    irr_col1.subheader("Predict Irrigation Status")
                    userinput=irrigation_naive.user_input_smart_irr(irr_col1,irr_col2)
                    prediction=irrigation_naive.predict_input(userinput)
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

#***************#***************#***************#***************#***************#***************
#***************#***************#***************#***************#***************#***************     
       
                
            elif selected_dataset=='Iris':
                import numpy as np
                #tabiris=st.tabs(("Iris flower Classification"))
                tabirisData,tabirisGraphs=st.tabs(("Data","Graphs"))
                with tabirisData:
                    iriscol1,iriscol2=tabirisData.columns((6,6))
                    from iris import naive_iris
                    iris_classifier=naive_iris()

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
                    iris_classifier.fit_model(x_train,y_train,naiev_algorithm,Alpha)
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

#***************#***************#***************#***************#***************#***************
#***************#***************#***************#***************#***************#***************
            elif selected_dataset=='Smoke detection':
                tab_smokeData,tab_smokeGraphs=st.tabs(("Data","Graphs"))
                with tab_smokeData:
                    from smoke_detection import smoke_naive
                    import numpy as np

                    smoke_detector=smoke_naive()    

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
                    smoke_detector.fit_model(xtrain,ytrain,naiev_algorithm,Alpha)
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
#***************#***************#***************#***************#***************#***************
#***************#***************#***************#***************#***************#***************



            elif selected_dataset=='Diabetes':
                tab_diabetesData,tab_diabetesGraphs=st.tabs(("Data","Graphs"))
                with tab_diabetesData:
                    from diabetes import diabetes_naiev
                    diabet_classifier=diabetes_naiev()
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
                    diabet_classifier.fit_model(x_train,y_train,naiev_algorithm,Alpha)
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
   
#***************#***************#***************#***************#***************#***************
#***************#***************#***************#***************#***************#***************
            elif selected_dataset=='Diseses':
                tabdiseses=st.tabs(("Diseses Prediction"))
                pass
#***************#***************#***************#***************#***************#***************
#***************#***************#***************#***************#***************#***************                
            elif selected_dataset=='Image classifier':
                images=st.tabs(('Image classifier'))
                pass
#***************#***************#***************#***************#***************#***************
#***************#***************#***************#***************#***************#***************               
                
            elif selected_dataset=='Article classifier':
                pass
#***************#***************#***************#***************#***************#***************
#***************#***************#***************#***************#***************#***************
            elif selected_dataset=='Digit recognition':
                tab_digitData,tab_digitGraph=st.tabs(("Data","Graphs"))

                with tab_digitData:
                    from streamlit_drawable_canvas import st_canvas
                    import cv2
                    from digit_recognition import digit_naive
                    import random
                    from PIL import Image

                    tab_digitData.subheader("Sample Data")
                    
                    digit_col1,digit_col2=tab_digitData.columns((6,6))
                    digit_recognizer=digit_naive()    
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
                
                    digit_recognizer.fit_model(x_train,y_train,naiev_algorithm,Alpha)
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
                















        
                

     
       
##########################################################################
        

        
    def naive_applied(self):
        pass
    
 
 
   # def naive_data(self):
        
        
   # def init(self):
    #    self.line.set_data([], [])
     #   return self.line    
    def draw(self,frames,interval,nsamples=100):
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=nsamples,n_features= 2, centers=2, random_state=2, cluster_std=1.5)
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(X, y)
        rng = np.random.RandomState(0)
        Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
        ynew = model.predict(Xnew)

        fig, ax = plt.subplots()

        ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
        ax.set_title('Naive Bayes Model', size=14)

        line, = ax.plot([], [], lw = 0)

        # what will our line dataset
        # contain?
        def init():
            line.set_data([], [])
            return line,

        # initializing empty values
        # for x and y co-ordinates
        xdata, ydata = [], []

        # animation function
        def animate(i):
            # t is a parameter which varies
            # with the frame number
            rng = np.random.rand(1,2)
            Xnew = [-6, -14] + [14, 18] * rng
            ynew = model.predict(Xnew)
            
            # x, y values to be plotted
            
            
            # appending values to the previously
            # empty x and y data holders
            xdata.append(Xnew[:, 0])
            ydata.append( Xnew[:, 1])
            line.set_data(xdata, ydata)
            if ynew == 0:
                ax.plot(Xnew[:, 0],Xnew[:, 1], 'r+')
            else:
                ax.plot(Xnew[:, 0],Xnew[:, 1], 'bo')
            return line,
        anim = animation.FuncAnimation(fig, animate,
                                    init_func = init,
                                    frames = frames,
                                    interval = interval,
                                    blit = True)
        return anim
                
            # calling the animation function
                

            # saves the animation in our desktonp
            #anim.save('growingCoil.mp4', writer = 'ffmpeg', fps = 30)
            # Enable interactive plot





