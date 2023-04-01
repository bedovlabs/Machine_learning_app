from hashlib import new
from cv2 import kmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import streamlit as st
from Page_layout import main_page
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
from itertools import product
from matplotlib import animation
from sklearn.datasets import make_blobs

class Supervised_kmeans(object):
    
    
    def kmeans_tut(self):      
        How_it_work,Try_algorithm=st.tabs(("How the algorithm work","Try Algorithm"))         
        with Try_algorithm:
            Try_col1,Try_col2=Try_algorithm.columns((5,5))
            self.n_clustersint=Try_col1.slider("No of clusters",2,15,1)
            self.algorithm=Try_col1.selectbox("Algorithm",options=['lloyd', 'elkan', 'auto','full'])
            
            Go_model_BTN=Try_col1.button("Go")
            self.cluster_std=Try_col2.slider("Cluster standerd deviation",.5,10.0,.5)
            self.nsamples=Try_col2.slider("No of samples",100,1000,50)
            Generate_dataBTN =Try_col2.button("Generate New Data")

            if Go_model_BTN:

              graph=self.load_and_visualize()
              components.html(graph.to_jshtml(), height=600,width=600,scrolling=True)
              
            
            if Generate_dataBTN:
              graph=self.load_and_visualize()
              components.html(graph.to_jshtml(), height=600,width=600,scrolling=True)   
                
                
        with How_it_work:
            how_col1,how_col2=How_it_work.columns((8,1))
            how_col1.image("media/kmeans1.png")
            how_col1.image("media/kmeans2.png")
            how_col1.image("media/kmeans3.png")
           
    
        #Animation function


    def kmeans_inaction(self):      

        col1, col2 = st.columns((3,3))
        col1.subheader("Set Kmeans  Parameter")
        self.n_clusters=col1.slider("No of clusters",2,15,1)
        self.algorithm=col1.selectbox("Algorithm",options=['lloyd', 'elkan', 'auto','full'])
        main_page.alignh(3,col2)
        self.max_iteration=col2.slider("Max Iterations",300,1000,50)
        articles_tab,tab_irrig,tab_new=st.tabs(("Article Classification","Smart Irrigation","new"))
        tab_newData,tab_newGraphs=tab_new.tabs(("Data","Graphs"))
        articlesData,articlesGraph=articles_tab.tabs(("Data","Graphs"))
        tab_irrigData,tab_irrigGraph=tab_irrig.tabs(("Data","Graphs"))
      #  with tabiris:
        with tab_newData:
           
            pass
      
        with tab_newGraphs:
            pass
        with tab_irrigData :
            from smart_irrigation import kmeans_irr
            irrigation_kmeans=kmeans_irr()
            df=irrigation_kmeans.load_data()
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
            xtrain,xtest,ytrain,ytest=irrigation_kmeans.train_test(df)

            irrigation_kmeans.fit_model(xtrain,ytrain,self.n_clusters,self.algorithm,self.max_iteration)
            acc=irrigation_kmeans.test_model(xtest,ytest)
            irr_col2.subheader("Acurracy ="+acc)
            irr_col1.subheader("Predict Irrigation Status")
            userinput=irrigation_kmeans.user_input_smart_irr(irr_col1,irr_col2)
            prediction=irrigation_kmeans.predict_input(userinput)
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


        with articlesData :   
                       
            articlesData.subheader("Sample Data ")
            text_col1,text_col2=articlesData.columns((6,6))
            articles_classifier=Kmeans_text()
            df=articles_classifier.load_articles()
            articlesData.subheader("Sample Data")
            articlesData.write(df.sample(10))
            text_col1,text_col2=articlesData.columns((6,6))
            text_col1.subheader("Missing values ")
            text_col1.write(df.isnull().sum())
            text_col2.subheader(" Equal Class representation")
            text_col2.write( df['category'].value_counts())
            X_train,y_train,x_test,y_test=articles_classifier.train_test(df)
            articles_classifier.fit_model(X_train,y_train)
            text_col1.subheader("predict article  Category ")
            article_to_predict=text_col2.text_area("Put Your Text Here",height=200)
            text_col1.write(articles_classifier.predict_input(article_to_predict))
           


   

    def load_and_visualize(self):
        
        #change number of centroids************************
        points, _ = make_blobs(cluster_std=self.cluster_std, n_samples=self.nsamples, n_features=2, random_state=1,)
        global centroids
        global closest

        def initialize_centroids(points, k):
            """returns k centroids from the initial points"""
            centroids = points.copy()
            np.random.shuffle(centroids)
            return centroids[:k]


        def closest_centroid(points, centroids):
            """returns an array containing the index to the nearest centroid for
        each point"""
            distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            return np.argmin(distances, axis=0)


        def move_centroids(points, closest, centroids):
            """returns the new centroids assigned from the points closest to them"""
            return np.array(
                [points[closest == k].mean(axis=0) for k in range(centroids.shape[0])]
            )
        def make_grid(xlim, ylim, xnum=500, ynum=500):
            xs = np.linspace(*xlim, num=xnum)
            ys = np.linspace(*ylim, num=ynum)
            points = np.zeros(shape=[xnum * ynum, 2])
            for n, (i, j) in enumerate(product(range(xnum), range(ynum))):
                x = xs[i]
                y = ys[j]
                points[n, :] = [x, y]
            return points

        def make_tessallation(grid, centroids, xnum=500, ynum=500):
            clusters = closest_centroid(grid, centroids)
            points = np.zeros(shape=[xnum, ynum])
            for n, (i, j) in enumerate(product(range(xnum), range(ynum))):
                points[j, i] = clusters[n]
            return points


        def init():
            return [fig]


        def animate(i):
            global centroids
            global closest
            ax = plt.axes()
            centroids = move_centroids(points, closest, centroids)
            closest = closest_centroid(points, centroids)
            grid = make_grid(xlim, ylim, num, num)
            tes = make_tessallation(grid, centroids)
            ax.cla()
            ax.contourf(xs, ys, tes, alpha=0.15, colors=colors)
            cs = [colors[k] for k in closest]
            ax.scatter(points[:, 0], points[:, 1], c=cs, s=15)
            ax.scatter(centroids[:, 0], centroids[:, 1], c="k", edgecolor="w", marker="X", s=150)
            ax.contour(xs, ys, tes, colors="k")
            ax.set_xticklabels("")
            ax.set_yticklabels("")
           # plt.show
            return [fig]

        centroids = initialize_centroids(points, 7)
        closest = closest_centroid(points, centroids)
        xlim = (points[:, 0].min(), points[:, 0].max())
        ylim = (points[:, 1].min(), points[:, 1].max())
        num = 500
        xs = np.linspace(*xlim, num)
        ys = np.linspace(*ylim, num)

        colors = sns.color_palette()
        fig = plt.figure(figsize=(6, 4))


        ani = animation.FuncAnimation( fig, animate, init_func=init, frames=20, interval=100, blit=True)


        return ani
            


    