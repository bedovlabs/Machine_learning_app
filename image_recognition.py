import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import pickle


class svm_image(object):
    def load_data(self):
        Categories=['Cats','Dogs']
        flat_data_arr=[] #input array
        target_arr=[] #output array
        targets=[]
        datadir='media/animals/' 
        #path which contains all the categories of images
        for i in Categories:
            #print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))
                img_resized=resize(img_array,(150,150,3))
                flat_data_arr.append(img_resized.flatten())
                targets.append(i)
                target_arr.append(Categories.index(i))
           # print(f'loaded category:{i} successfully')
        flat_data=np.array(flat_data_arr)
        #target=np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        origdf=df.copy()
        origdf['Target']=targets
        df['Target']=target_arr
        x=df.iloc[:,:-1] #input data 
        y=df.iloc[:,-1] #output data
        return x,y,origdf
    def train_test(self,x,y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
        #print('Splitted Successfully')
        return x_train,x_test,y_train,y_test
    def fit_model(self,x_train,y_train):
          def fit_model(self,x_train,y_train):
            from sklearn import svm
            from sklearn.svm import SVC
            model=SVC(C=1, kernel='rbf', gamma=0.0001)
            model.fit(x_train,y_train)
            filename = 'savedmodels/svm_image_class.sav'
            pickle.dump(model, open(filename, 'wb'))

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/svm_image_class.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/svm_image_class.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
#***********************************************************************
#***********************************************************************


class image_dectree(object):
    def load_data(self):
        Categories=['Cats','Dogs']
        flat_data_arr=[] #input array
        target_arr=[] #output array
        targets=[]
        datadir='media/animals/' 
        #path which contains all the categories of images
        for i in Categories:
            #print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))
                img_resized=resize(img_array,(150,150,3))
                flat_data_arr.append(img_resized.flatten())
                targets.append(i)
                target_arr.append(Categories.index(i))
           # print(f'loaded category:{i} successfully')
        flat_data=np.array(flat_data_arr)
        #target=np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        origdf=df.copy()
        origdf['Target']=targets
        df['Target']=target_arr
        x=df.iloc[:,:-1] #input data 
        y=df.iloc[:,-1] #output data
        return x,y,origdf
    def train_test(self,x,y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
        #print('Splitted Successfully')
        return x_train,x_test,y_train,y_test
   
    def fit_model(self,x_train,y_train,criteration,maxdepth,maxfeature,nsamplsplit):
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(criterion=criteration,max_depth=maxdepth,
                                            max_features=maxfeature ,
                                            min_samples_split=nsamplsplit ).fit(x_train,y_train)

            filename = 'savedmodels/dc_tree_image_model.sav'
            pickle.dump(model, open(filename, 'wb'))

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/dc_tree_image_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/dc_tree_image_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction

#***********************************************************************
#***********************************************************************
class image_knn(object):
    def load_data(self):
        Categories=['Cats','Dogs']
        flat_data_arr=[] #input array
        target_arr=[] #output array
        targets=[]
        datadir='media/animals/' 
        #path which contains all the categories of images
        for i in Categories:
            #print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))
                img_resized=resize(img_array,(150,150,3))
                flat_data_arr.append(img_resized.flatten())
                targets.append(i)
                target_arr.append(Categories.index(i))
           # print(f'loaded category:{i} successfully')
        flat_data=np.array(flat_data_arr)
        #target=np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        origdf=df.copy()
        origdf['Target']=targets
        df['Target']=target_arr
        x=df.iloc[:,:-1] #input data 
        y=df.iloc[:,-1] #output data
        return x,y,origdf
    def train_test(self,x,y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
        #print('Splitted Successfully')
        return x_train,x_test,y_train,y_test
   
    
    def fit_model(self,x_train,y_train,no_neigbors,algorithm,leaf_size):
        from sklearn.neighbors import KNeighborsClassifier
        
        model = KNeighborsClassifier(n_neighbors=no_neigbors,algorithm=algorithm,leaf_size=leaf_size).fit(x_train,y_train)

        filename = 'savedmodels/knn_image_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data


    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/knn_image_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/knn_image_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction


#***********************************************************************
#***********************************************************************
class image_naieve(object):
    def load_data(self):
        Categories=['Cats','Dogs']
        flat_data_arr=[] #input array
        target_arr=[] #output array
        targets=[]
        datadir='media/animals/' 
        #path which contains all the categories of images
        for i in Categories:
            #print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))
                img_resized=resize(img_array,(150,150,3))
                flat_data_arr.append(img_resized.flatten())
                targets.append(i)
                target_arr.append(Categories.index(i))
           # print(f'loaded category:{i} successfully')
        flat_data=np.array(flat_data_arr)
        #target=np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        origdf=df.copy()
        origdf['Target']=targets
        df['Target']=target_arr
        x=df.iloc[:,:-1] #input data 
        y=df.iloc[:,-1] #output data
        return x,y,origdf
    def train_test(self,x,y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
        #print('Splitted Successfully')
        return x_train,x_test,y_train,y_test
   
    def fit_model(self,x_train,y_train,naive_algorithm,alfa):


        if naive_algorithm=='GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
        elif naive_algorithm=='MultinomialNB':
            from sklearn.naive_bayes import MultinomialNB
            model = MultinomialNB(alpha=alfa)

        elif naive_algorithm=='CategoricalNB':
            from sklearn.naive_bayes import CategoricalNB 
            model = CategoricalNB(alpha=alfa)
        elif naive_algorithm=='ComplementNB':
            from sklearn.naive_bayes import ComplementNB
            model = ComplementNB(alpha=alfa)
        elif  naive_algorithm=='BernoulliNB':
            from sklearn.naive_bayes import BernoulliNB
            model = BernoulliNB(alpha=alfa)
        from sklearn.naive_bayes import GaussianNB

        model.fit(x_train, y_train)
        filename = 'savedmodels/naive_image_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naive_image_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/naive_image_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction


#***********************************************************************
#***********************************************************************
class image_logistic(object):
    def load_data(self):
        Categories=['Cats','Dogs']
        flat_data_arr=[] #input array
        target_arr=[] #output array
        targets=[]
        datadir='media/animals/' 
        #path which contains all the categories of images
        for i in Categories:
            #print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))
                img_resized=resize(img_array,(150,150,3))
                flat_data_arr.append(img_resized.flatten())
                targets.append(i)
                target_arr.append(Categories.index(i))
           # print(f'loaded category:{i} successfully')
        flat_data=np.array(flat_data_arr)
        #target=np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        origdf=df.copy()
        origdf['Target']=targets
        df['Target']=target_arr
        x=df.iloc[:,:-1] #input data 
        y=df.iloc[:,-1] #output data
        return x,y,origdf
    def train_test(self,x,y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
        #print('Splitted Successfully')
        return x_train,x_test,y_train,y_test
   
    def fit_model(self,x_train,y_train,Clf,Solver):
        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression(penalty=Clf,solver=Solver)
        model.fit(x_train,y_train)

        filename = 'savedmodels/logistic_image_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data


    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/logistic_image_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/logistic_image_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction