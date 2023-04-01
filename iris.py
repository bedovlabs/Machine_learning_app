import pandas as pd
from requests import options
import seaborn as sns
from sklearn import svm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from Page_layout import main_page

class svm_iris(object):
    def load_data(self):
        df=pd.read_csv('data/iris.csv')
        return df
    def train_test(self,df):
        x=df.iloc[:,:4]
        y=df.iloc[:,4]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        return x_train,x_test,y_train,y_test

    
    def fit_model(self,x_train,y_train):
        from sklearn.svm import SVC
        model = SVC(gamma='auto')
        model.fit(x_train, y_train)
        filename = 'savedmodels/svmiris_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/svmiris_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/svmiris_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
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
########################################################################################################################
class logistic_iris(object):
    def load_data(self):
        df=pd.read_csv('data/iris.csv')
        return df
    def train_test(self,df):
        x=df.iloc[:,:4]
        y=df.iloc[:,4]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train,Clf,Solver):
        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression(penalty=Clf,solver=Solver)
        model.fit(x_train,y_train)

        filename = 'savedmodels/logisticiris_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/logisticiris_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/logisticiris_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
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
  
  
    
####################################################################################################
class naive_iris(object):
    def load_data(self):
        df=pd.read_csv('data/iris.csv')
        return df
    def train_test(self,df):
        x=df.iloc[:,:4]
        y=df.iloc[:,4]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
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
        filename = 'savedmodels/naive_iris_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naive_iris_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/naive_iris_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
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
################################################################################
class dec_tree_iris(object):
    def load_data(self):
        df=pd.read_csv('data/iris.csv')
        return df
    def train_test(self,df):
        x=df.iloc[:,:4]
        y=df.iloc[:,4]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        return x_train,x_test,y_train,y_test


    def fit_model(self,x_train,y_train,criteration,maxdepth,maxfeature,nsamplsplit):
        
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion=criteration,max_depth=maxdepth,
                                            max_features=maxfeature ,
                                            min_samples_split=nsamplsplit ).fit(x_train,y_train)

        filename = 'savedmodels/dec_tree_iris_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/dec_tree_iris_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/dec_tree_iris_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
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

   
################################################################################
class knn_iris(object):
    def load_data(self):
        df=pd.read_csv('data/iris.csv')
        return df
    def train_test(self,df):
        x=df.iloc[:,:4]
        y=df.iloc[:,4]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train,no_neigbors,algorithm,leaf_size):
        
        model = KNeighborsClassifier(n_neighbors=no_neigbors,algorithm=algorithm,leaf_size=leaf_size).fit(x_train,y_train)

        filename = 'savedmodels/knn_iris_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/knn_iris_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/knn_iris_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
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
