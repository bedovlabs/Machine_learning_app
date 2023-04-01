import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
import pickle


class digit_naive(object):
   
    def load_and_split_data_(self):
        train_data=pd.read_csv('data/mnist_train.csv')
        test_data=pd.read_csv('data/mnist_test.csv')
        x_train=train_data.iloc[:,1:]
        y_train=train_data.iloc[:,0]
        x_test=test_data.iloc[:,1:]
        y_test=test_data.iloc[:,0]
      
        return x_train, y_train, x_test, y_test

    #def train_test(self,digits):
     #   y = digits.target
      #  x = digits.images.reshape((len(digits.images), -1))
       # X_train = x[:1000]
        #Y_train = y[:1000]
        #X_test = x[1000:]
        #Y_test = y[1000:]

        #return X_train,X_test,Y_train,Y_test 
        
    def fit_model(self,X_train, Y_train,naive_algorithm,alfa):
        if naive_algorithm=='GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
        elif naive_algorithm=='MultinomialNB':
            from sklearn.naive_bayes import MultinomialNB
            model = MultinomialNB(alpha=alfa)

       # elif naive_algorithm=='CategoricalNB':
          #  from sklearn.naive_bayes import CategoricalNB 
           # model = CategoricalNB(alpha=alfa)
        elif naive_algorithm=='ComplementNB':
            from sklearn.naive_bayes import ComplementNB
            model = ComplementNB(alpha=alfa)
        elif  naive_algorithm=='BernoulliNB':
            from sklearn.naive_bayes import BernoulliNB
            model = BernoulliNB(alpha=alfa)

         
        model.fit(X_train, Y_train)
        filename = 'savedmodels/naive_digit_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naive_digit_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/naive_digit_model.sav', 'rb'))
        digit_prediction=loaded_model.predict(user_input)
        return str(digit_prediction)
    
####################################################################################################
####################################################################################################

class digit_dctree(object):
   
    def load_and_split_data_(self):
        train_data=pd.read_csv('data/mnist_train.csv')
        test_data=pd.read_csv('data/mnist_test.csv')
        x_train=train_data.iloc[:,1:]
        y_train=train_data.iloc[:,0]
        x_test=test_data.iloc[:,1:]
        y_test=test_data.iloc[:,0]
      
        return x_train, y_train, x_test, y_test

   
        
    def fit_model(self,x_train,y_train,criteration,maxdepth,maxfeature,nsamplsplit):
        
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion=criteration,max_depth=maxdepth,
                                            max_features=maxfeature ,
                                            min_samples_split=nsamplsplit ).fit(x_train,y_train)

        filename = 'savedmodels/dec_tree_digit_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/dec_tree_digit_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/dec_tree_digit_model.sav', 'rb'))
        digit_prediction=loaded_model.predict(user_input)
        return str(digit_prediction)
    
####################################################################################################
####################################################################################################
class digit_knn(object):
   
    def load_and_split_data_(self):
        train_data=pd.read_csv('data/mnist_train.csv')
        test_data=pd.read_csv('data/mnist_test.csv')
        x_train=train_data.iloc[:,1:]
        y_train=train_data.iloc[:,0]
        x_test=test_data.iloc[:,1:]
        y_test=test_data.iloc[:,0]
      
        return x_train, y_train, x_test, y_test

   
    def fit_model(self,x_train,y_train,no_neigbors,algorithm,leaf_size):
        from sklearn.neighbors import KNeighborsClassifier
        
        model = KNeighborsClassifier(n_neighbors=no_neigbors,algorithm=algorithm,leaf_size=leaf_size).fit(x_train,y_train)

        filename = 'savedmodels/knn_digit_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/knn_digit_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/knn_digit_model.sav', 'rb'))
        digit_prediction=loaded_model.predict(user_input)
        return str(digit_prediction)
    

####################################################################################################
####################################################################################################
class digit_logistic(object):
   
    def load_and_split_data_(self):
        train_data=pd.read_csv('data/mnist_train.csv')
        test_data=pd.read_csv('data/mnist_test.csv')
        x_train=train_data.iloc[:,1:]
        y_train=train_data.iloc[:,0]
        x_test=test_data.iloc[:,1:]
        y_test=test_data.iloc[:,0]
      
        return x_train, y_train, x_test, y_test

 
        
    def fit_model(self,x_train,y_train,Clf,Solver):
        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression(penalty=Clf,solver=Solver)
        model.fit(x_train,y_train)

        filename = 'savedmodels/logistic_digit_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/logistic_digit_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/logistic_digit_model.sav', 'rb'))
        digit_prediction=loaded_model.predict(user_input)
        return str(digit_prediction)
    

####################################################################################################
####################################################################################################

class digit_svm(object):
   
    def load_and_split_data_(self):
        train_data=pd.read_csv('data/mnist_train.csv')
        test_data=pd.read_csv('data/mnist_test.csv')
        x_train=train_data.iloc[:,1:]
        y_train=train_data.iloc[:,0]
        x_test=test_data.iloc[:,1:]
        y_test=test_data.iloc[:,0]
      
        return x_train, y_train, x_test, y_test

           
    def fit_model(self,x_train,y_train):
        from sklearn.svm import SVC
        model = SVC(gamma='auto')
        model.fit(x_train, y_train)
        filename = 'savedmodels/svm_digit_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/svm_digit_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/svm_digit_model.sav', 'rb'))
        digit_prediction=loaded_model.predict(user_input)
        return str(digit_prediction)
    