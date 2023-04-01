import pandas as pd
from requests import options
import seaborn as sns
from sklearn import svm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class svm_disese(object):
    def __init__(self):
        #global vectorizer
        self.vectorizer= CountVectorizer()
        self.le = LabelEncoder()

    def load_data(self):
        original_df=pd.read_csv('data/diesese.csv')
        original_df['symptoms'] = original_df[original_df.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df=original_df[['Disease','symptoms']]
        y = df['Disease']
        X = df['symptoms']
        X =self.vectorizer.fit_transform(X)
        maxfeat=self.vectorizer.get_feature_names()
      
        X = X.toarray()
        return df,X,y,maxfeat
    
    def train_test(self,X,y):
        yy=self.le.fit_transform(y)
        x_train,x_test,y_train,y_test=train_test_split(X,yy,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train):
        from sklearn.svm import SVC
        model = SVC(gamma='auto')
        model.fit(x_train, y_train)
        filename = 'savedmodels/svmdieses_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/svmdieses_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/svmdieses_model.sav', 'rb'))
        
        symptomps=' '.join(user_input)
       
        user_input = self.vectorizer.transform([symptomps]).toarray()
        user_prediction=loaded_model.predict(user_input)
        disease =self.le.classes_[user_prediction]
        return disease

#**************************************************************************************
#**************************************************************************************



class naieve_disese(object):
    def __init__(self):
        #global vectorizer
        self.vectorizer= CountVectorizer()
        self.le = LabelEncoder()

    def load_data(self):
        original_df=pd.read_csv('data/diesese.csv')
        original_df['symptoms'] = original_df[original_df.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df=original_df[['Disease','symptoms']]
        y = df['Disease']
        X = df['symptoms']
        X =self.vectorizer.fit_transform(X)
        maxfeat=self.vectorizer.get_feature_names()
      
        X = X.toarray()
        return df,X,y,maxfeat
    
    def train_test(self,X,y):
        yy=self.le.fit_transform(y)
        x_train,x_test,y_train,y_test=train_test_split(X,yy,random_state=0)
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
        filename = 'savedmodels/naive_diseses_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naive_diseses_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/naive_diseses_model.sav', 'rb'))
        
        symptomps=' '.join(user_input)
       
        user_input = self.vectorizer.transform([symptomps]).toarray()
        user_prediction=loaded_model.predict(user_input)
        disease =self.le.classes_[user_prediction]
        return disease


#**************************************************************************************
#**************************************************************************************
class dctree_disese(object):
    def __init__(self):
        #global vectorizer
        self.vectorizer= CountVectorizer()
        self.le = LabelEncoder()

    def load_data(self):
        original_df=pd.read_csv('data/diesese.csv')
        original_df['symptoms'] = original_df[original_df.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df=original_df[['Disease','symptoms']]
        y = df['Disease']
        X = df['symptoms']
        X =self.vectorizer.fit_transform(X)
        maxfeat=self.vectorizer.get_feature_names()
        X = X.toarray()
        return df,X,y,maxfeat
    
    def train_test(self,X,y):
        yy=self.le.fit_transform(y)
        x_train,x_test,y_train,y_test=train_test_split(X,yy,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train,criteration,maxdepth,maxfeature,nsamplsplit):
        
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion=criteration,max_depth=maxdepth,
                                            max_features=maxfeature ,
                                            min_samples_split=nsamplsplit ).fit(x_train,y_train)

        filename = 'savedmodels/dec_tree_diseses_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/dec_tree_diseses_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/dec_tree_diseses_model.sav', 'rb'))
        
        symptomps=' '.join(user_input)
       
        user_input = self.vectorizer.transform([symptomps]).toarray()
        user_prediction=loaded_model.predict(user_input)
        disease =self.le.classes_[user_prediction]
        return disease



#**************************************************************************************
#**************************************************************************************
class knn_disese(object):
    def __init__(self):
        #global vectorizer
        self.vectorizer= CountVectorizer()
        self.le = LabelEncoder()

    def load_data(self):
        original_df=pd.read_csv('data/diesese.csv')
        original_df['symptoms'] = original_df[original_df.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df=original_df[['Disease','symptoms']]
        y = df['Disease']
        X = df['symptoms']
        X =self.vectorizer.fit_transform(X)
        maxfeat=self.vectorizer.get_feature_names()
      
        X = X.toarray()
        return df,X,y,maxfeat
    
    def train_test(self,X,y):
        yy=self.le.fit_transform(y)
        x_train,x_test,y_train,y_test=train_test_split(X,yy,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train,no_neigbors,algorithm,leaf_size):
        
        model = KNeighborsClassifier(n_neighbors=no_neigbors,algorithm=algorithm,leaf_size=leaf_size).fit(x_train,y_train)

        filename = 'savedmodels/kmeans_disese_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/kmeans_disese_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/kmeans_disese_model.sav', 'rb'))
        
        symptomps=' '.join(user_input)
       
        user_input = self.vectorizer.transform([symptomps]).toarray()
        user_prediction=loaded_model.predict(user_input)
        disease =self.le.classes_[user_prediction]
        return disease


#**************************************************************************************
#**************************************************************************************

class logoitic_disese(object):
    def __init__(self):
        #global vectorizer
        self.vectorizer= CountVectorizer()
        self.le = LabelEncoder()

    def load_data(self):
        original_df=pd.read_csv('data/diesese.csv')
        original_df['symptoms'] = original_df[original_df.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df=original_df[['Disease','symptoms']]
        y = df['Disease']
        X = df['symptoms']
        X =self.vectorizer.fit_transform(X)
        maxfeat=self.vectorizer.get_feature_names()
        X = X.toarray()
        return df,X,y,maxfeat
    
    def train_test(self,X,y):
        yy=self.le.fit_transform(y)
        x_train,x_test,y_train,y_test=train_test_split(X,yy,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train,Clf,Solver):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model=LogisticRegression(penalty=Clf,solver=Solver)
        model.fit(x_train,y_train)

        filename = 'savedmodels/logistic_disese_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
         
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/logistic_disese_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/logistic_disese_model.sav', 'rb'))
        
        symptomps=' '.join(user_input)
       
        user_input = self.vectorizer.transform([symptomps]).toarray()
        user_prediction=loaded_model.predict(user_input)
        disease =self.le.classes_[user_prediction]
        return disease









    

    def visualize_word_freq(self,input_data,max_words,tfidf=False):
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        import matplotlib.pyplot as plt

        """ 
        Input data should be a list of docs.
        Each doc is represented by one whole string (with preprocessing, eg.remove markups)    
        """
        # Plot configuration
        fig1= plt.figure(figsize=(12, 4), dpi=200)

        #
        
        vectorizer = CountVectorizer(max_features=max_words,stop_words='english')
        TITLE = "Most Freq Words"
        mat = vectorizer.fit_transform(input_data)

        df1=pd.DataFrame(mat.sum(axis=0).T,index=vectorizer.get_feature_names(),columns=['freq']).sort_values(by='freq',ascending=False)
        return df1


        
