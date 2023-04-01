

import re
import json
from pandas.io.json import json_normalize
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
class svc_text(object):
    #def __init__(self):
         # IMPORTING THE REQUIRED LIBRARY FOR CALSSIFICATION AND GROUPING 
        # text cleanup
       
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        #words will be stemed
        try:
            stemer= SnowballStemmer('english')
            STOPWORDS = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stemer= SnowballStemmer('english')
        #stopwords will be removed from articles
            STOPWORDS = set(stopwords.words('english'))
                
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) 
        # lowercase text
        text = text.lower()     
        # delete stopwords from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
        
        # removes any words =< 2 or >= 15 letters
        text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 15))

        # Stemming the words
        text = ' '.join([stemer.stem(word) for word in text.split()])
        return text    


    def load_articles(self):

        f = open ('data/articles.json', "r")
        # Reading from file
        data = json.loads(f.read())
        # READING FILE CONTENT INTO PANDAS DATA FRAME
        articles = pd.DataFrame(data).sample(250)
        #CLOSING FILE
        f.close()
        
        return articles

    def train_test(self,articles):
        articles['cleaned_body'] = articles['body'].apply(self.clean_text)
        # cleaning articles content in a new column "cleaned body"
        # separting input and output 
        artcat=articles['category']
        #drop category column from the data frame
        articles=articles.drop(columns='category')
         # splitiing data To training ,validating and testing
        # train, validate, test = \
        #      np.split(articles.sample(frac=1, random_state=42), 
        #             [int(.6*len(articles)), int(.8*len(articles))])
                
        X_train, x_test, y_train, y_test = train_test_split(articles['cleaned_body'],artcat,test_size=0.2,train_size=0.8)

        return X_train,y_train,x_test,y_test
    
    def fit_model(self,X_train,y_train,x_test,y_test):
        #LinearSVC using SVC clasifier
        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                            ('chi',  SelectKBest(chi2, k=10000)),
                            ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

        #Fitting the model with the training data
        model = pipeline.fit(X_train, y_train)
        filename = 'savedmodels/article_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/article_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/article_model.sav', 'rb'))
        user_prediction=loaded_model.predict([user_input])
        return user_prediction

#*****************************************************************************
#*****************************************************************************

class Knn_text(object):
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        #words will be stemed
        try:
            stemer= SnowballStemmer('english')
            STOPWORDS = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stemer= SnowballStemmer('english')
        #stopwords will be removed from articles
            STOPWORDS = set(stopwords.words('english'))
                
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) 
        # lowercase text
        text = text.lower()     
        # delete stopwords from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
        
        # removes any words =< 2 or >= 15 letters
        text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 15))

        # Stemming the words
        text = ' '.join([stemer.stem(word) for word in text.split()])
        return text    


    def load_articles(self):

        f = open ('data/articles.json', "r")
        # Reading from file
        data = json.loads(f.read())
        # READING FILE CONTENT INTO PANDAS DATA FRAME
        articles = pd.DataFrame(data).sample(250)
        #CLOSING FILE
        f.close()
        
        return articles

        
   
       
    def train_test(self,articles):
        articles['cleaned_body'] = articles['body'].apply(self.clean_text)
        # cleaning articles content in a new column "cleaned body"
        # separting input and output 
        artcat=articles['category']
        #drop category column from the data frame
        articles=articles.drop(columns='category')
        X_train, x_test, y_train, y_test = train_test_split(articles['cleaned_body'],artcat,test_size=0.2,train_size=0.8)
        return X_train,y_train,x_test,y_test
   
   
        

    def fit_model(self,X_train,y_train,no_neigbors,algorithm,leaf_size):
        from sklearn.neighbors import KNeighborsClassifier
        #LinearSVC using SVC clasifier
        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                            ('chi',  SelectKBest(chi2, k=10000)),
                            ('clf',KNeighborsClassifier(n_neighbors=no_neigbors,algorithm=algorithm,leaf_size=leaf_size))])

        #Fitting the model with the training data
        model = pipeline.fit(X_train, y_train)
        filename = 'savedmodels/knn_articles_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/knn_articles_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/knn_articles_model.sav', 'rb'))
        user_prediction=loaded_model.predict([user_input])
        return user_prediction

#*****************************************************************************
#*****************************************************************************
class dctree_text(object):
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        #words will be stemed
        try:
            stemer= SnowballStemmer('english')
            STOPWORDS = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stemer= SnowballStemmer('english')
        #stopwords will be removed from articles
            STOPWORDS = set(stopwords.words('english'))
                
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) 
        # lowercase text
        text = text.lower()     
        # delete stopwords from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
        
        # removes any words =< 2 or >= 15 letters
        text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 15))

        # Stemming the words
        text = ' '.join([stemer.stem(word) for word in text.split()])
        return text    


    def load_articles(self):

        f = open ('data/articles.json', "r")
        # Reading from file
        data = json.loads(f.read())
        # READING FILE CONTENT INTO PANDAS DATA FRAME
        articles = pd.DataFrame(data).sample(250)
        #CLOSING FILE
        f.close()
        
        return articles

        
   
       
    def train_test(self,articles):
        articles['cleaned_body'] = articles['body'].apply(self.clean_text)
        # cleaning articles content in a new column "cleaned body"
        # separting input and output 
        artcat=articles['category']
        #drop category column from the data frame
        articles=articles.drop(columns='category')
        X_train, x_test, y_train, y_test = train_test_split(articles['cleaned_body'],artcat,test_size=0.2,train_size=0.8)
        return X_train,y_train,x_test,y_test
   
   
        

    def fit_model(self,x_train,y_train,criteration,maxdepth,maxfeature,nsamplsplit):
        from sklearn.tree import DecisionTreeClassifier
        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                            ('chi',  SelectKBest(chi2, k=10000)),
                            ('clf',DecisionTreeClassifier(criterion=criteration,max_depth=maxdepth,
                                            max_features=maxfeature ,min_samples_split=nsamplsplit ))])

        #Fitting the model with the training data
        model = pipeline.fit(x_train, y_train)
        filename = 'savedmodels/dctree_articles_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/dctree_articles_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/dctree_articles_model.sav', 'rb'))
        user_prediction=loaded_model.predict([user_input])
        return user_prediction

#*****************************************************************************
#*****************************************************************************
class naieve_text(object):
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        #words will be stemed
        try:
            stemer= SnowballStemmer('english')
            STOPWORDS = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stemer= SnowballStemmer('english')
        #stopwords will be removed from articles
            STOPWORDS = set(stopwords.words('english'))
                
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) 
        # lowercase text
        text = text.lower()     
        # delete stopwords from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
        
        # removes any words =< 2 or >= 15 letters
        text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 15))

        # Stemming the words
        text = ' '.join([stemer.stem(word) for word in text.split()])
        return text    


    def load_articles(self):

        f = open ('data/articles.json', "r")
        # Reading from file
        data = json.loads(f.read())
        # READING FILE CONTENT INTO PANDAS DATA FRAME
        articles = pd.DataFrame(data).sample(250)
        #CLOSING FILE
        f.close()
        
        return articles

        
   
       
    def train_test(self,articles):
        articles['cleaned_body'] = articles['body'].apply(self.clean_text)
        # cleaning articles content in a new column "cleaned body"
        # separting input and output 
        artcat=articles['category']
        #drop category column from the data frame
        articles=articles.drop(columns='category')
        X_train, x_test, y_train, y_test = train_test_split(articles['cleaned_body'],artcat,test_size=0.2,train_size=0.8)
        return X_train,y_train,x_test,y_test
   
   
        

    def fit_model(self,x_train,y_train,naive_algorithm,alfa):


        if naive_algorithm=='GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            modelclf = GaussianNB()
        elif naive_algorithm=='MultinomialNB':
            from sklearn.naive_bayes import MultinomialNB
            modelclf = MultinomialNB(alpha=alfa)

        elif naive_algorithm=='CategoricalNB':
            from sklearn.naive_bayes import CategoricalNB 
            modelclf = CategoricalNB(alpha=alfa)
        elif naive_algorithm=='ComplementNB':
            from sklearn.naive_bayes import ComplementNB
            modelclf = ComplementNB(alpha=alfa)
        elif  naive_algorithm=='BernoulliNB':
            from sklearn.naive_bayes import BernoulliNB
            modelclf = BernoulliNB(alpha=alfa)
        from sklearn.naive_bayes import GaussianNB


        #LinearSVC using SVC clasifier
        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                            ('chi',  SelectKBest(chi2, k=10000)),('clf',modelclf)])

        #Fitting the model with the training data
        model = pipeline.fit(x_train, y_train)
        filename = 'savedmodels/naiev_articles_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naiev_articles_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/naiev_articles_model.sav', 'rb'))
        user_prediction=loaded_model.predict([user_input])
        return user_prediction

#*****************************************************************************
#*****************************************************************************
class logistic_text(object):
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        #words will be stemed
        try:
            stemer= SnowballStemmer('english')
            STOPWORDS = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stemer= SnowballStemmer('english')
        #stopwords will be removed from articles
            STOPWORDS = set(stopwords.words('english'))
                
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) 
        # lowercase text
        text = text.lower()     
        # delete stopwords from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
        
        # removes any words =< 2 or >= 15 letters
        text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 15))

        # Stemming the words
        text = ' '.join([stemer.stem(word) for word in text.split()])
        return text    


    def load_articles(self):

        f = open ('data/articles.json', "r")
        # Reading from file
        data = json.loads(f.read())
        # READING FILE CONTENT INTO PANDAS DATA FRAME
        articles = pd.DataFrame(data).sample(250)
        #CLOSING FILE
        f.close()
        
        return articles

        
   
       
    def train_test(self,articles):
        articles['cleaned_body'] = articles['body'].apply(self.clean_text)
        # cleaning articles content in a new column "cleaned body"
        # separting input and output 
        artcat=articles['category']
        #drop category column from the data frame
        articles=articles.drop(columns='category')
        X_train, x_test, y_train, y_test = train_test_split(articles['cleaned_body'],artcat,test_size=0.2,train_size=0.8)
        return X_train,y_train,x_test,y_test
   
   
        

    def fit_model(self,x_train,y_train,Clf,Solver):

        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression(penalty=Clf,solver=Solver)

        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                            ('chi',  SelectKBest(chi2, k=10000)),
                            ('clf',LogisticRegression(penalty=Clf,solver=Solver))])

        #Fitting the model with the training data
        model = pipeline.fit(x_train, y_train)
        filename = 'savedmodels/logistic_articles_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/logistic_articles_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/logistic_articles_model.sav', 'rb'))
        user_prediction=loaded_model.predict([user_input])
        return user_prediction