import pandas as pd
from requests import options
import seaborn as sns
from sklearn import svm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

class svm_irr(object):
    def load_data(self):
        irr_df=pd.read_csv('data/irrigation.csv')
        return irr_df
    def train_test(self,irr_df):
        le = LabelEncoder()
        le.fit(irr_df['CropType'])
        irr_df['CropType']=le.transform(irr_df['CropType'])
        x=irr_df.iloc[:,:5]
        y=irr_df.iloc[:,5]
        scaledx=MinMaxScaler()
        x=scaledx.fit_transform(x)
        print(y.shape)
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train):
        from sklearn.svm import SVC
        model = SVC(gamma='auto')
        model.fit(x_train, y_train)
        filename = 'savedmodels/svmirr_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/svmirr_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/svmirr_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
    def user_input_smart_irr(self,irr_col1,irr_col2):
       
        options=['Garden Flowers' , 'Maize' , 'Paddy,Paddy','Garden Flowers'  ,'Wheat','Groundnuts','Potato','Groundnuts','Sugarcane']
        CropType =irr_col1.selectbox('CropType', options=options)
        CropType= (options.index(CropType)+1)/9
        CropDays = irr_col1.slider('CropDays', 1, 210, 1)
        CropDays=CropDays/210
        SoilMoisture = irr_col1.slider('SoilMoisture', 120, 990, 1)
        SoilMoisture=SoilMoisture /990
        temperature = irr_col1.slider('temperature',10,263, 1)
        temperature=temperature/263 
        Humidity = irr_col1.slider('Humidity', 10, 85, 1)
        Humidity=Humidity /85
        data = {'CropType': CropType,
            'CropDays': CropDays,
            'SoilMoisture': SoilMoisture,
            'temperature': temperature,
            'Humidity': Humidity,
            
            }
        features = pd.DataFrame(data, index=[0])
        return features





class logistic_irrg(object):
    def load_data(self):
        irr_df=pd.read_csv('data/irrigation.csv')
        return irr_df
    def train_test(self,irr_df):
        
        le = LabelEncoder()
        le.fit(irr_df['CropType'])
        irr_df['CropType']=le.transform(irr_df['CropType'])
        x=irr_df.iloc[:,:5]
        y=irr_df.iloc[:,5]
        scaledx=MinMaxScaler()
        x=scaledx.fit_transform(x)
        print(y.shape)
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,x_test,y_train,y_test):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(x_train, y_train)
        filename = 'savedmodels/logisticirrg_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/logisticirrg_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/logisticirrg_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
  
  
    def user_input_smart_irr(self,irr_col1,irr_col2):
       
        options=['Garden Flowers' , 'Maize' , 'Paddy,Paddy','Garden Flowers'  ,'Wheat','Groundnuts','Potato','Groundnuts','Sugarcane']
        CropType =irr_col2.selectbox('CropType', options=options)
        CropType= (options.index(CropType)+1)/9
        CropDays = irr_col2.slider('CropDays', 1, 210, 1)
        CropDays=CropDays/210
        SoilMoisture = irr_col2.slider('SoilMoisture', 120, 990, 1)
        SoilMoisture=SoilMoisture /990
        temperature = irr_col2.slider('temperature',10,263, 1)
        temperature=temperature/263 
        Humidity = irr_col2.slider('Humidity', 10, 85, 1)
        Humidity=Humidity /85
        data = {'CropType': CropType,
            'CropDays': CropDays,
            'SoilMoisture': SoilMoisture,
            'temperature': temperature,
            'Humidity': Humidity,
            
            }
        features = pd.DataFrame(data, index=[0])
        return features




class naive_irr(object):
    def load_data(self):
        irr_df=pd.read_csv('data/irrigation.csv')
        return irr_df
    def train_test(self,irr_df):
        
        le = LabelEncoder()
        le.fit(irr_df['CropType'])
        irr_df['CropType']=le.transform(irr_df['CropType'])
        x=irr_df.iloc[:,:5]
        y=irr_df.iloc[:,5]
        scaledx=MinMaxScaler()
        x=scaledx.fit_transform(x)
        print(y.shape)
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
        filename = 'savedmodels/naive_irr_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naive_irr_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/naive_irr_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction


    def user_input_smart_irr(self,irr_col1,irr_col2):
       
        options=['Garden Flowers' , 'Maize' , 'Paddy,Paddy','Garden Flowers'  ,'Wheat','Groundnuts','Potato','Groundnuts','Sugarcane']
        CropType =irr_col1.selectbox('CropType', options=options)
        CropType= (options.index(CropType)+1)/9
        CropDays = irr_col1.slider('CropDays', 1, 210, 1)
        CropDays=CropDays/210
        SoilMoisture = irr_col1.slider('SoilMoisture', 120, 990, 1)
        SoilMoisture=SoilMoisture /990
        temperature = irr_col1.slider('temperature',10,263, 1)
        temperature=temperature/263 
        Humidity = irr_col1.slider('Humidity', 10, 85, 1)
        Humidity=Humidity /85
        data = {'CropType': CropType,
            'CropDays': CropDays,
            'SoilMoisture': SoilMoisture,
            'temperature': temperature,
            'Humidity': Humidity,
            
            }
        features = pd.DataFrame(data, index=[0])
        return features

class dec_tree_irr(object):
    def load_data(self):
        irr_df=pd.read_csv('data/irrigation.csv')
        return irr_df
    def train_test(self,irr_df):
        
        le = LabelEncoder()
        le.fit(irr_df['CropType'])
        irr_df['CropType']=le.transform(irr_df['CropType'])
        x=irr_df.iloc[:,:5]
        y=irr_df.iloc[:,5]
        scaledx=MinMaxScaler()
        x=scaledx.fit_transform(x)
        print(y.shape)
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train,criteration,maxdepth,maxfeature,nsamplsplit):
        
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion=criteration,max_depth=maxdepth,
                                            max_features=maxfeature ,
                                            min_samples_split=nsamplsplit ).fit(x_train,y_train)

        filename = 'savedmodels/dec_tree_irr_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/dec_tree_irr_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/dec_tree_irr_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction
  
    def user_input_smart_irr(self,irr_col1,irr_col2):
       
        options=['Garden Flowers' , 'Maize' , 'Paddy,Paddy','Garden Flowers'  ,'Wheat','Groundnuts','Potato','Groundnuts','Sugarcane']
        CropType =irr_col1.selectbox('CropType', options=options)
        CropType= (options.index(CropType)+1)/9
        CropDays = irr_col1.slider('CropDays', 1, 210, 1)
        CropDays=CropDays/210
        SoilMoisture = irr_col1.slider('SoilMoisture', 120, 990, 1)
        SoilMoisture=SoilMoisture /990
        temperature = irr_col1.slider('temperature',10,263, 1)
        temperature=temperature/263 
        Humidity = irr_col1.slider('Humidity', 10, 85, 1)
        Humidity=Humidity /85
        data = {'CropType': CropType,
            'CropDays': CropDays,
            'SoilMoisture': SoilMoisture,
            'temperature': temperature,
            'Humidity': Humidity,
            
            }
        features = pd.DataFrame(data, index=[0])
        return features

class knn_irr(object):
    def load_data(self):
        irr_df=pd.read_csv('data/irrigation.csv')
        return irr_df
    def train_test(self,irr_df):
        
        le = LabelEncoder()
        le.fit(irr_df['CropType'])
        irr_df['CropType']=le.transform(irr_df['CropType'])
        x=irr_df.iloc[:,:5]
        y=irr_df.iloc[:,5]
        scaledx=MinMaxScaler()
        x=scaledx.fit_transform(x)
       # print(y.shape)
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        return x_train,x_test,y_train,y_test

    def fit_model(self,x_train,y_train,no_neigbors,algorithm,leaf_size):
        
        model = KNeighborsClassifier(n_neighbors=no_neigbors,algorithm=algorithm,leaf_size=leaf_size).fit(x_train,y_train)

        filename = 'savedmodels/knn_irr_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/knn_irr_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/knn_irr_model.sav', 'rb'))
        user_prediction=loaded_model.predict(user_input)
        return user_prediction


    def user_input_smart_irr(self,irr_col1,irr_col2):
       
        options=['Garden Flowers' , 'Maize' , 'Paddy,Paddy','Garden Flowers'  ,'Wheat','Groundnuts','Potato','Groundnuts','Sugarcane']
        CropType =irr_col1.selectbox('CropType', options=options)
        CropType= (options.index(CropType)+1)/9
        CropDays = irr_col1.slider('CropDays', 1, 210, 1)
        CropDays=CropDays/210
        SoilMoisture = irr_col1.slider('SoilMoisture', 120, 990, 1)
        SoilMoisture=SoilMoisture /990
        temperature = irr_col1.slider('temperature',10,263, 1)
        temperature=temperature/263 
        Humidity = irr_col1.slider('Humidity', 10, 85, 1)
        Humidity=Humidity /85
        data = {'CropType': CropType,
            'CropDays': CropDays,
            'SoilMoisture': SoilMoisture,
            'temperature': temperature,
            'Humidity': Humidity,
            
            }
        features = pd.DataFrame(data, index=[0])
        return features




