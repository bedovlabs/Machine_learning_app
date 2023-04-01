import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from Page_layout import main_page

class smoke_naive(object):
   
    def load_data(self):
        from scipy import stats
        originaldf=pd.read_csv("data/smoke_detection.csv", index_col=0)
        smoke_data=originaldf[(np.abs(stats.zscore(originaldf)) < 3).all(axis=1)]
        corr_matrix = smoke_data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features 
        smoke_data.drop(to_drop, axis=1, inplace=True)
        return smoke_data

    def train_test(self,smoke_data):
       
        x = smoke_data.drop(columns=["Fire Alarm","UTC"], axis = 1).values
        y = smoke_data['Fire Alarm'].values 
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        mx = MinMaxScaler()
        X_train = mx.fit_transform(X_train)
        X_test = mx.transform(X_test)
        return X_train,X_test,Y_train,Y_test 
        
    def fit_model(self,X_train, Y_train,naive_algorithm,alfa):
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

         
        model.fit(X_train, Y_train)
        filename = 'savedmodels/naive_smoke_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naive_smoke_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/naive_smoke_model.sav', 'rb'))
        new_fire_predicion=loaded_model.predict(user_input)
        return str(new_fire_predicion)
    
    def user_input_smoke(self,colname1,colname2,df):
        #Temperature[C],Humidity[%],TVOC[ppb], eCO2[ppm],Raw H2,Raw Ethanol,Pressure[hPa],PM1.0,PM2.5,NC0.5,NC1.0,NC2.5,CNT,Fire Alarm
        
        Temperature= colname1.slider('Temperature[C]', -22, 50, 1)
        Temperature=(Temperature - df['Temperature[C]'].mean())/df['Temperature[C]'].std()

        Humidity = colname1.slider('Humidity[%]', 10, 75, 1)
        Humidity=(Humidity - df['Humidity[%]'].mean())/df['Humidity[%]'].std()

        Tvoc = colname1.slider('TVOC[ppb]', 0, 60000, 1000)
        Tvoc=(Tvoc - df['TVOC[ppb]'].mean())/df['TVOC[ppb]'].std()

        eCO2 = colname1.slider('eCO2[ppm]',400,60000, 1000)
        eCO2=(eCO2- df['eCO2[ppm]'].mean())/df['eCO2[ppm]'].std()

        Raw_H2 = colname1.slider('Raw H2', 10700, 14000, 500)
        Raw_H2=(Raw_H2 - df['Raw H2'].mean())/df['Raw H2'].std()

        main_page.alignh(2,colname2)


        Raw_Ethanol = colname2.slider('Raw Ethanol', 15400, 21500, 500)
        Raw_Ethanol=(Raw_Ethanol - df['Raw Ethanol'].mean())/df['Raw Ethanol'].std()

        Pressure = colname2.slider('Pressure[hPa]', 930.0, 940.0, .1)
        Pressure=(Pressure - df['Pressure[hPa]'].mean())/df['Pressure[hPa]'].std()
        PM_1 = colname2.slider('PM1.0', 0, 14500,250)
        PM_1=(PM_1 - df['PM1.0'].mean())/df['PM1.0'].std()
        
        NC_2 = colname2.slider('NC2.5', 0, 31000, 300)
        NC_2=(NC_2 - df['NC2.5'].mean())/df['NC2.5'].std()

        CNT = colname2.slider('CNT', 0,25000, 300)
        CNT=(CNT - df['CNT'].mean())/df['CNT'].std()

        data = {'Temperature[C]': Temperature,
                'Humidity[%]': Humidity,
                'TVOC[ppb]': Tvoc,
                'eCO2[ppm]': eCO2,
                'Raw H2': Raw_H2,
                'Raw Ethanol': Raw_Ethanol,
                'Pressure[hPa]': Pressure,
                'PM1.0': PM_1,
                'NC2.5': NC_2,
                'CNT': CNT
              }
            
            
        features = pd.DataFrame(data, index=[0])
        return features

#*********************************************************************************************

class smoke_Dctree(object):
   
    def load_data(self):
        from scipy import stats
        originaldf=pd.read_csv("data/smoke_detection.csv", index_col=0)
        smoke_data=originaldf[(np.abs(stats.zscore(originaldf)) < 3).all(axis=1)]
        corr_matrix = smoke_data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features 
        smoke_data.drop(to_drop, axis=1, inplace=True)
        return smoke_data

    def train_test(self,smoke_data):
       
        x = smoke_data.drop(columns=["Fire Alarm","UTC"], axis = 1).values
        y = smoke_data['Fire Alarm'].values 
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        mx = MinMaxScaler()
        X_train = mx.fit_transform(X_train)
        X_test = mx.transform(X_test)
        return X_train,X_test,Y_train,Y_test 
        
    def fit_model(self,x_train,y_train,criteration,maxdepth,maxfeature,nsamplsplit):
        
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion=criteration,max_depth=maxdepth,
                                            max_features=maxfeature ,
                                            min_samples_split=nsamplsplit ).fit(x_train,y_train)

        filename = 'savedmodels/dec_tree_smoke_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data
    

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/dec_tree_smoke_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/dec_tree_smoke_model.sav', 'rb'))
        new_fire_predicion=loaded_model.predict(user_input)
        return str(new_fire_predicion)
    
    def user_input_smoke(self,colname1,colname2,df):
        #Temperature[C],Humidity[%],TVOC[ppb], eCO2[ppm],Raw H2,Raw Ethanol,Pressure[hPa],PM1.0,PM2.5,NC0.5,NC1.0,NC2.5,CNT,Fire Alarm
        
        Temperature= colname1.slider('Temperature[C]', -22, 50, 1)
        Temperature=(Temperature - df['Temperature[C]'].mean())/df['Temperature[C]'].std()

        Humidity = colname1.slider('Humidity[%]', 10, 75, 1)
        Humidity=(Humidity - df['Humidity[%]'].mean())/df['Humidity[%]'].std()

        Tvoc = colname1.slider('TVOC[ppb]', 0, 60000, 1000)
        Tvoc=(Tvoc - df['TVOC[ppb]'].mean())/df['TVOC[ppb]'].std()

        eCO2 = colname1.slider('eCO2[ppm]',400,60000, 1000)
        eCO2=(eCO2- df['eCO2[ppm]'].mean())/df['eCO2[ppm]'].std()

        Raw_H2 = colname1.slider('Raw H2', 10700, 14000, 500)
        Raw_H2=(Raw_H2 - df['Raw H2'].mean())/df['Raw H2'].std()

        main_page.alignh(2,colname2)


        Raw_Ethanol = colname2.slider('Raw Ethanol', 15400, 21500, 500)
        Raw_Ethanol=(Raw_Ethanol - df['Raw Ethanol'].mean())/df['Raw Ethanol'].std()

        Pressure = colname2.slider('Pressure[hPa]', 930.0, 940.0, .1)
        Pressure=(Pressure - df['Pressure[hPa]'].mean())/df['Pressure[hPa]'].std()
        PM_1 = colname2.slider('PM1.0', 0, 14500,250)
        PM_1=(PM_1 - df['PM1.0'].mean())/df['PM1.0'].std()
        
        NC_2 = colname2.slider('NC2.5', 0, 31000, 300)
        NC_2=(NC_2 - df['NC2.5'].mean())/df['NC2.5'].std()

        CNT = colname2.slider('CNT', 0,25000, 300)
        CNT=(CNT - df['CNT'].mean())/df['CNT'].std()

        data = {'Temperature[C]': Temperature,
                'Humidity[%]': Humidity,
                'TVOC[ppb]': Tvoc,
                'eCO2[ppm]': eCO2,
                'Raw H2': Raw_H2,
                'Raw Ethanol': Raw_Ethanol,
                'Pressure[hPa]': Pressure,
                'PM1.0': PM_1,
                'NC2.5': NC_2,
                'CNT': CNT
              }
            
            
        features = pd.DataFrame(data, index=[0])
        return features

#*********************************************************************************************

class smoke_logitsic(object):
   
    def load_data(self):
        from scipy import stats
        originaldf=pd.read_csv("data/smoke_detection.csv", index_col=0)
        smoke_data=originaldf[(np.abs(stats.zscore(originaldf)) < 3).all(axis=1)]
        corr_matrix = smoke_data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features 
        smoke_data.drop(to_drop, axis=1, inplace=True)
        return smoke_data

    def train_test(self,smoke_data):
       
        x = smoke_data.drop(columns=["Fire Alarm","UTC"], axis = 1).values
        y = smoke_data['Fire Alarm'].values 
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        mx = MinMaxScaler()
        X_train = mx.fit_transform(X_train)
        X_test = mx.transform(X_test)
        return X_train,X_test,Y_train,Y_test 
        
    def fit_model(self,x_train,x_test,y_train,y_test):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(x_train, y_train)
        filename = 'savedmodels/logistic_smoke_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/logistic_smoke_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/logistic_smoke_model.sav', 'rb'))
        new_fire_predicion=loaded_model.predict(user_input)
        return str(new_fire_predicion)
    
    def user_input_smoke(self,colname1,colname2,df):
        #Temperature[C],Humidity[%],TVOC[ppb], eCO2[ppm],Raw H2,Raw Ethanol,Pressure[hPa],PM1.0,PM2.5,NC0.5,NC1.0,NC2.5,CNT,Fire Alarm
        
        Temperature= colname1.slider('Temperature[C]', -22, 50, 1)
        Temperature=(Temperature - df['Temperature[C]'].mean())/df['Temperature[C]'].std()

        Humidity = colname1.slider('Humidity[%]', 10, 75, 1)
        Humidity=(Humidity - df['Humidity[%]'].mean())/df['Humidity[%]'].std()

        Tvoc = colname1.slider('TVOC[ppb]', 0, 60000, 1000)
        Tvoc=(Tvoc - df['TVOC[ppb]'].mean())/df['TVOC[ppb]'].std()

        eCO2 = colname1.slider('eCO2[ppm]',400,60000, 1000)
        eCO2=(eCO2- df['eCO2[ppm]'].mean())/df['eCO2[ppm]'].std()

        Raw_H2 = colname1.slider('Raw H2', 10700, 14000, 500)
        Raw_H2=(Raw_H2 - df['Raw H2'].mean())/df['Raw H2'].std()

        main_page.alignh(2,colname2)


        Raw_Ethanol = colname2.slider('Raw Ethanol', 15400, 21500, 500)
        Raw_Ethanol=(Raw_Ethanol - df['Raw Ethanol'].mean())/df['Raw Ethanol'].std()

        Pressure = colname2.slider('Pressure[hPa]', 930.0, 940.0, .1)
        Pressure=(Pressure - df['Pressure[hPa]'].mean())/df['Pressure[hPa]'].std()
        PM_1 = colname2.slider('PM1.0', 0, 14500,250)
        PM_1=(PM_1 - df['PM1.0'].mean())/df['PM1.0'].std()
        
        NC_2 = colname2.slider('NC2.5', 0, 31000, 300)
        NC_2=(NC_2 - df['NC2.5'].mean())/df['NC2.5'].std()

        CNT = colname2.slider('CNT', 0,25000, 300)
        CNT=(CNT - df['CNT'].mean())/df['CNT'].std()

        data = {'Temperature[C]': Temperature,
                'Humidity[%]': Humidity,
                'TVOC[ppb]': Tvoc,
                'eCO2[ppm]': eCO2,
                'Raw H2': Raw_H2,
                'Raw Ethanol': Raw_Ethanol,
                'Pressure[hPa]': Pressure,
                'PM1.0': PM_1,
                'NC2.5': NC_2,
                'CNT': CNT
              }
            
            
        features = pd.DataFrame(data, index=[0])
        return features


#*********************************************************************************************

class smoke_knn(object):
   
    def load_data(self):
        from scipy import stats
        originaldf=pd.read_csv("data/smoke_detection.csv", index_col=0)
        smoke_data=originaldf[(np.abs(stats.zscore(originaldf)) < 3).all(axis=1)]
        corr_matrix = smoke_data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features 
        smoke_data.drop(to_drop, axis=1, inplace=True)
        return smoke_data

    def train_test(self,smoke_data):
       
        x = smoke_data.drop(columns=["Fire Alarm","UTC"], axis = 1).values
        y = smoke_data['Fire Alarm'].values 
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        mx = MinMaxScaler()
        X_train = mx.fit_transform(X_train)
        X_test = mx.transform(X_test)
        return X_train,X_test,Y_train,Y_test 
        
    def fit_model(self,x_train,y_train,no_neigbors,algorithm,leaf_size):
        from sklearn.neighbors import KNeighborsClassifier
        
        model = KNeighborsClassifier(n_neighbors=no_neigbors,algorithm=algorithm,leaf_size=leaf_size).fit(x_train,y_train)

        filename = 'savedmodels/knn_irr_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/knn_smoke_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/knn_smoke_model.sav', 'rb'))
        new_fire_predicion=loaded_model.predict(user_input)
        return str(new_fire_predicion)
    
    def user_input_smoke(self,colname1,colname2,df):
        #Temperature[C],Humidity[%],TVOC[ppb], eCO2[ppm],Raw H2,Raw Ethanol,Pressure[hPa],PM1.0,PM2.5,NC0.5,NC1.0,NC2.5,CNT,Fire Alarm
        
        Temperature= colname1.slider('Temperature[C]', -22, 50, 1)
        Temperature=(Temperature - df['Temperature[C]'].mean())/df['Temperature[C]'].std()

        Humidity = colname1.slider('Humidity[%]', 10, 75, 1)
        Humidity=(Humidity - df['Humidity[%]'].mean())/df['Humidity[%]'].std()

        Tvoc = colname1.slider('TVOC[ppb]', 0, 60000, 1000)
        Tvoc=(Tvoc - df['TVOC[ppb]'].mean())/df['TVOC[ppb]'].std()

        eCO2 = colname1.slider('eCO2[ppm]',400,60000, 1000)
        eCO2=(eCO2- df['eCO2[ppm]'].mean())/df['eCO2[ppm]'].std()

        Raw_H2 = colname1.slider('Raw H2', 10700, 14000, 500)
        Raw_H2=(Raw_H2 - df['Raw H2'].mean())/df['Raw H2'].std()

        main_page.alignh(2,colname2)


        Raw_Ethanol = colname2.slider('Raw Ethanol', 15400, 21500, 500)
        Raw_Ethanol=(Raw_Ethanol - df['Raw Ethanol'].mean())/df['Raw Ethanol'].std()

        Pressure = colname2.slider('Pressure[hPa]', 930.0, 940.0, .1)
        Pressure=(Pressure - df['Pressure[hPa]'].mean())/df['Pressure[hPa]'].std()
        PM_1 = colname2.slider('PM1.0', 0, 14500,250)
        PM_1=(PM_1 - df['PM1.0'].mean())/df['PM1.0'].std()
        
        NC_2 = colname2.slider('NC2.5', 0, 31000, 300)
        NC_2=(NC_2 - df['NC2.5'].mean())/df['NC2.5'].std()

        CNT = colname2.slider('CNT', 0,25000, 300)
        CNT=(CNT - df['CNT'].mean())/df['CNT'].std()

        data = {'Temperature[C]': Temperature,
                'Humidity[%]': Humidity,
                'TVOC[ppb]': Tvoc,
                'eCO2[ppm]': eCO2,
                'Raw H2': Raw_H2,
                'Raw Ethanol': Raw_Ethanol,
                'Pressure[hPa]': Pressure,
                'PM1.0': PM_1,
                'NC2.5': NC_2,
                'CNT': CNT
              }
            
            
        features = pd.DataFrame(data, index=[0])
        return features


#*********************************************************************************************
class smoke_svm(object):
   
    def load_data(self):
        from scipy import stats
        originaldf=pd.read_csv("data/smoke_detection.csv", index_col=0)
        smoke_data=originaldf[(np.abs(stats.zscore(originaldf)) < 3).all(axis=1)]
        corr_matrix = smoke_data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features 
        smoke_data.drop(to_drop, axis=1, inplace=True)
        return smoke_data

    def train_test(self,smoke_data):
       
        x = smoke_data.drop(columns=["Fire Alarm","UTC"], axis = 1).values
        y = smoke_data['Fire Alarm'].values 
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        mx = MinMaxScaler()
        X_train = mx.fit_transform(X_train)
        X_test = mx.transform(X_test)
        return X_train,X_test,Y_train,Y_test 
        
    def fit_model(self,x_train,y_train):
        from sklearn.svm import SVC
        model = SVC(gamma='auto')
        model.fit(x_train, y_train)
        filename = 'savedmodels/svm_smoke_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        # showing  model accuracy score with validating and testing data

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/svm_smoke_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/svm_smoke_model.sav', 'rb'))
        new_fire_predicion=loaded_model.predict(user_input)
        return str(new_fire_predicion)
    
    def user_input_smoke(self,colname1,colname2,df):
        #Temperature[C],Humidity[%],TVOC[ppb], eCO2[ppm],Raw H2,Raw Ethanol,Pressure[hPa],PM1.0,PM2.5,NC0.5,NC1.0,NC2.5,CNT,Fire Alarm
        
        Temperature= colname1.slider('Temperature[C]', -22, 50, 1)
        Temperature=(Temperature - df['Temperature[C]'].mean())/df['Temperature[C]'].std()

        Humidity = colname1.slider('Humidity[%]', 10, 75, 1)
        Humidity=(Humidity - df['Humidity[%]'].mean())/df['Humidity[%]'].std()

        Tvoc = colname1.slider('TVOC[ppb]', 0, 60000, 1000)
        Tvoc=(Tvoc - df['TVOC[ppb]'].mean())/df['TVOC[ppb]'].std()

        eCO2 = colname1.slider('eCO2[ppm]',400,60000, 1000)
        eCO2=(eCO2- df['eCO2[ppm]'].mean())/df['eCO2[ppm]'].std()

        Raw_H2 = colname1.slider('Raw H2', 10700, 14000, 500)
        Raw_H2=(Raw_H2 - df['Raw H2'].mean())/df['Raw H2'].std()

        main_page.alignh(2,colname2)


        Raw_Ethanol = colname2.slider('Raw Ethanol', 15400, 21500, 500)
        Raw_Ethanol=(Raw_Ethanol - df['Raw Ethanol'].mean())/df['Raw Ethanol'].std()

        Pressure = colname2.slider('Pressure[hPa]', 930.0, 940.0, .1)
        Pressure=(Pressure - df['Pressure[hPa]'].mean())/df['Pressure[hPa]'].std()
        PM_1 = colname2.slider('PM1.0', 0, 14500,250)
        PM_1=(PM_1 - df['PM1.0'].mean())/df['PM1.0'].std()
        
        NC_2 = colname2.slider('NC2.5', 0, 31000, 300)
        NC_2=(NC_2 - df['NC2.5'].mean())/df['NC2.5'].std()

        CNT = colname2.slider('CNT', 0,25000, 300)
        CNT=(CNT - df['CNT'].mean())/df['CNT'].std()

        data = {'Temperature[C]': Temperature,
                'Humidity[%]': Humidity,
                'TVOC[ppb]': Tvoc,
                'eCO2[ppm]': eCO2,
                'Raw H2': Raw_H2,
                'Raw Ethanol': Raw_Ethanol,
                'Pressure[hPa]': Pressure,
                'PM1.0': PM_1,
                'NC2.5': NC_2,
                'CNT': CNT
              }
         
        features = pd.DataFrame(data, index=[0])
        return features


#*********************************************************************************************
class smoke_kmean(object):
   
    def load_data(self):
        from scipy import stats
        originaldf=pd.read_csv("data/smoke_detection.csv", index_col=0)
        smoke_data=originaldf[(np.abs(stats.zscore(originaldf)) < 3).all(axis=1)]
        corr_matrix = smoke_data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features 
        smoke_data.drop(to_drop, axis=1, inplace=True)
        return smoke_data

    def train_test(self,smoke_data):
       
        x = smoke_data.drop(columns=["Fire Alarm","UTC"], axis = 1).values
        y = smoke_data['Fire Alarm'].values 
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        mx = MinMaxScaler()
        X_train = mx.fit_transform(X_train)
        X_test = mx.transform(X_test)
        return X_train,X_test,Y_train,Y_test 
        
    def fit_model(self,X_train, Y_train,naive_algorithm,alfa):
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

         
        model.fit(X_train, Y_train)
        filename = 'savedmodels/naive_smoke_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def test_model(self,X_test,Y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naive_smoke_model.sav', 'rb'))
        y_pred=loaded_model.predict(X_test)
        return str(accuracy_score(y_pred,Y_test)*100)
    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/naive_smoke_model.sav', 'rb'))
        new_fire_predicion=loaded_model.predict(user_input)
        return str(new_fire_predicion)
    
    def user_input_smoke(self,colname1,colname2,df):
        #Temperature[C],Humidity[%],TVOC[ppb], eCO2[ppm],Raw H2,Raw Ethanol,Pressure[hPa],PM1.0,PM2.5,NC0.5,NC1.0,NC2.5,CNT,Fire Alarm
        
        Temperature= colname1.slider('Temperature[C]', -22, 50, 1)
        Temperature=(Temperature - df['Temperature[C]'].mean())/df['Temperature[C]'].std()

        Humidity = colname1.slider('Humidity[%]', 10, 75, 1)
        Humidity=(Humidity - df['Humidity[%]'].mean())/df['Humidity[%]'].std()

        Tvoc = colname1.slider('TVOC[ppb]', 0, 60000, 1000)
        Tvoc=(Tvoc - df['TVOC[ppb]'].mean())/df['TVOC[ppb]'].std()

        eCO2 = colname1.slider('eCO2[ppm]',400,60000, 1000)
        eCO2=(eCO2- df['eCO2[ppm]'].mean())/df['eCO2[ppm]'].std()

        Raw_H2 = colname1.slider('Raw H2', 10700, 14000, 500)
        Raw_H2=(Raw_H2 - df['Raw H2'].mean())/df['Raw H2'].std()

        main_page.alignh(2,colname2)


        Raw_Ethanol = colname2.slider('Raw Ethanol', 15400, 21500, 500)
        Raw_Ethanol=(Raw_Ethanol - df['Raw Ethanol'].mean())/df['Raw Ethanol'].std()

        Pressure = colname2.slider('Pressure[hPa]', 930.0, 940.0, .1)
        Pressure=(Pressure - df['Pressure[hPa]'].mean())/df['Pressure[hPa]'].std()
        PM_1 = colname2.slider('PM1.0', 0, 14500,250)
        PM_1=(PM_1 - df['PM1.0'].mean())/df['PM1.0'].std()
        
        NC_2 = colname2.slider('NC2.5', 0, 31000, 300)
        NC_2=(NC_2 - df['NC2.5'].mean())/df['NC2.5'].std()

        CNT = colname2.slider('CNT', 0,25000, 300)
        CNT=(CNT - df['CNT'].mean())/df['CNT'].std()

        data = {'Temperature[C]': Temperature,
                'Humidity[%]': Humidity,
                'TVOC[ppb]': Tvoc,
                'eCO2[ppm]': eCO2,
                'Raw H2': Raw_H2,
                'Raw Ethanol': Raw_Ethanol,
                'Pressure[hPa]': Pressure,
                'PM1.0': PM_1,
                'NC2.5': NC_2,
                'CNT': CNT
              }
            
            
        features = pd.DataFrame(data, index=[0])
        return features