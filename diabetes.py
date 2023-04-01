import pandas as pd
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pickle

class diabetes_logistic(object):
    def load_data(self):
        df= pd.read_csv('data/diabetes.csv')
        return df


class diabetes_knn(object):

    def load_data(self):
        df= pd.read_csv('data/diabetes.csv')
        return df

    def train_test(self,df):
            
        df1= self.scaledata(df)
        x=df1.iloc[:,:8]
        y=df1.iloc[:,8]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        print (x_test.shape)
        return    x_train,x_test,y_train,y_test
    
      
       
    def fit_model(seelf,X_train,y_train,no_neigbors,algorithm,leaf_size):
        model = KNeighborsClassifier(n_neighbors=no_neigbors,algorithm=algorithm,leaf_size=leaf_size).fit(X_train,y_train)
        filename = 'savedmodels/knn_diabetes_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        
    def test_model(self,x_test,y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/knn_diabetes_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/knn_diabetes_model.sav', 'rb'))
        prediction=loaded_model.predict(user_input)
        return str(prediction)
     

    def scaledata(self,df):
        mms = MinMaxScaler() # Normalization
        ss = StandardScaler() # Standardization
        scaled_df=df.copy()
        scaled_df['Pregnancies'] = mms.fit_transform(df[['Pregnancies']])
        scaled_df['Insulin'] = ss.fit_transform(df[['Insulin']])
        scaled_df['DiabetesPedigreeFunction'] = ss.fit_transform(df[['DiabetesPedigreeFunction']])
        scaled_df['Age'] = ss.fit_transform(df[['Age']])
        scaled_df['BloodPressure'] = ss.fit_transform(df[['BloodPressure']])
        scaled_df['SkinThickness'] = ss.fit_transform(df[['SkinThickness']])
        scaled_df['Glucose'] = ss.fit_transform(df[['Glucose']])
        scaled_df['BMI'] = ss.fit_transform(df[['BMI']])
        
        return scaled_df

    def user_input_features_diabetes(self,df,colname1):
        Pregnancies = colname1.slider('Pregnancies', 0, 17, 1)/17
        Glucose = colname1.slider('Glucose', 0, 200, 1)
        Glucose=(Glucose - df['Glucose'].mean())/df['Glucose'].std()
        BloodPressure = colname1.slider('BloodPressure', 0, 150, 1)
        BloodPressure=(BloodPressure - df['BloodPressure'].mean())/df['BloodPressure'].std()
        SkinThickness = colname1.slider('SkinThickness',0,100, 1)
        SkinThickness=(SkinThickness - df['SkinThickness'].mean())/df['SkinThickness'].std()
        Insulin = colname1.slider('Insulin', 0, 900, 1)
        SkinThickness=(Insulin - df['Insulin'].mean())/df['Insulin'].std()
        BMI = colname1.slider('BMI', 0.0, 80.0, 0.01)
        BMI=(BMI - df['BMI'].mean())/df['BMI'].std()

        DiabetesPedigreeFunction = colname1.slider('DiabetesPedigreeFunction', 0.005, 3.0, 0.001)
        DiabetesPedigreeFunction=(DiabetesPedigreeFunction - df['DiabetesPedigreeFunction'].mean())/df['DiabetesPedigreeFunction'].std()

        Age = colname1.slider('Age', 20, 85, 1)
        Age=(BMI - df['Age'].mean())/df['Age'].std()


        data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age      
            }
        features = pd.DataFrame(data, index=[0])
        return features


class diabetes_dctree(object):

    def load_data(self):
        df= pd.read_csv('data/diabetes.csv')
        return df

    def train_test(self,df):
            
        df1= self.scaledata(df)
        x=df1.iloc[:,:8]
        y=df1.iloc[:,8]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        print (x_test.shape)
        return    x_train,x_test,y_train,y_test
    
      
       
    def fit_model(self,x_train,y_train,criteration,maxdepth,maxfeature,nsamplsplit):
        
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion=criteration,max_depth=maxdepth,
                                            max_features=maxfeature ,
                                            min_samples_split=nsamplsplit ).fit(x_train,y_train)
        
        filename = 'savedmodels/dctree_diabetes_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        
    def test_model(self,x_test,y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/dctree_diabetes_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/dctree_diabetes_model.sav', 'rb'))
        prediction=loaded_model.predict(user_input)
        return str(prediction)
     

    def scaledata(self,df):
        mms = MinMaxScaler() # Normalization
        ss = StandardScaler() # Standardization
        scaled_df=df.copy()
        scaled_df['Pregnancies'] = mms.fit_transform(df[['Pregnancies']])
        scaled_df['Insulin'] = ss.fit_transform(df[['Insulin']])
        scaled_df['DiabetesPedigreeFunction'] = ss.fit_transform(df[['DiabetesPedigreeFunction']])
        scaled_df['Age'] = ss.fit_transform(df[['Age']])
        scaled_df['BloodPressure'] = ss.fit_transform(df[['BloodPressure']])
        scaled_df['SkinThickness'] = ss.fit_transform(df[['SkinThickness']])
        scaled_df['Glucose'] = ss.fit_transform(df[['Glucose']])
        scaled_df['BMI'] = ss.fit_transform(df[['BMI']])
        
        return scaled_df

    def user_input_features_diabetes(self,df,colname1):
        Pregnancies = colname1.slider('Pregnancies', 0, 17, 1)/17
        Glucose = colname1.slider('Glucose', 0, 200, 1)
        Glucose=(Glucose - df['Glucose'].mean())/df['Glucose'].std()
        BloodPressure = colname1.slider('BloodPressure', 0, 150, 1)
        BloodPressure=(BloodPressure - df['BloodPressure'].mean())/df['BloodPressure'].std()
        SkinThickness = colname1.slider('SkinThickness',0,100, 1)
        SkinThickness=(SkinThickness - df['SkinThickness'].mean())/df['SkinThickness'].std()
        Insulin = colname1.slider('Insulin', 0, 900, 1)
        SkinThickness=(Insulin - df['Insulin'].mean())/df['Insulin'].std()
        BMI = colname1.slider('BMI', 0.0, 80.0, 0.01)
        BMI=(BMI - df['BMI'].mean())/df['BMI'].std()

        DiabetesPedigreeFunction = colname1.slider('DiabetesPedigreeFunction', 0.005, 3.0, 0.001)
        DiabetesPedigreeFunction=(DiabetesPedigreeFunction - df['DiabetesPedigreeFunction'].mean())/df['DiabetesPedigreeFunction'].std()

        Age = colname1.slider('Age', 20, 85, 1)
        Age=(BMI - df['Age'].mean())/df['Age'].std()


        data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age      
            }
        features = pd.DataFrame(data, index=[0])
        return features

class diabetes_naiev(object):

    def load_data(self):
        df= pd.read_csv('data/diabetes.csv')
        return df

    def train_test(self,df):
            
        df1= self.scaledata(df)
        x=df1.iloc[:,:8]
        y=df1.iloc[:,8]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        print (x_test.shape)
        return    x_train,x_test,y_train,y_test
    
      
       
    def fit_model(seelf,X_train,y_train,naive_algorithm,alfa):

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

        model.fit(X_train,y_train)
        filename = 'savedmodels/naive_diabetes_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        
    def test_model(self,x_test,y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/naive_diabetes_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/knn_diabetes_model.sav', 'rb'))
        prediction=loaded_model.predict(user_input)
        return str(prediction)
     

    def scaledata(self,df):
        mms = MinMaxScaler() # Normalization
        ss = StandardScaler() # Standardization
        scaled_df=df.copy()
        scaled_df['Pregnancies'] = mms.fit_transform(df[['Pregnancies']])
        scaled_df['Insulin'] = ss.fit_transform(df[['Insulin']])
        scaled_df['DiabetesPedigreeFunction'] = ss.fit_transform(df[['DiabetesPedigreeFunction']])
        scaled_df['Age'] = ss.fit_transform(df[['Age']])
        scaled_df['BloodPressure'] = ss.fit_transform(df[['BloodPressure']])
        scaled_df['SkinThickness'] = ss.fit_transform(df[['SkinThickness']])
        scaled_df['Glucose'] = ss.fit_transform(df[['Glucose']])
        scaled_df['BMI'] = ss.fit_transform(df[['BMI']])
        
        return scaled_df

    def user_input_features_diabetes(self,df,colname1):
        Pregnancies = colname1.slider('Pregnancies', 0, 17, 1)/17
        Glucose = colname1.slider('Glucose', 0, 200, 1)
        Glucose=(Glucose - df['Glucose'].mean())/df['Glucose'].std()
        BloodPressure = colname1.slider('BloodPressure', 0, 150, 1)
        BloodPressure=(BloodPressure - df['BloodPressure'].mean())/df['BloodPressure'].std()
        SkinThickness = colname1.slider('SkinThickness',0,100, 1)
        SkinThickness=(SkinThickness - df['SkinThickness'].mean())/df['SkinThickness'].std()
        Insulin = colname1.slider('Insulin', 0, 900, 1)
        SkinThickness=(Insulin - df['Insulin'].mean())/df['Insulin'].std()
        BMI = colname1.slider('BMI', 0.0, 80.0, 0.01)
        BMI=(BMI - df['BMI'].mean())/df['BMI'].std()

        DiabetesPedigreeFunction = colname1.slider('DiabetesPedigreeFunction', 0.005, 3.0, 0.001)
        DiabetesPedigreeFunction=(DiabetesPedigreeFunction - df['DiabetesPedigreeFunction'].mean())/df['DiabetesPedigreeFunction'].std()

        Age = colname1.slider('Age', 20, 85, 1)
        Age=(BMI - df['Age'].mean())/df['Age'].std()


        data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age      
            }
        features = pd.DataFrame(data, index=[0])
        return features


class diabetes_svm(object):

    def load_data(self):
        df= pd.read_csv('data/diabetes.csv')
        return df

    def train_test(self,df):
            
        df1= self.scaledata(df)
        x=df1.iloc[:,:8]
        y=df1.iloc[:,8]
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
        print (x_test.shape)
        return    x_train,x_test,y_train,y_test
    
      
       
    def fit_model(self,x_train,y_train):
        from sklearn.svm import SVC
        model = SVC(gamma='auto')
        model.fit(x_train, y_train)
        filename = 'savedmodels/svm_diabetes_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        
    def test_model(self,x_test,y_test ):
        from sklearn.metrics import accuracy_score
        loaded_model = pickle.load(open('savedmodels/svm_diabetes_model.sav', 'rb'))
        y_pred=loaded_model.predict(x_test)
        return str(accuracy_score(y_pred,y_test)*100)

    def predict_input(self,user_input):
        loaded_model = pickle.load(open('savedmodels/svm_diabetes_model.sav', 'rb'))
        prediction=loaded_model.predict(user_input)
        return str(prediction)
     

    def scaledata(self,df):
        mms = MinMaxScaler() # Normalization
        ss = StandardScaler() # Standardization
        scaled_df=df.copy()
        scaled_df['Pregnancies'] = mms.fit_transform(df[['Pregnancies']])
        scaled_df['Insulin'] = ss.fit_transform(df[['Insulin']])
        scaled_df['DiabetesPedigreeFunction'] = ss.fit_transform(df[['DiabetesPedigreeFunction']])
        scaled_df['Age'] = ss.fit_transform(df[['Age']])
        scaled_df['BloodPressure'] = ss.fit_transform(df[['BloodPressure']])
        scaled_df['SkinThickness'] = ss.fit_transform(df[['SkinThickness']])
        scaled_df['Glucose'] = ss.fit_transform(df[['Glucose']])
        scaled_df['BMI'] = ss.fit_transform(df[['BMI']])
        
        return scaled_df

    def user_input_features_diabetes(self,df,colname1):
        Pregnancies = colname1.slider('Pregnancies', 0, 17, 1)/17
        Glucose = colname1.slider('Glucose', 0, 200, 1)
        Glucose=(Glucose - df['Glucose'].mean())/df['Glucose'].std()
        BloodPressure = colname1.slider('BloodPressure', 0, 150, 1)
        BloodPressure=(BloodPressure - df['BloodPressure'].mean())/df['BloodPressure'].std()
        SkinThickness = colname1.slider('SkinThickness',0,100, 1)
        SkinThickness=(SkinThickness - df['SkinThickness'].mean())/df['SkinThickness'].std()
        Insulin = colname1.slider('Insulin', 0, 900, 1)
        SkinThickness=(Insulin - df['Insulin'].mean())/df['Insulin'].std()
        BMI = colname1.slider('BMI', 0.0, 80.0, 0.01)
        BMI=(BMI - df['BMI'].mean())/df['BMI'].std()

        DiabetesPedigreeFunction = colname1.slider('DiabetesPedigreeFunction', 0.005, 3.0, 0.001)
        DiabetesPedigreeFunction=(DiabetesPedigreeFunction - df['DiabetesPedigreeFunction'].mean())/df['DiabetesPedigreeFunction'].std()

        Age = colname1.slider('Age', 20, 85, 1)
        Age=(BMI - df['Age'].mean())/df['Age'].std()


        data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age      
            }
        features = pd.DataFrame(data, index=[0])
        return features