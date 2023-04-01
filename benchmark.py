import streamlit as st
class benchmark_classifiers(object):
    def __init__(self):
        st.subheader("Select Algorithms")
        selected_algorithms=st.multiselect("Algorithms",options=['Logistic Regression','K Nearset Neighpors KNN','K Means','Support Vector Machine','Decession Tree','Naive Bayes'])
        st.subheader("Select Dataset")
        selected_dataset=st.selectbox("Dataset",options=['Smart Irrigation','Iris','Smoke detection',
                                                        'Diabetes','Diseses','Image classifier',
                                                        'Article classifier','Digit recognition'])
       
        benchmarker= st.button("Benchamrk Selected Algorithms")

        if benchmarker :
            if selected_dataset=='Smart Irrigation':
                self.load_smart_irr(selected_algorithms)
                pass
            elif selected_dataset=='Iris':
                self.load_Iris(selected_algorithms)
                pass
            elif selected_dataset=='Smoke detection':
                self.load_smoke(selected_algorithms)
                pass
            elif selected_dataset=='Diabetes':
                self.load_diabetes(selected_algorithms)
                pass
            elif selected_dataset=='Diseses':
                self.load_diseses(selected_algorithms)
                pass
            elif selected_dataset=='Image classifier':
                self.load_image(selected_algorithms)
                pass
            elif selected_dataset=='Article classifier':
                self.load_article(selected_algorithms)
                pass
            elif selected_dataset=='Digit recognition':
                self.load_digit(selected_algorithms)
                pass

#************************************************************************************************************
#************************************************************************************************************
#************************************************************************************************************


    def load_smart_irr(self,selected_algorithms):
        for i in range(0,len(selected_algorithms)):
                    #st.write(selected_algorithms[i])
                    if selected_algorithms[i]=='Logistic Regression':
                        from smart_irrigation import logistic_irrg
                        logirrbench=logistic_irrg()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Logistic regression accuracy is "+ acc )
                     
                    elif selected_algorithms[i]=='K Nearset Neighpors KNN':
                        from smart_irrigation import knn_irr
                        logirrbench=knn_irr()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("K Nearset Neighpors KNN accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Support Vector Machine':
                        from smart_irrigation import svm_irr
                        logirrbench=svm_irr()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Support Vector Machine accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Decession Tree':
                        from smart_irrigation import dec_tree_irr
                        logirrbench=logistic_irrg()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Decession Tree accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Naive Bayes':
                        from smart_irrigation import naive_irr
                        logirrbench=naive_irr()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Naive Bayes is accuracy "+ acc )

#************************************************************************************************************
   
    def load_Iris(self,selected_algorithms):
        for i in range(0,len(selected_algorithms)):
                    if selected_algorithms[i]=='Logistic Regression':
                        from iris import logistic_iris
                        logirrbench=logistic_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Logistic regression accuracy is "+ acc )
                     
                    elif selected_algorithms[i]=='K Nearset Neighpors KNN':
                        from iris import knn_iris
                        logirrbench=knn_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("K Nearset Neighpors KNN accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Support Vector Machine':
                        from iris import svm_iris
                        logirrbench=svm_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Support Vector Machine accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Decession Tree':
                        from iris import dec_tree_iris
                        logirrbench=dec_tree_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Decession Tree accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Naive Bayes':
                        from iris import naive_iris   
                        logirrbench=naive_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Naive Bayes is accuracy "+ acc )
#************************************************************************************************************
#************************************************************************************************************

    def load_smoke(self,selected_algorithms):
        for i in range(0,len(selected_algorithms)):
                    if selected_algorithms[i]=='Logistic Regression':
                        from iris import logistic_iris
                        logirrbench=logistic_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Logistic regression accuracy is "+ acc )
                     
                    elif selected_algorithms[i]=='K Nearset Neighpors KNN':
                        from iris import knn_iris
                        logirrbench=knn_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("K Nearset Neighpors KNN accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Support Vector Machine':
                        from iris import svm_iris
                        logirrbench=svm_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Support Vector Machine accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Decession Tree':
                        from iris import dec_tree_iris
                        logirrbench=dec_tree_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Decession Tree accuracy is "+ acc )
                        
                    elif selected_algorithms[i]=='Naive Bayes':
                        from iris import naive_iris   
                        logirrbench=naive_iris()
                        df=logirrbench.load_data()
                        x_train,x_test,y_train,y_test=logirrbench.train_test(df)
                        acc=logirrbench.test_model(x_test,y_test)
                        st.subheader("Naive Bayes is accuracy "+ acc )


    def load_diabetes(self,selected_algorithms):
        pass
    def load_diseses(self,selected_algorithms):
        pass
    def load_article(self,selected_algorithms):
        pass
    def load_image(self,selected_algorithms):
        pass
    def load_digit(self,selected_algorithms):
        pass