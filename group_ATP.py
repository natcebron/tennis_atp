import streamlit as st
import streamlit.components.v1 as components
import os                      #+Deployment
import inspect                 #+Deployment
#importing all the necessary libraries
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from PIL import Image, ImageStat
import matplotlib.image as mpimg
import glob
import numpy as np
from matplotlib import pyplot as plt

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# MODEL PREDICTION")

####################
# IMPORTATION DU MODEL 
####################
    concatenated = pd.read_csv("concatenated.csv")
    list_of_names = concatenated['player_1'].to_list()
    list_of_names2 = concatenated['player_2'].to_list()
    final_list = list_of_names + list_of_names2
    final_list = list(dict.fromkeys(final_list))

    lst = list(range(0,len(final_list)))
    fruit_dictionary = dict(zip(final_list, lst))



    df2=concatenated.dropna(axis=0)
    df2.drop(['Tournament'], axis=1,inplace=True)
    df2 = df2.drop(['player_1_PS', 'player_2_PS','player_1_Max','player_2_Max','player_1_Pts','player_2_Pts'],axis=1)
    df2.drop(['player_1_sets','player_2_sets'], axis=1,inplace=True)
    df2.drop(['Date'], axis=1,inplace=True)
    df2 = df2.iloc[: , 1:]

    df2=df2.replace({"player_1": fruit_dictionary})
    df2=df2.replace({"player_2": fruit_dictionary})
    df2["Round"].replace({"1st Round": "1",
                         "2nd Round": "2",
                         "3rd Round": "3",
                         "4th Round" : "4",
                         "Quarterfinals": "5",
                         "Semifinals": "6",
                         "The Final": "7",
                         "Round Robin":"8"}, inplace=True)

    df2["Surface"].replace({"Clay": "1",
                         "Grass": "2",
                         "Hard": "3"}, inplace=True)

    df2["Court"].replace({"Outdoor": "1",
                         "Indoor": "2"}, inplace=True)

    df2["Series"].replace({"ATP250": "1",
                         "ATP500": "2",
                         "Masters 1000": "3",
                         "Masters Cup": "4",
                         "Grand Slam": "5"}, inplace=True)


    y = df2.target
    df2.drop(['target'], axis=1,inplace=True)
    df2.drop(['player_1','player_2'], axis=1,inplace=True)



    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size = 0.2, random_state = None,shuffle=False)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix
    from sklearn.linear_model import LogisticRegression


    # param_grid = {'n_neighbors': np.arange(1, 20),
    #           'metric': ['euclidean', 'manhattan']}

    # grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

    # grid.fit(X_train, y_train)
    # st.write(grid.best_score_)
    # st.write(grid.best_params_)

    #params_lr = {'solver': ['liblinear', 'lbfgs'], 'C': [10**(i) for i in range(-4, 3)]}

    #gridcv = GridSearchCV(clf_lr, param_grid=params_lr, scoring='accuracy', cv=3) 
    #gridcv.fit(X_train, y_train)

    #gf = pd.DataFrame(gridcv.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
    #st.dataframe(gf)

    #####
    # KNN
    #####

    # Instanciation du modèle
    knn = KNeighborsClassifier(n_neighbors = 19,metric='manhattan')

    # Entraînement du modèle sur le jeu d'entraînement
    knn.fit(X_train, y_train)
    y_pred_test_knn = knn.predict(X_test)

    # Calcul de l'accuracy, precision et rappel
    (VN, FP), (FN, VP) = confusion_matrix(y_test, y_pred_test_knn)
    n = len(y_test)
    st.write('KNN')

    st.write((VP + VN) / n)

    ##################
    # LOGIC REGRESSION
    ##################


    clf_lr = LogisticRegression(max_iter=1000,solver='lbfgs',C=100)
    clf_lr.fit(X_train, y_train)
    # Prédiction sur les données de test
    y_pred_test_clf_lr = clf_lr.predict(X_test)
    # Calcul de l'accuracy, precision et rappel
    (VN, FP), (FN, VP) = confusion_matrix(y_test, y_pred_test_clf_lr)
    n = len(y_test)
    st.write('Logic regression')
    st.write((VP + VN) / n)


    ###############
    # RANDOM FOREST
    ###############
    from sklearn.ensemble import RandomForestClassifier

    # rfc=RandomForestClassifier(random_state=42)
    # param_grid = { 
    # 'n_estimators': [200, 500],
    # 'max_features': ['auto', 'sqrt', 'log2'],
    # 'max_depth' : [4,5,6,7,8],
    #'criterion' :['gini', 'entropy']
    #}

    #CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    #CV_rfc.fit(X_train, y_train)
    #gf = pd.DataFrame(CV_rfc.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
    #st.dataframe(gf)

    rfc=RandomForestClassifier(n_estimators=200,max_features="log2",max_depth=5,criterion="entropy",random_state=42)
    rfc.fit(X_train, y_train)

    y_pred_test_CV_rfc = rfc.predict(X_test)
    # Calcul de l'accuracy, precision et rappel
    (VN, FP), (FN, VP) = confusion_matrix(y_test, y_pred_test_CV_rfc)
    n = len(y_test)
    st.write('Random Forest')
    st.write((VP + VN) / n)



    ##########
    # ADABOOST
    ##########
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV

    # param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
    #           "base_estimator__splitter" :   ["best", "random"],
    #          "n_estimators": [1, 2]
    #         }


    # DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)
    # ABC = AdaBoostClassifier(base_estimator = DTC)

    # run grid search
    # grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
    # grid_search_ABC.fit(X_train, y_train)
    # gf = pd.DataFrame(grid_search_ABC.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
    # st.dataframe(gf)

    DTC = DecisionTreeClassifier(criterion='entropy',splitter='random',random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)
    ABC = AdaBoostClassifier(base_estimator = DTC,n_estimators=2)
    ABC.fit(X_train, y_train)
    y_pred_test_ABC = ABC.predict(X_test)
    # Calcul de l'accuracy, precision et rappel
    (VN, FP), (FN, VP) = confusion_matrix(y_test, y_pred_test_CV_rfc)
    n = len(y_test)
    st.write('Decision tree')
    st.write((VP + VN) / n)
    ############
    # SVM
    ############

    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
    # fitting the model for grid search
    grid.fit(X_train, y_train)
    gf = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
    st.dataframe(gf)

    ###################
    # GRADIENT BOOSTING
    ###################

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report
    from sklearn.grid_search import GridSearchCV


    baseline = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10)
    baseline.fit(X_train,y_train)
    y_pred_test_CV_rfc = baseline.predict(X_test)

    (VN, FP), (FN, VP) = confusion_matrix(y_test, y_pred_test_CV_rfc)
    n = len(y_test)
    st.write('Gradient boosting')
    st.write((VP + VN) / n)

    ###################
    # VOTING CLASSIFIER
    ###################



    ############
    # PREDICTION
    ############

    st.markdown('## PREDICTION')
    df_v3 = pd.read_csv("df_v3.csv")
    df_v3['Date'] = pd.to_datetime(df_v3['Date'])
    df_v3 = df_v3.sort_values(by='Date') 
    P1_list = list(df_v3['player'].unique())
    P2_list = list(df_v3['Round'].unique())
    P3_list = list(df_v3['Surface'].unique())
    P4_list = list(df_v3['Series'].unique())
    P5_list = list(df_v3['Court'].unique())
    P6_list = list(df_v3['Tournament'].unique())

    col1,col2,col3 = st.columns(3)
    with col1:
        P1 = st.selectbox(label = "Player 1", options = P1_list)
        player_1_Rank = st.text_input('player_1_Rank')
        player_1_B365 = st.text_input('player_1_B365')
        P6 = st.selectbox(label = "Court", options = P5_list)

    with col2:
        P2 = st.selectbox(label = "Player 2", options = P1_list)
        player_2_Rank = st.text_input('player_2_Rank')
        player_2_B365 = st.text_input('player_2_B365')
        P7 = st.selectbox(label = "Tournament", options = P6_list)

    with col3:
        P3 = st.selectbox(label = "Round", options = P2_list)
        P4 = st.selectbox(label = "Surface", options = P3_list)
        P5 = st.selectbox(label = "Series", options = P4_list)

    with st.form(key='my_form_to_submit'):

        submit_button = st.form_submit_button(label='Submit')
    if submit_button:


        dfv4 = df_v3[['Tournament', 'ATP']] 
        dict2 = dfv4.set_index('Tournament').to_dict()['ATP']


        data = [{'Tournament':P7,'Player_1': P1, 'Player_2': P2, 'Round': P3,'Surface':P4,'Series':P5,
            'player_1_Rank':player_1_Rank,'player_2_Rank':player_2_Rank,'player_1_B365':player_1_B365,'player_2_B365':player_2_B365,'Court':P6}]
        dataf = pd.DataFrame(data)
        dataf['Best of'] = np.where(dataf['Series']=="Grand Slam", '5', '3')
        dataf=dataf.replace({"Tournament": dict2})

        dataf["Round"].replace({"1st Round": "1",
                         "2nd Round": "2",
                         "3rd Round": "3",
                         "4th Round" : "4",
                         "Quarterfinals": "5",
                         "Semifinals": "6",
                         "The Final": "7",
                         "Round Robin":"8"}, inplace=True)

        dataf["Surface"].replace({"Clay": "1",
                         "Grass": "2",
                         "Hard": "3"}, inplace=True)

        dataf["Court"].replace({"Outdoor": "1",
                         "Indoor": "2"}, inplace=True)

        dataf["Series"].replace({"ATP250": "1",
                         "ATP500": "2",
                         "Masters 1000": "3",
                         "Masters Cup": "4",
                         "Grand Slam": "5"}, inplace=True)

        concatenated = pd.read_csv("concatenated.csv")
        list_of_names = concatenated['player_1'].to_list()
        list_of_names2 = concatenated['player_2'].to_list()
        final_list = list_of_names + list_of_names2
        final_list = list(dict.fromkeys(final_list))

        lst = list(range(0,len(final_list)))
        fruit_dictionary = dict(zip(final_list, lst))

        dataf=dataf.replace({"Player_1": fruit_dictionary})
        dataf=dataf.replace({"Player_2": fruit_dictionary})




        dataf = dataf.drop(['Player_1', 'Player_2'],axis=1)






        y_pred_test_knn = knn.predict(dataf)

        if y_pred_test_knn==1:
            st.markdown('### Player 1 gagnant')
        else:
            st.markdown('### Player 2 gagnant')







