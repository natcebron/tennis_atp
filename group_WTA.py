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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

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
    df = pd.read_csv("WTA/df.csv")
    df_v3 = pd.read_csv("WTA/df_v3.csv")
    list_of_names = df['player_1'].to_list()
    list_of_names2 = df['player_2'].to_list()
    final_list = list_of_names + list_of_names2
    final_list = list(dict.fromkeys(final_list))

    lst = list(range(0,len(final_list)))
    fruit_dictionary = dict(zip(final_list, lst))

    df = df.sort_values(by='Date_x') 
    df = df.drop(df[(df['Tier'] == 'WTA251') | (df['Tier'] == 'WTA252')| (df['Tier'] == 'WTA254') | (df['Tier'] == 'WTA255')| (df['Tier'] == 'WTA256')| (df['Tier'] == 'WTA258')| (df['Tier'] == 'WTA259')| (df['Tier'] == 'WTA261')| (df['Tier'] == 'WTA263')| (df['Tier'] == 'WTA264')| (df['Tier'] == 'WTA265')| (df['Tier'] == 'WTA266')| (df['Tier'] == 'WTA268')| (df['Tier'] == 'WTA269')| (df['Tier'] == 'WTA271')| (df['Tier'] == 'WTA272')| (df['Tier'] == 'WTA273')| (df['Tier'] == 'WTA275')].index)


    df["Surface"].replace({"Clay": "1",
                         "Grass": "2",
                         "Hard": "3"}, inplace=True)

    df["Court"].replace({"Outdoor": "1",
                         "Indoor": "2"}, inplace=True)

    df["Tier"].replace({"International": "1",
                         "Premier": "2",
                         "WTA1000": "3",
                         "WTA500": "4",
                         "Grand Slam": "5",
                           "WTA250":'6',
                   "Tour Championships":"7"}, inplace=True)
    #df = df.drop(['WTA', 'Location','Tournament','Date_x','second_sets','first_sets','score','minutes','second_ace',
    #          'second_df','second_svpt','second_1stIn','second_1stWon','second_2ndWon','second_SvGms','second_bpSaved',
    #         'second_bpFaced','first_ace','first_df','first_svpt','first_1stIn','first_1stWon','first_2ndWon','first_SvGms',
    #          'first_bpSaved','first_bpFaced','Surface','Round'],axis=1)

    df["hand_1"].replace({"R": "1",
                         "L": "2",'U':'3'}, inplace=True)
    df["hand_2"].replace({"R": "1",
                         "L": "2",'U':'3'}, inplace=True)
    list_of_names = df['player_1'].to_list()
    list_of_names2 = df['player_2'].to_list()
    final_list = list_of_names + list_of_names2
    final_list = list(dict.fromkeys(final_list))

    lst = list(range(0,len(final_list)))
    fruit_dictionary = dict(zip(final_list, lst))

    df=df.replace({"player_1": fruit_dictionary})
    df=df.replace({"player_2": fruit_dictionary})


    df = df.drop(df[(df['surface'] == 'Carpet')].index)

    df=df.dropna(axis=0)
    df = df.iloc[: , 1:]
    df2 = df.copy()

    y = df.target
    df.drop(['target'], axis=1,inplace=True)
    #df.drop(['first_age','second_age'], axis=1,inplace=True)
    df.drop(['hand_1','hand_2','Round'], axis=1,inplace=True)
    df.drop(['Best of','Date_x','tourney_name','Location','sets_1','sets_2','surface','score','minutes'], axis=1,inplace=True)
    df = df.drop(['1_ace','1_df','1_svpt','1_1stIn','1_1stWon','1_2ndWon','1_SvGms','1_bpSaved','1_bpFaced','2_ace','2_df','2_svpt','2_1stIn','2_1stWon',
    '2_2ndWon','2_SvGms','2_bpSaved','2_bpFaced','WTA'],axis=1)

    #######
    # MODEL
    #######

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = None,shuffle=False)
    #rfc=RandomForestClassifier(n_estimators=500,max_features="sqrt",max_depth=8,criterion="gini",random_state=42)
    #rfc.fit(X_train, y_train)
    #y_pred_test = rfc.predict(X_test)
    #st.write(accuracy_score(y_test, y_pred_test))
    #joblib.dump(rfc, "WTA/WTA.joblib")
    loaded_rf = joblib.load("WTA/WTA.joblib")

    test20 = loaded_rf.predict_proba(X_test)
    test20=pd.DataFrame(test20)
    test20.columns = ['prono_1', 'prono_2']
    df2 = df2.iloc[-len(X_test):]


    odds_3 = pd.read_table('ATP/test5.csv',sep=',')
    test6 = pd.read_table('ATP/test6.csv',sep=',')

    list_of_names = odds_3['player'].to_list()
    list_of_names2 = odds_3['rank_2022'].to_list()
    final_list = list(dict.fromkeys(list_of_names))
    lst = list(range(0,len(final_list)))

    fruit_dictionary = dict(zip(list_of_names, lst))
    fruit_dictionary10 = dict(zip(lst, final_list))
    df2 = df2.reset_index()

    result = pd.concat([ test20,df2], axis=1,ignore_index=False)
    result=result.replace({"player_1": fruit_dictionary10})
    result=result.replace({"player_2": fruit_dictionary10})

        #result = result.drop(['league_name','league_id','first_age','second_age','first_rank','second_rank','category','userName','surface','Court','series', 'Best of','first_Pts','second_Pts','first_hand','second_hand'],axis=1)
    result = result[['player_1','player_2','target','prono_1','prono_2','B365_1','B365_2']]
    m=10
    result['diff'] = result['prono_1'] - result['prono_2']

    result['points'] = np.where( ( (result['target'] == 1) & (result['prono_1'] > result['prono_2'] ) ) | ( (result['target'] == 2) & (result['prono_1'] < result['prono_2'] ) ) , 'bp', 'mp')
    result['choice'] = np.where(result['prono_1'] >= result['prono_2'], result['B365_1'], result['B365_2'])
    result['ROI'] = np.where(result['points'] == 'bp', (result['choice']*m), (result['choice']*-m))
    result.loc[result['ROI'] < 10, 'ROI'] = -10
    rslt_df = result[(result['diff'] > 0.5) |
          (result['diff'] < -0.5)]
    from datetime import date
    from datetime import datetime
    st.dataframe(result)


    def diff_month(d1, d2):
        return (d2.year - d1.year) * 12 + d2.month - d1.month   
    month2 = diff_month(datetime(2019,8,7), datetime(2022,6,24))


    # initialize list of lists
    data = [['ROI global', result['ROI'].sum(),len(result['ROI']),result['ROI'].sum()/len(result['ROI']),result['ROI'].sum()/month2],
            ['ROI select (diff cote >0.5)', rslt_df['ROI'].sum(),len(rslt_df['ROI']),rslt_df['ROI'].sum()/len(rslt_df['ROI']),rslt_df['ROI'].sum()/month2]]
    # Create the pandas DataFrame
    st.markdown('Gain pour une mise de 10 euros par match')

    d_ROI = pd.DataFrame(data, columns=['ROI', 'Somme','Nb match','ROI par match','Gain par mois'])
    d_ROI['month'] = month2
    # print dataframe.
    st.dataframe(d_ROI)


    ############
    # PREDICTION
    ############
    st.markdown('## PREDICTION')
    df_v3 = pd.read_csv("WTA/df_v3.csv")
    df_v3['Date_x'] = pd.to_datetime(df_v3['Date_x'])
    df_v3 = df_v3.sort_values(by='Date_x') 
    P1_list = list(df_v3['player'].unique())
    P2_list = list(df_v3['Round'].unique())
    P3_list = list(df_v3['Surface'].unique())
    P4_list = list(df_v3['Tier'].unique())
    P5_list = list(df_v3['Court'].unique())
    P6_list = list(df_v3['tourney_name'].unique())

    col1,col2,col3 = st.columns(3)
    with col1:
        P1 = st.selectbox(label = "Player 1", options = P1_list)
        player_1_B365 = st.text_input('player_1_B365')
        P6 = st.selectbox(label = "Court", options = P5_list)

    with col2:
        P2 = st.selectbox(label = "Player 2", options = P1_list)
        player_2_B365 = st.text_input('player_2_B365')
        # P7 = st.selectbox(label = "Tournament", options = P6_list)

    with col3:
        P3 = st.selectbox(label = "Round", options = P2_list)
        P4 = st.selectbox(label = "Surface", options = P3_list)
        P5 = st.selectbox(label = "Tier", options = P4_list)

    with st.form(key='my_form_to_submit'):

        submit_button = st.form_submit_button(label='Submit')
    if submit_button:


        dfv4 = df_v3[['tourney_name', 'WTA']] 
        dict2 = dfv4.set_index('tourney_name').to_dict()['WTA']


        data = [{'Tier':P5,
                'Court':P6,
                'Bestof':P5,
                'player_1':P1,
                '1_hand':P1,
                'Pts_1':P1,
                'Rank_1':P1,
                'age_1':P1,
                'B365_1':player_1_B365,
                'B365_2':player_2_B365,
                'Surface':P4,
                '2_hand':P2,
                'age_2':P2,
                'player_2':P2,
                'Rank_2':P2,
                'Pts_2':P2
                }]
        
        dataf = pd.DataFrame(data)
        dataf['Bestof'] = np.where(dataf['Tier']=="Grand Slam", '5', '3')
        dataf=dataf.replace({"Tournament": dict2})


        dataf["Surface"].replace({"Clay": "1",
                         "Grass": "2",
                         "Hard": "3"}, inplace=True)

        dataf["Court"].replace({"Outdoor": "1",
                         "Indoor": "2"}, inplace=True)

        dataf["Tier"].replace({"WTA250": "1",
                         "WTA500": "2",
                         "Masters 1000": "3",
                         "Masters Cup": "4",
                         "Grand Slam": "5",
                         "International":'6'}, inplace=True)

        concatenated = pd.read_csv("WTA/df.csv")
        list_of_names = concatenated['player_1'].to_list()
        list_of_names2 = concatenated['player_2'].to_list()
        final_list = list_of_names + list_of_names2
        final_list = list(dict.fromkeys(final_list))

        lst = list(range(0,len(final_list)))
        fruit_dictionary = dict(zip(final_list, lst))

        dataf=dataf.replace({"player_1": fruit_dictionary})
        dataf=dataf.replace({"player_2": fruit_dictionary})


        test5 = pd.read_csv("WTA/test5.csv")
        test6 = pd.read_csv("WTA/test6.csv")

        list_of_names = test5['player'].to_list()
        list_of_names2 = test5['rank_2022'].to_list()
        final_list = list(dict.fromkeys(list_of_names))

        fruit_dictionary10 = dict(zip(final_list, list_of_names2))
        dataf=dataf.replace({"Rank_1": fruit_dictionary10})
        dataf=dataf.replace({"Rank_2": fruit_dictionary10})
        #dataf = dataf.drop(['player_1_B365', 'player_2_B365'],axis=1)

        list_of_names5 = test5['points'].to_list()
        final_list = list(dict.fromkeys(list_of_names))

        fruit_dictionary11 = dict(zip(final_list, list_of_names5))
        dataf=dataf.replace({"Pts_1": fruit_dictionary11})
        dataf=dataf.replace({"Pts_2": fruit_dictionary11})

        list_of_names5 = test5['hand'].to_list()
        fruit_dictionary11 = dict(zip(final_list, list_of_names5))
        dataf=dataf.replace({"1_hand": fruit_dictionary11})
        dataf=dataf.replace({"2_hand": fruit_dictionary11})


        list_of_names5 = test6['Age'].to_list()
        list_of_names = test6['player'].to_list()

        final_list = list(dict.fromkeys(list_of_names))

        fruit_dictionary11 = dict(zip(final_list, list_of_names5))

        dataf["1_hand"].replace({"R": "1",
                         "L": "2",'U':'3'}, inplace=True)
        dataf["2_hand"].replace({"R": "1",
                         "L": "2",'U':'3'}, inplace=True)
        dataf=dataf.replace({"age_1": fruit_dictionary11})
        dataf=dataf.replace({"age_2": fruit_dictionary11})
        dataf.drop(['1_hand','2_hand','Bestof'], axis=1,inplace=True)
        dataf = dataf[['Tier','Court','Surface','player_1','player_2','Rank_1','Rank_2','Pts_1','Pts_2','B365_1','B365_2','age_1','age_2']]
        st.dataframe(dataf)


        if loaded_rf.predict_proba(dataf)[:,0]>0.5:
            st.markdown('### Player 1 gagnant')
            st.write(loaded_rf.predict_proba(dataf)[:,0])

        else:
            st.markdown('### Player 2 gagnant')
            st.write(loaded_rf.predict_proba(dataf)[:,1])







