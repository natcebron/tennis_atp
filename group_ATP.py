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

#################################################
# IMPORTATION DU MODEL ET PREPARATION DES DONNEES 
#################################################
    
    df_v3 = pd.read_csv("ATP/df_v3.csv")
    import joblib
    valeur = 0
    df_merged2 =pd.read_csv('ATP/df_versus.csv')
    df=df_merged2


    df = df.rename(columns={"loser_Rank": "first_rank", 
                            "loser_Pts": "first_Pts", 
                            "loser_Sets": "first_sets",
                            "loser_Odds": "first_odds",
                            "loser_Hand":"first_hand",
                            "loser_Age":"first_age",
                            "loser_series":"first_series",
                            'loser_forme':'first_forme',
                            'loser_series_surf':'first_series_surf',
                            'loser_forme_surf':'first_forme_surf',



                            "winner_Rank": "second_rank", 
                            "winner_Pts": "second_Pts", 
                            "winner_Sets": "second_sets",
                            "winner_Odds": "second_odds",
                            "winner_Hand":"second_hand",
                            "winner_Age":"second_age",
                            "winner_series":"second_series",
                            'winner_forme':'second_forme',
                            'winner_series_surf':'second_series_surf',
                            'winner_forme_surf':'second_forme_surf'
                        },)

    import numpy as np
    copy_2_df = df.copy()
    copy_2_df[["first_rank","first_Pts","first_sets","first_odds","first_hand","first_age","first_series","first_forme",'first_series_surf',
            'first_forme_surf',
                            "second_rank","second_Pts","second_sets","second_odds",
                            "second_hand","second_age","second_series",'second_forme','second_series_surf','second_forme_surf']]\
    = copy_2_df[["second_rank","second_Pts","second_sets","second_odds","second_hand","second_age","second_series",'second_forme','second_series_surf',
                'second_forme_surf',
                            "first_rank",
                            "first_Pts","first_sets","first_odds","first_hand","first_age","first_series",'first_forme','first_series_surf',
                            'first_forme_surf']]
    #shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    first_player = []
    second_player = []
    labels = []


    for winner, looser in zip(df['winner_player'], df['loser_player']):
        number = np.random.choice([0,1],1)[0] #the number of the winner
        if number == 1: #the winner is player 0 and the loser is player 1 => label = 0
            first_player.append(winner)
            second_player.append(looser)

        else: #the loser is player 0 and the winner is player 1 => label = 1
            second_player.append(winner)
            first_player.append(looser)

        labels.append(number)
    df['first_player_id'] = first_player
    df['second_player_id'] = second_player
    df['label'] = labels
    df = df.sort_values(by=['Date_x_x'])
    pd.set_option('display.max_rows', 50)  # or 1000



    winner_player2 = np.zeros(df.shape[0]) # second player wins so label=0
    df['label'] = winner_player2


    winner_player1 = np.ones(copy_2_df.shape[0]) # first player wins so label=1
    copy_2_df['label'] = winner_player1 

    #df = pd.concat([df,copy_2_df])
    #shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    first_player = []
    second_player = []
    labels = []


    for winner, looser in zip(df['winner_player'], df['loser_player']):
        number = np.random.choice([0,1],1)[0] #the number of the winner
        if number == 1: #the winner is player 0 and the loser is player 1 => label = 0
            first_player.append(winner)
            second_player.append(looser)

        else: #the loser is player 0 and the winner is player 1 => label = 1
            second_player.append(winner)
            first_player.append(looser)

        labels.append(number)
    df['first_player_id'] = first_player
    df['second_player_id'] = second_player
    df['label'] = labels
    df = df.sort_values(by=['Date_x_x'])

    df
    df = df.drop(columns=['winner_player', 'loser_player'])
    df2 = df
    df2.to_csv('ATP/df2.csv')
    df = df.drop(['ATP_y',
    'Location_y',
    'Tournament_y',
    'Date_x_y',
    'Series_y',
    'Court_y',
    'Surface_y',
    'Round_y',
    'Best of_y','group_y','group_x'],axis=1)

    df
    df = df.sort_values(by='Date_x_x') 
    df["Round_x"].replace({"1st Round": "1",
                            "2nd Round": "2",
                            "3rd Round": "3",
                            "4th Round" : "4",
                            "Quarterfinals": "5",
                            "Semifinals": "6",
                            "The Final": "7",
                            "Round Robin":"8"}, inplace=True)

    df["Surface_x"].replace({"Clay": "1",
                            "Grass": "2",
                            "Hard": "3"}, inplace=True)

    df["Court_x"].replace({"Outdoor": "1",
                            "Indoor": "2"}, inplace=True)

    df["Series_x"].replace({"ATP250": "1",
                            "ATP500": "2",
                            "Masters 1000": "3",
                            "Masters Cup": "4",
                            "Grand Slam": "5"}, inplace=True)
    df = df.drop(['match_id','ATP_x', 'Location_x','Tournament_x','Date_x_x','second_sets'],axis=1)
    df = df.dropna(subset=['first_series', 'first_forme'])
    df["first_hand"].replace({"R": "1",
                            "L": "2",'U':'3'}, inplace=True)
    df["second_hand"].replace({"R": "1",
                            "L": "2",'U':'3'}, inplace=True)



    list_of_names = df['first_player_id'].to_list()
    list_of_names2 = df['second_player_id'].to_list()
    final_list = list_of_names + list_of_names2
    final_list = list(dict.fromkeys(final_list))

    lst = list(range(0,len(final_list)))
    fruit_dictionary = dict(zip(final_list, lst))

    df=df.replace({"first_player_id": fruit_dictionary})
    df=df.replace({"second_player_id": fruit_dictionary})
    df = df.dropna()
    df = df.drop(['Round_x','Best of_x','first_sets'],axis=1)
    df = df.iloc[: , 1:]

    df=df.dropna(axis=0)
    y = df.label
    df.drop(['label'], axis=1,inplace=True)
        #df2.drop(['player_1','player_2'], axis=1,inplace=True)



    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = None,shuffle=False)
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier

    rfc=RandomForestClassifier(n_estimators=500,max_features="auto",max_depth=8,criterion="gini",random_state=42)
    rfc.fit(X_train, y_train)
    y_pred_test = rfc.predict(X_test)
    print(accuracy_score(y_test, y_pred_test))
    data = df
        
    valeur = accuracy_score(y_test, y_pred_test)
    valeur2 = accuracy_score(y_test, y_pred_test)
    
    data = df
    rfc2 = rfc
    joblib.dump(rfc, "ATP/ATP.joblib")
    st.write(valeur2)
    
    loaded_rf = joblib.load("ATP/ATP.joblib")

    loaded_rf.fit(X_train, y_train)
    test30 = df2.copy()

    test20 = rfc.predict_proba(X_test)
    test20=pd.DataFrame(test20)
    test20.columns = ['player_1','player_2']
    test30 = test30.iloc[-len(X_test):]
    test30 = test30.reset_index()

    result = pd.concat([ test20,test30], axis=1,ignore_index=False)
    result = result[['Surface_x','player_1','player_2','Date_x_x','first_player_id','second_player_id','second_forme','first_forme','second_odds','first_odds','label']]
    result['points'] = np.where( ( ((result['label'] == 1) & (result['player_1'] > 0.5 )) | ((result['label'] == 0) & (result['player_2'] > 0.5))), 'bp', 'mp')


    conditions  = [ (result['player_1'] > result['player_2']) , (result['player_2'] > result['player_1'])]
    choices     = [ result['first_odds'], result['second_odds'] ]
        
    result["choice"] = np.select(conditions, choices, default=np.nan)
    m=10
    result['ROI'] = np.where(result['points'] == 'bp', (result['choice']*m-10), (result['choice']*-m-10))
    result.loc[result['ROI'] < 1, 'ROI'] = -10

    from datetime import datetime

    def diff_month(d1, d2):
            return (d2.year - d1.year) * 12 + d2.month - d1.month   
    month2 = diff_month(datetime(2020,2,24), datetime(2022,7,24))


        # initialize list of lists
    data = [['ROI global', result['ROI'].sum(),len(result['ROI']),result['ROI'].sum()/len(result['ROI']),result['ROI'].sum()/month2]]
        # Create the pandas DataFrame
    print('Gain pour une mise de 10 euros par match')

    d_ROI = pd.DataFrame(data, columns=['ROI', 'Somme','Nb match','ROI par match','Gain par mois'])
    d_ROI['month'] = month2
        # print dataframe.
    d_ROI

    pd.set_option('display.max_rows', 5)  # or 1000

    import numpy as np
    df_v1 = result.drop(['first_player_id','first_forme','player_1'],axis=1)
    df_v1['group'] = 'second'
    df_v2 = result.drop(['second_player_id','second_forme','player_2'],axis=1)
    df_v2['group'] = 'first'

    df_v2.rename(columns = {'first_player_id':'player',

                            'first_forme':'forme','player_1':'player_prono'}, inplace = True)

    df_v1.rename(columns = {'second_player_id':'player',

                            'second_forme':'forme','player_2':'player_prono'}, inplace = True)


    df_v10 = pd.concat([df_v1,df_v2])
    df_v10 = df_v10.reset_index()

    df_v10 = df_v10.sort_values(by='index') 
    df_v10 = df_v10.sort_values(by='ROI') 
    df_v10 = df_v10.dropna()
    import plotly.express as px
    df_v10["ROI"]=df_v10["ROI"].apply(int)
    test = df_v10.groupby(['player']).sum()
    pro2 = df_v10.groupby(['player']).sum()
    df_v10.to_csv('ATP/df_v10.csv')
    test = test.reset_index()

    pro2 = pro2.groupby(level=[0]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    test['percent'] = pro2['index']
    test = test.sort_values(by='ROI', ascending=False)

    st.markdown('## Top player ROI')
    top10_ROI = test[:10]
    st.dataframe(top10_ROI)

    fig = px.bar(top10_ROI, x="ROI", y="player",title="Top10 par player",color='player')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,barmode='relative')
    st.plotly_chart(fig)

    df_v10["ROI"]=df_v10["ROI"].apply(int)
    test = df_v10.groupby(['player','points']).sum()
    pro2 = df_v10.groupby(['player','points']).sum()

    test = test.reset_index()

    pro2 = pro2.groupby(level=[0]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    test['percent'] = pro2['index']
    st.dataframe(test)
    test2 = test.loc[test['points'] == 'bp']
    list_1 = list(test2['player'])
    list_2 = list(test2['percent'])
    fruit_dictionary = dict(zip(list_1, list_2))
    st.write(fruit_dictionary)

   







