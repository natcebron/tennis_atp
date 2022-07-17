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
import plotly.express as px

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# PRONOSTIC DU JOUR")

    ######################
    # IMPORTATION DES ODDS
    ######################
    import requests
    url = "https://pinnacle-odds.p.rapidapi.com/kit/v1/markets"
    querystring = {"sport_id":"2","is_have_odds":"true"}
    headers = {
	  "X-RapidAPI-Key": "0dd58a2799mshf9ac25a9f307ee3p18d19fjsnd86178a061e3",
	  "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)

    df=pd.DataFrame(response.json())
    odds_1 = df['events'].apply(pd.Series)

    odds_1 = odds_1.drop(['sport_id','event_id','last','is_have_odds','period_results','parent_id','event_type'],axis=1)
    odds_1 = odds_1.drop(odds_1[(odds_1['resulting_unit'] == 'Sets')].index)
    odds_1 = odds_1.drop(odds_1[(odds_1['resulting_unit'] == 'Games')].index)

    odds_2 = odds_1['periods'].apply(pd.Series)
    odds_3 = odds_2['num_1'].apply(pd.Series)


    odds_4 = odds_2['num_0'].apply(pd.Series)
    odds_4 = odds_4.drop(['cutoff','line_id','number','period_status','spreads','team_total','totals','meta'],axis=1)
    odds_4 = odds_4['money_line'].apply(pd.Series)

    odds_5 = pd.concat([odds_1,odds_4],axis=1)
    odds_5 = odds_5.drop(['resulting_unit','periods','draw'],axis=1)
    odds_5.columns = ['league_id', 'league_name', 'starts', 'home_player','away_player','home_odds','away_odds']
    odds_5['last_name_home'] = odds_5['home_player'].str.split(' ').str[1]
    odds_5['last_name_away'] = odds_5['away_player'].str.split(' ').str[1]

    odds_5['first_name_home'] = odds_5['home_player'].str.split(' ').str[0]
    odds_5['first_name_away'] = odds_5['away_player'].str.split(' ').str[0]
    odds_5

    odds_5['first_name_home'] = odds_5['first_name_home'].str.split('').str[1]
    odds_5["home_player"] = odds_5['last_name_home'] +" "+ odds_5["first_name_home"]+"."
    odds_5['first_name_away'] = odds_5['first_name_away'].str.split('').str[1]
    odds_5["away_player"] = odds_5['last_name_away'] +" "+ odds_5["first_name_away"]+"."

    odds_5 = odds_5.drop(['last_name_home','last_name_away','first_name_home','first_name_away'],axis=1)
    odds_5["category"] = odds_5['league_name'].str.split(' ').str[0]
    df_merged = pd.read_csv("WTA/df_merged.csv")

    odds_5['starts'] = pd.to_datetime(odds_5['starts'])
    list_of_names = df_merged['tourney_name'].to_list()
    list_of_names2 = df_merged['Surface'].to_list()
    d_surface = dict(zip(list_of_names, list_of_names2))

    list_of_names = df_merged['tourney_name'].to_list()
    list_of_names2 = df_merged['Court'].to_list()
    d_court = dict(zip(list_of_names, list_of_names2))

    list_of_names = df_merged['tourney_name'].to_list()
    list_of_names2 = df_merged['Tier'].to_list()
    d_series = dict(zip(list_of_names, list_of_names2))


    a_2022 = pd.read_table('WTA/test6.csv',sep=',')
    a_2021 = pd.read_table('WTA/test5.csv',sep=',')

    data = pd.merge(a_2021, a_2022, on=['player'], how='inner')

    list_of_names5 = data['Age'].to_list()
    list_of_names = data['player'].to_list()
    final_list = list(dict.fromkeys(list_of_names))
    d_age = dict(zip(final_list, list_of_names5))


    list_of_names5 = data['hand'].to_list()
    list_of_names = data['player'].to_list()
    final_list = list(dict.fromkeys(list_of_names))
    d_hand = dict(zip(final_list, list_of_names5))

    list_of_names5 = data['points'].to_list()
    list_of_names = data['player'].to_list()
    final_list = list(dict.fromkeys(list_of_names))
    d_points = dict(zip(final_list, list_of_names5))

    list_of_names5 = data['rank_2022'].to_list()
    list_of_names = data['player'].to_list()
    final_list = list(dict.fromkeys(list_of_names))
    d_rank = dict(zip(final_list, list_of_names5))
    for real_name in df_merged['tourney_name'].to_list():
      odds_5.loc[ odds_5['league_name'].str.contains(real_name), 'userName' ] = real_name

    odds_5=odds_5.dropna(axis=0)
    odds_5 = odds_5[odds_5['category']=='WTA' ]

    odds_5['surface'] = odds_5['userName']
    odds_5=odds_5.replace({"surface": d_surface})
    odds_5['Court'] = odds_5['userName']
    odds_5=odds_5.replace({"Court": d_court})
    odds_5['series'] = odds_5['userName']
    odds_5=odds_5.replace({"series": d_series})
    odds_5['Best of'] = odds_5['series']
    odds_5['Best of'] = np.where(odds_5['series']=="Grand Slam", '5', '3')

    odds_5['home_player']= odds_5['home_player'].str.replace('-',' ')
    odds_5['away_player']= odds_5['away_player'].str.replace('-',' ')

    odds_5['home_player']= odds_5['home_player'].str.replace('Sorribes','Sorribes Tormo')
    odds_5['away_player']= odds_5['away_player'].str.replace('Sorribes','Sorribes Tormo')

    odds_5['first_rank'] = odds_5['home_player']
    odds_5['first_Pts'] = odds_5['home_player']
    odds_5['first_hand'] = odds_5['home_player']
    odds_5['first_age'] = odds_5['home_player']

    odds_5['second_rank'] = odds_5['away_player']
    odds_5['second_Pts'] = odds_5['away_player']
    odds_5['second_hand'] = odds_5['away_player']
    odds_5['second_age'] = odds_5['away_player']

    discard = ["Doubles",'Qualifiers']

    odds_5 = odds_5[~odds_5.league_name.str.contains('|'.join(discard))]

    odds_5=odds_5.replace({"first_rank": d_rank})
    odds_5=odds_5.replace({"first_Pts": d_points})
    odds_5=odds_5.replace({"first_hand": d_hand})
    odds_5=odds_5.replace({"first_age": d_age})

    odds_5=odds_5.replace({"second_rank": d_rank})
    odds_5=odds_5.replace({"second_Pts": d_points})
    odds_5=odds_5.replace({"second_hand": d_hand})
    odds_5=odds_5.replace({"second_age": d_age})

    odds_5["surface"].replace({"Clay": "1",
                         "Grass": "2",
                         "Hard": "3"}, inplace=True)

    odds_5["Court"].replace({"Outdoor": "1",
                         "Indoor": "2"}, inplace=True)

    odds_5["series"].replace({"WTA250": "1",
                         "WTA500": "2",
                         "Masters 1000": "3",
                         "Masters Cup": "4",
                         "Grand Slam": "5",
                         "International":'6','WTA275':'7'}, inplace=True)

    odds_3 = pd.read_table('WTA/test5.csv',sep=',')
    test6 = pd.read_table('WTA/test6.csv',sep=',')

    list_of_names = odds_3['player'].to_list()
    list_of_names2 = odds_3['rank_2022'].to_list()
    final_list = list(dict.fromkeys(list_of_names))
    lst = list(range(0,len(final_list)))

    fruit_dictionary = dict(zip(list_of_names, lst))
    fruit_dictionary10 = dict(zip(lst, final_list))
    
    odds_5["first_hand"].replace({"R": "1",
                         "L": "2",'U':'3'}, inplace=True)
    odds_5["second_hand"].replace({"R": "1",
                         "L": "2",'U':'3'}, inplace=True)
    odds_10 = odds_5

    odds_5=odds_5.replace({"home_player": fruit_dictionary})
    odds_5=odds_5.replace({"away_player": fruit_dictionary})
    odds_5['second_rank'] = pd.to_numeric(odds_5['second_rank'], errors = 'coerce')
    odds_5['first_rank'] = pd.to_numeric(odds_5['first_rank'], errors = 'coerce')
    odds_5['away_player'] = pd.to_numeric(odds_5['away_player'], errors = 'coerce')
    odds_5['home_player'] = pd.to_numeric(odds_5['home_player'], errors = 'coerce')
    inv_map = {v: k for k, v in fruit_dictionary.items()}

    
    odds_5.dropna(inplace = True)
    odds_5 = odds_5.drop_duplicates(['home_player'])    

    odds_5 = odds_5.drop(['userName','category','starts','league_name','league_id'],axis=1)


    odds_5.columns = ['player_1', 'player_2',  'B365_1','B365_2','Surface','Court','Series','Best of','Rank_1',
                      'Pts_1','hand_1','age_1','Rank_2','Pts_2','hand_2','age_2']
    odds_5 = odds_5.drop(['hand_1','hand_2'],axis=1)

    odds_5 = odds_5[['Series','Court','Surface','player_1','player_2','Rank_1','Rank_2','Pts_1','Pts_2','B365_1','B365_2','age_1','age_2']]

    import joblib

    loaded_rf = joblib.load("WTA/WTA.joblib")
    test20 = loaded_rf.predict_proba(odds_5)
    test20=pd.DataFrame(test20)
    test20.columns = ['prono_1', 'prono_2']

    odds_10 = odds_10.reset_index()
    result = pd.concat([odds_10, test20], axis=1,ignore_index=False)
    result=result.replace({"player_1": inv_map})
    result=result.replace({"player_2": inv_map})
    result = result.drop(['Surface','Court','Series','Pts_1','Pts_2'],axis=1)
    result.dropna(inplace = True)

    st.dataframe(result)



    ###############
    # VISUEL PLAYER
    ###############
    base = pd.read_csv("WTA/df_merged.csv")
    base['Date_x'] = pd.to_datetime(base['Date_x'])
    base = base.sort_values(by='Date_x') 

    d_exploration = pd.read_csv("WTA/df_v3.csv")
    d_exploration['Date_x'] = pd.to_datetime(d_exploration['Date_x'])
    d_exploration = d_exploration.sort_values(by='Date_x') 



    list_1 = list(result['player_1'].unique())
    list_2 = list(result['player_2'].unique())
    fruit_dictionary = dict(zip(list_1, list_2))
    final_list = list_1 + list_2
    final_list.sort()
    list_3 = list(base['Winner'].unique())
    list_4 = list(base['Loser'].unique())
    final_list2 = list_3 + list_4

    f_list = list(set(final_list).intersection(final_list2))

    length = len(f_list)
    for i in range(length):
        with st.expander(f_list[i]):
          query = f"player=='{f_list[i]}'"
          query2= f"Winner=='{f_list[i]}' | Loser=='{f_list[i]}'"
          df_filtered = d_exploration.query(query)
          df_filtered2 = base.query(query2)
          fig = px.line(df_filtered, x='Date_x', y='Rank',color='player',title = 'Player rank')
          fig.update_yaxes(autorange="reversed")
          st.plotly_chart(fig, use_container_width=True)


          st.markdown('Ten last results')
          df_filtered2 = df_filtered2.tail(10)
          #df_filtered2 = df_filtered2.drop(['ATP', 'Location','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','Comment','AvgW','AvgL','Court','Tournament','PSW','PSL','MaxL','MaxW','WPts','LPts','Location'],axis=1)
          df_filtered2 = df_filtered2.iloc[: , 1:]
          st.dataframe(df_filtered2)


          pro = df_filtered.groupby(['Tier', 'Surface','group']).size()
          pro = pro.reset_index()
          pro.columns = ['Tier', 'Surface', 'group', 'value']
          pro2 = df_filtered.groupby(['Tier', 'Surface','group']).sum()
          pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
          pro2 = pro2.reset_index()
          st.dataframe(pro2)
          pro['percent'] = pro2['WTA']
          fig = px.pie(pro, values='value', names='group',color='group',facet_col='Tier',facet_row='Surface',title='% victoire par surface et tournoi',color_discrete_map={'Winner': 'green','Loser': 'red'})
          fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
          st.plotly_chart(fig, use_container_width=True)


          df_filtered2 = base.query(query2)
          #df_filtered2 = df_filtered2.drop(['Location','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','Comment','AvgW','AvgL','Court','Tournament','PSW','PSL','MaxL','MaxW','WPts','LPts','Location'],axis=1)
          df_filtered2 = df_filtered2.iloc[: , 1:]
          df_filtered2['prono'] = np.where(df_filtered2['B365W'] <= df_filtered2['B365L'], 'b_prono', 'm_prono')
          pro = df_filtered2.groupby(['Tier','prono','Surface']).size()
          pro = pro.reset_index()
          pro.columns = ['Tier','prono','Surface', 'value']
          pro2 = df_filtered2.groupby(['Tier','Surface','prono']).sum()
          pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
          pro2 = pro2.reset_index()
          pro['percent'] = pro2['WTA']
          fig = px.pie(pro, values='value', names='prono',color='prono',facet_col='Tier',facet_row='Surface',title='% pronostic reussi par bookmaker',color_discrete_map={'b_prono': 'green','m_prono': 'red'})
          fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
          st.plotly_chart(fig, use_container_width=True)
