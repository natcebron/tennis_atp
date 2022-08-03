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
	  "X-RapidAPI-Key": "72fe0001damsh869f6f26b29a2c4p1f37d8jsnfb56f90133f4",
	  "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    df=pd.DataFrame(response.json())
    odds_1 = df['events'].apply(pd.Series)
    odds_1 = odds_1.drop(['sport_id','event_id','last','is_have_odds','parent_id','event_type'],axis=1)
    odds_1 = odds_1.drop(odds_1[(odds_1['resulting_unit'] == 'Sets')].index)
    odds_1 = odds_1.drop(odds_1[(odds_1['resulting_unit'] == 'Games')].index)

    odds_2 = odds_1['periods'].apply(pd.Series)
    odds_3 = odds_2['num_1'].apply(pd.Series)

    odds_4 = odds_2['num_0'].apply(pd.Series)
    odds_4 = odds_4.drop(['cutoff','line_id','number','period_status','spreads','team_total','totals','meta'],axis=1)
    odds_4 = odds_4['money_line'].apply(pd.Series)

    odds_5 = pd.concat([odds_1,odds_4],axis=1)
    #st.dataframe(odds_5)
    odds_5 = odds_5.drop(['resulting_unit','periods','draw','period_results'],axis=1)
    odds_5.columns = ['league_id', 'league_name', 'starts', 'home_player','away_player','home_odds','away_odds']
    #odds_5.columns = ['league_id', 'league_name', 'starts', 'home_player','away_player','t','away_odds','home_odds']
    #odds_5 = odds_5[['league_id', 'league_name', 'starts', 'home_player','away_player','t','home_odds','away_odds']]

    #odds_5 = odds_5.drop(['t'],axis=1)
    odds_5['last_name_home'] = odds_5['home_player'].str.split(' ').str[1]
    odds_5['last_name_away'] = odds_5['away_player'].str.split(' ').str[1]

    odds_5['first_name_home'] = odds_5['home_player'].str.split(' ').str[0]
    odds_5['first_name_away'] = odds_5['away_player'].str.split(' ').str[0]
    odds_5

    odds_5['first_name_home'] = odds_5['first_name_home'].str.split('').str[1]
    odds_5["home_player"] = odds_5['last_name_home'] +" "+ odds_5["first_name_home"]+"."
    odds_5['first_name_away'] = odds_5['first_name_away'].str.split('').str[1]
    odds_5["away_player"] = odds_5['last_name_away'] +" "+ odds_5["first_name_away"]+"."
    df_merged = pd.read_csv("C:/Users/ncebron/tennis_ATP/ATP/df_versus.csv")

    odds_5 = odds_5.drop(['last_name_home','last_name_away','first_name_home','first_name_away'],axis=1)
    odds_5["category"] = odds_5['league_name'].str.split(' ').str[0]
    odds_5['starts'] = pd.to_datetime(odds_5['starts'])
    list_of_names = df_merged['tourney_name'].to_list()
    list_of_names2 = df_merged['Surface'].to_list()
    d_surface = dict(zip(list_of_names, list_of_names2))

    list_of_names = df_merged['tourney_name'].to_list()
    list_of_names2 = df_merged['Court'].to_list()
    d_court = dict(zip(list_of_names, list_of_names2))

    list_of_names = df_merged['tourney_name'].to_list()
    list_of_names2 = df_merged['Series'].to_list()
    d_series = dict(zip(list_of_names, list_of_names2))


    a_2022 = pd.read_table('C:/Users/ncebron/tennis_ATP/test6.csv',sep=',')
    a_2021 = pd.read_table('C:/Users/ncebron/tennis_ATP/test5.csv',sep=',')

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
    odds_5 = odds_5[odds_5['category']=='ATP' ]
    df_v4 = pd.read_table('C:/Users/ncebron/tennis_ATP/ATP/df_v4.csv',sep=',')
    df_v4 = df_v4.groupby("player").last()
    df_v4 = df_v4.reset_index()
    series_1 = df_v4.set_index('player').to_dict()['series']
    forme = df_v4.set_index('player').to_dict()['forme']
    series_surf = df_v4.set_index('player').to_dict()['series_surf']
    forme_surf = df_v4.set_index('player').to_dict()['forme_surf']
    
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

    odds_5['home_player']= odds_5['home_player'].str.replace('Davidovich','Davidovich Fokina')
    odds_5['away_player']= odds_5['away_player'].str.replace('Davidovich','Davidovich Fokina')
    odds_5['home_player']= odds_5['home_player'].str.replace('Auger','Auger Aliassime')
    odds_5['away_player']= odds_5['away_player'].str.replace('Auger','Auger Aliassime')
    odds_5['home_player']= odds_5['home_player'].str.replace('Ramos Vinolas','Ramos')
    odds_5['away_player']= odds_5['away_player'].str.replace('Ramos Vinolas','Ramos')
    odds_5['home_player']= odds_5['home_player'].str.replace('Bautista','Bautista Agut')
    odds_5['away_player']= odds_5['away_player'].str.replace('Bautista','Bautista Agut')
    odds_5['first_rank'] = odds_5['home_player']
    odds_5['first_Pts'] = odds_5['home_player']
    odds_5['first_hand'] = odds_5['home_player']
    odds_5['first_age'] = odds_5['home_player']

    odds_5['second_rank'] = odds_5['away_player']
    odds_5['second_Pts'] = odds_5['away_player']
    odds_5['second_hand'] = odds_5['away_player']
    odds_5['second_age'] = odds_5['away_player']
    
    odds_5['first_series'] = odds_5['home_player']
    odds_5['first_forme'] = odds_5['home_player']
    odds_5['first_series_surf'] = odds_5['home_player']
    odds_5['first_forme_surf'] = odds_5['home_player']
    
    odds_5['second_series'] = odds_5['away_player']
    odds_5['second_forme'] = odds_5['away_player']
    odds_5['second_series_surf'] = odds_5['away_player']
    odds_5['second_forme_surf'] = odds_5['away_player']
    
    discard = ["Doubles"]

    odds_5 = odds_5[~odds_5.league_name.str.contains('|'.join(discard))]


    odds_5=odds_5.replace({"first_rank": d_rank})
    odds_5=odds_5.replace({"first_Pts": d_points})
    odds_5=odds_5.replace({"first_hand": d_hand})
    odds_5=odds_5.replace({"first_age": d_age})

    odds_5=odds_5.replace({"second_rank": d_rank})
    odds_5=odds_5.replace({"second_Pts": d_points})
    odds_5=odds_5.replace({"second_hand": d_hand})
    odds_5=odds_5.replace({"second_age": d_age})
    
    odds_5=odds_5.replace({"first_series": series_1})
    odds_5=odds_5.replace({"first_forme": forme})
    odds_5=odds_5.replace({"first_series_surf": series_surf})
    odds_5=odds_5.replace({"first_forme_surf": forme_surf})

    odds_5=odds_5.replace({"second_series": series_1})
    odds_5=odds_5.replace({"second_forme": forme})
    odds_5=odds_5.replace({"second_series_surf": series_surf})
    odds_5=odds_5.replace({"second_forme_surf": forme_surf})

    odds_5["surface"].replace({"Clay": "1",
                         "Grass": "2",
                         "Hard": "3"}, inplace=True)

    odds_5["Court"].replace({"Outdoor": "1",
                         "Indoor": "2"}, inplace=True)

    odds_5["series"].replace({"ATP250": "1",
                         "ATP500": "2",
                         "Masters 1000": "3",
                         "Masters Cup": "4",
                         "Grand Slam": "5"}, inplace=True)




    odds_3 = pd.read_table('C:/Users/ncebron/tennis_ATP/ATP/test5.csv',sep=',')
    test6 = pd.read_table('C:/Users/ncebron/tennis_ATP/ATP/test6.csv',sep=',')

    list_of_names = odds_3['player'].to_list()
    list_of_names2 = odds_3['rank_2022'].to_list()
    final_list = list(dict.fromkeys(list_of_names))
    lst = list(range(0,len(final_list)))

    fruit_dictionary = dict(zip(list_of_names, lst))
    fruit_dictionary10 = dict(zip(lst, final_list))
    inv_map = {v: k for k, v in fruit_dictionary.items()}



    odds_5["first_hand"].replace({"R": "1",
                         "L": "2",'U':'3'}, inplace=True)
    odds_5["second_hand"].replace({"R": "1",
                         "L": "2",'U':'3'}, inplace=True)

    odds30 = odds_5
    odds_5=odds_5.replace({"home_player": fruit_dictionary})
    odds_5=odds_5.replace({"away_player": fruit_dictionary})
    odds_5['second_rank'] = pd.to_numeric(odds_5['second_rank'], errors = 'coerce')
    odds_5['first_rank'] = pd.to_numeric(odds_5['first_rank'], errors = 'coerce')
    odds_5['away_player'] = pd.to_numeric(odds_5['away_player'], errors = 'coerce')
    odds_5['home_player'] = pd.to_numeric(odds_5['home_player'], errors = 'coerce')

    odds_5['first_series'] = pd.to_numeric(odds_5['first_series'], errors = 'coerce')
    odds_5['first_forme'] = pd.to_numeric(odds_5['first_forme'], errors = 'coerce')
    odds_5['first_series_surf'] = pd.to_numeric(odds_5['first_series_surf'], errors = 'coerce')
    odds_5['first_forme_surf'] = pd.to_numeric(odds_5['first_forme_surf'], errors = 'coerce')
    odds_5['second_series'] = pd.to_numeric(odds_5['second_series'], errors = 'coerce')
    odds_5['second_forme'] = pd.to_numeric(odds_5['second_forme'], errors = 'coerce')
    odds_5['second_series_surf'] = pd.to_numeric(odds_5['second_series_surf'], errors = 'coerce')
    odds_5['second_forme_surf'] = pd.to_numeric(odds_5['second_forme_surf'], errors = 'coerce')
    odds_5.dropna(inplace = True)
    odds_5 = odds_5.drop_duplicates(['home_player'])    

    odds_5 = odds_5.drop(['userName','category','starts','league_name','league_id'],axis=1)


    odds_5.columns = ['player_1', 'player_2',  'B365_1','B365_2','Surface','Court','Series','Best of','Rank_1',
                      'Pts_1','hand_1','age_1','Rank_2','Pts_2','hand_2','age_2','first_series','first_forme','first_series_surf',
                      'first_forme_surf','second_series','second_forme','second_series_surf','second_forme_surf']
    #odds_5 = odds_5.drop(['hand_1','hand_2'],axis=1)
    odds_5 = odds_5[['Series', 'Court', 'Surface', 'Rank_2','Pts_2','B365_2','hand_2','age_2','second_series',
                     'second_forme','second_series_surf','second_forme_surf',
                    'Rank_1','Pts_1','B365_1','hand_1','age_1','first_series','first_forme','first_series_surf',
                      'first_forme_surf','player_1','player_2']]

    
    odds10 = odds_5




    ############
    # PREDICTION
    ############
    import joblib
    test = pd.read_table('df_model.csv',sep=',')

    loaded_rf = joblib.load("ATP/ATP.joblib")
    test20 = loaded_rf.predict_proba(odds_5)
    test20=pd.DataFrame(test20)
    test20.columns = ['prono_1', 'prono_2']


    odds10 = odds10.reset_index()
    result = pd.concat([odds10, test20], axis=1,ignore_index=False)
    result=result.replace({"player_1": inv_map})
    result=result.replace({"player_2": inv_map})
    df_v10 = pd.read_csv("C:/Users/ncebron/tennis_ATP/ATP/df_v10.csv")

    df_v10["ROI"]=df_v10["ROI"].apply(int)
    test = df_v10.groupby(['player','points']).sum()
    pro2 = df_v10.groupby(['player','points']).sum()
    pro2 = pro2.groupby(level=[0]).apply(lambda g: g / g.sum())

    pro2 = pro2.reset_index()
    test = test.reset_index()

    test['percent'] = pro2['index']
    test2 = test.loc[test['points'] == 'bp']
    list_1 = list(test2['player'])
    list_2 = list(test2['percent'])
    fruit_dictionary = dict(zip(list_1, list_2))

    result["%_bp_player_1"] = result['player_1']
    result["%_bp_player_2"] = result['player_2']
    result=result.replace({"%_bp_player_1": fruit_dictionary})
    result=result.replace({"%_bp_player_2": fruit_dictionary})


    result['%_bp_player_1'] = pd.to_numeric(result['%_bp_player_1'], errors = 'coerce')
    result['%_bp_player_2'] = pd.to_numeric(result['%_bp_player_2'], errors = 'coerce')
    
    
    df2 = pd.read_csv('C:/Users/ncebron/tennis_ATP/ATP/df2.csv')

    df2["group"] = df2["first_player_id"] +' '+ df2["second_player_id"]
    df2['value'] = 1
    test = df2.groupby(['group']).sum()
    test
    test = test.reset_index()
    test2 = test[['group','value','label']]
    test2['vict_2'] = test2['value'] - test2['label']



    list_of_names = test2['group'].to_list()
    list_of_names2 = test2['value'].to_list()
    list_of_names3 = test2['label'].to_list()
    list_of_names4 = test2['vict_2'].to_list()

    fruit_dictionary = dict(zip(list_of_names, list_of_names2))
    fruit_dictionary2 = dict(zip(list_of_names, list_of_names3))
    fruit_dictionary3 = dict(zip(list_of_names, list_of_names4))

    fruit_dictionary2

    result["nb_match_1"] = result["player_1"] +' '+ result["player_2"]
    result["nb_match_2"] = result["player_2"] +' '+ result["player_1"]

    result["victory_1_1"] = result["player_1"] +' '+ result["player_2"]
    result["victory_1_2"] = result["player_2"] +' '+ result["player_1"]




    result=result.replace({"nb_match_1": fruit_dictionary})
    result=result.replace({"nb_match_2": fruit_dictionary})
    result=result.replace({"victory_1_1": fruit_dictionary2})
    result=result.replace({"victory_1_2": fruit_dictionary3})




    result['nb_match_1'] = pd.to_numeric(result['nb_match_1'], errors = 'coerce')
    result['nb_match_2'] = pd.to_numeric(result['nb_match_2'], errors = 'coerce')
    result['victory_1_1'] = pd.to_numeric(result['victory_1_1'], errors = 'coerce')
    result['victory_1_2'] = pd.to_numeric(result['victory_1_2'], errors = 'coerce')
    result = result.fillna(0)

    result["victory_1"] = result["victory_1_1"].astype(int) + result["victory_1_2"].astype(int)
    result = result.drop(['victory_1_2','victory_1_1'],axis=1)

    result['victory_1'] = pd.to_numeric(result['victory_1'], errors = 'coerce')



    result['Total'] = result['nb_match_1'] + result['nb_match_2']
    result["victory_2"] = result["Total"] - result["victory_1"]

    result["versus"] = result["Total"].astype(str) +" (" + result["victory_1"].astype(str) + " / " + result["victory_2"].astype(str) + ")"
    result = result.drop(['nb_match_1','nb_match_2','victory_1','Total','victory_2'],axis=1)

    result = result[['player_1', 'first_forme', 'Rank_1', 'B365_1','player_2','second_forme','Rank_2',
                        'B365_2','prono_1','prono_2','versus','%_bp_player_1','%_bp_player_2']]
    
    st.dataframe(result)




    '''
    ###############
    # VISUEL PLAYER
    ###############
    base = pd.read_csv("ATP/df_merged.csv")
    base['Date_x_x'] = pd.to_datetime(base['Date_x_x'])
    base = base.sort_values(by='Date_x_x') 

    d_exploration = pd.read_csv("ATP/df_v3.csv")
    d_exploration['Date_x'] = pd.to_datetime(d_exploration['Date_x'])
    d_exploration = d_exploration.sort_values(by='Date_x') 



    list_1 = list(result['player_1'].unique())
    list_2 = list(result['player_2'].unique())
    fruit_dictionary = dict(zip(list_1, list_2))
    final_list = list_1 + list_2
    final_list.sort()
    list_3 = list(base['winner_player'].unique())
    list_4 = list(base['loser_player'].unique())
    final_list2 = list_3 + list_4

    f_list = list(set(final_list).intersection(final_list2))

    length = len(f_list)
    for i in range(length):
        with st.expander(f_list[i]):
          query = f"player=='{f_list[i]}'"
          query2= f"winner_player=='{f_list[i]}' | loser_player=='{f_list[i]}'"
          df_filtered = d_exploration.query(query)
          df_filtered2 = base.query(query2)
          fig = px.line(df_filtered, x='Date_x', y='Rank',color='player',title = 'Player rank')
          fig.update_yaxes(autorange="reversed")
          st.plotly_chart(fig, use_container_width=True)



          st.markdown('Ten last results')
          df_filtered2 = df_filtered2.tail(10)
          #df_filtered2 = df_filtered2.drop(['ATP', 'Location','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','Comment','AvgW','AvgL','Court','Tournament','PSW','PSL','MaxL','MaxW','WPts','LPts','Location'],axis=1)
          df_filtered2 = df_filtered2.iloc[: , 1:]



          pro = df_filtered.groupby(['Series', 'Surface','group']).size()
          pro = pro.reset_index()
          pro.columns = ['Series', 'Surface', 'group', 'value']
          pro2 = df_filtered.groupby(['Series', 'Surface','group']).sum()
          pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
          pro2 = pro2.reset_index()
          pro['percent'] = pro2['ATP']
          fig = px.pie(pro, values='value', names='group',color='group',facet_col='Series',facet_row='Surface',title='% victoire par surface et tournoi',color_discrete_map={'Winner': 'green','Loser': 'red'})
          fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
          st.plotly_chart(fig, use_container_width=True)


          df_filtered2 = base.query(query2)
          #df_filtered2 = df_filtered2.drop(['Location','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','Comment','AvgW','AvgL','Court','Tournament','PSW','PSL','MaxL','MaxW','WPts','LPts','Location'],axis=1)
          df_filtered2 = df_filtered2.iloc[: , 1:]
          df_filtered2['prono'] = np.where(df_filtered2['B365W'] <= df_filtered2['B365L'], 'b_prono', 'm_prono')
          pro = df_filtered2.groupby(['Series','prono','Surface']).size()
          pro = pro.reset_index()
          pro.columns = ['Series','prono','Surface', 'value']
          pro2 = df_filtered2.groupby(['Series','Surface','prono']).sum()
          pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
          pro2 = pro2.reset_index()
          pro['percent'] = pro2['ATP']
          fig = px.pie(pro, values='value', names='prono',color='prono',facet_col='Series',facet_row='Surface',title='% pronostic reussi par bookmaker',color_discrete_map={'b_prono': 'green','m_prono': 'red'})
          fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
          st.plotly_chart(fig, use_container_width=True)
      '''

