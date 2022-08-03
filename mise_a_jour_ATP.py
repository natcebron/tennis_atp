import datetime
import streamlit as st
import streamlit.components.v1 as components
import os                      #+Deployment
import inspect                 #+Deployment
#importing all the necessary libraries
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt
import os
import random               #+Deployment
from PIL import Image, ImageStat

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))


    ##############
    # DATA PLAYERS
    ##############
    atp_p = pd.read_table('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv',sep=',')
    atp_p.rename(columns = {'player_id':'player'}, inplace = True)

    atp_r = pd.read_table('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv',sep=',')
    atp_r.rename(columns = {'player_id':'player'}, inplace = True)

    atp_r = atp_r[atp_r['ranking_date'] == 20220627]
    atp_r = atp_r.sort_values(by='rank') 

    data_p = pd.merge(atp_r,atp_p,on='player')
    data_p['First'] = data_p['name_first'].str.split('').str[1]
    data_p["Period"] = data_p['name_last'] +" "+ data_p["First"]+"."
    data_p.drop(['player','name_first','name_last','ioc','height','wikidata_id','First','ranking_date'], axis=1,inplace=True)
    data_p.rename(columns = {'Period':'player','rank':'rank_2022'}, inplace = True)
    from datetime import datetime, date
    data_p_f = data_p[['dob','player']]
    data_p_f=data_p_f.dropna(axis=0)

    def age(born):
            born = datetime.strptime(born, "%Y%m%d").date()
            today = date.today()
            return today.year - born.year - ((today.month, 
                                      today.day) < (born.month, 
                                                    born.day))
    data_p = data_p.astype({"dob": str})

    data_p_f['Age'] = data_p_f['dob'].apply(age)
    data_p.to_csv('ATP/test5.csv')    
    data_p_f.to_csv('ATP/test6.csv')    

    #############################
    # PREPARATION FICHIERS MODELE
    #############################

    a_2010 = pd.read_excel('http://www.tennis-data.co.uk/2010/2010.xlsx')
    a_2011 = pd.read_excel('http://www.tennis-data.co.uk/2011/2011.xlsx')
    a_2012 = pd.read_excel('http://www.tennis-data.co.uk/2012/2012.xlsx')
    a_2013 = pd.read_excel('http://www.tennis-data.co.uk/2013/2013.xlsx')
    a_2014 = pd.read_excel('http://www.tennis-data.co.uk/2014/2014.xlsx')

    a_2015 = pd.read_excel('http://www.tennis-data.co.uk/2015/2015.xlsx')
    a_2016 = pd.read_excel('http://www.tennis-data.co.uk/2016/2016.xlsx')
    a_2017 = pd.read_excel('http://www.tennis-data.co.uk/2017/2017.xlsx')
    a_2018 = pd.read_excel('http://www.tennis-data.co.uk/2018/2018.xlsx')

    a_2019 = pd.read_excel('http://www.tennis-data.co.uk/2019/2019.xlsx')
    a_2020 = pd.read_excel('http://www.tennis-data.co.uk/2020/2020.xlsx')
    a_2021 = pd.read_excel('http://www.tennis-data.co.uk/2021/2021.xlsx')
    a_2022 = pd.read_excel('http://www.tennis-data.co.uk/2022/2022.xlsx')

    a_2015.drop(['EXW','EXL','LBW','LBL'], axis=1,inplace=True)
    a_2016.drop(['EXW','EXL','LBW','LBL'], axis=1,inplace=True)
    a_2017.drop(['EXW','EXL','LBW','LBL'], axis=1,inplace=True)
    a_2018.drop(['EXW','EXL','LBW','LBL'], axis=1,inplace=True)

    a_2010.drop(['EXW','EXL','LBW','LBL','SJW','SJL'], axis=1,inplace=True)
    a_2011.drop(['EXW','EXL','LBW','LBL','SJW','SJL'], axis=1,inplace=True)
    a_2012.drop(['EXW','EXL','LBW','LBL','SJW','SJL'], axis=1,inplace=True)
    a_2013.drop(['EXW','EXL','LBW','LBL','SJW','SJL'], axis=1,inplace=True)
    a_2014.drop(['EXW','EXL','LBW','LBL','SJW','SJL'], axis=1,inplace=True)

    dataset1 = pd.concat([a_2010,a_2011,a_2012,a_2013,a_2014,a_2015,a_2016,a_2017,a_2018,a_2019,a_2020,a_2021, a_2022])

    a_2010 = pd.read_table('ATP/data/atp_matches_2010.csv',sep=',')
    a_2011 = pd.read_table('ATP/data/atp_matches_2011.csv',sep=',')
    a_2012 = pd.read_table('ATP/data/atp_matches_2012.csv',sep=',')
    a_2013 = pd.read_table('ATP/data/atp_matches_2013.csv',sep=',')
    a_2014 = pd.read_table('ATP/data/atp_matches_2014.csv',sep=',')

    a_2015 = pd.read_table('ATP/data/atp_matches_2015.csv',sep=',')
    a_2016 = pd.read_table('ATP/data/atp_matches_2016.csv',sep=',')
    a_2017 = pd.read_table('ATP/data/atp_matches_2017.csv',sep=',')
    a_2018 = pd.read_table('ATP/data/atp_matches_2018.csv',sep=',')

    a_2019 = pd.read_table('ATP/data/atp_matches_2019.csv',sep=',')
    a_2020 = pd.read_table('ATP/data/atp_matches_2020.csv',sep=',')
    a_2021 = pd.read_table('ATP/data/atp_matches_2021.csv',sep=',')
    a_2022 = pd.read_table('ATP/data/atp_matches_2022.csv',sep=',')
    dataset2 = pd.concat([a_2010,a_2011,a_2012,a_2013,a_2014,a_2015,a_2016,a_2017,a_2018,a_2019,a_2020,a_2021, a_2022])

    #dataset2 = dataset2.drop(['tourney_id','match_num','winner_rank_points','loser_rank_points','draw_size'],axis=1)
    dataset2 = dataset2.drop(['winner_ioc','loser_ioc','loser_seed','winner_seed','tourney_level','winner_entry','loser_entry','loser_ht','winner_ht',],axis=1)
    dataset2.head(5)

    import datetime
    dataset2['tourney_date'] = pd.to_datetime(dataset2['tourney_date'], format = '%Y%m%d')


    test5 = pd.read_table('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv',sep=',')
    test5['First'] = test5['name_first'].str.split('').str[1]
    test5["Period"] = test5['name_last'] +" "+ test5["First"]+"."

    test6 = test5[['Period','player_id']]
    dict5 = test6.set_index('player_id').to_dict()['Period']
    dict5

    dataset2=dataset2.replace({"winner_id": dict5})
    dataset2=dataset2.replace({"loser_id": dict5})

    dataset2 = dataset2.drop(['winner_name','loser_name'],axis=1)
    dataset2.rename(columns = {'tourney_date':'Date', 'winner_id':'Winner','loser_id':'Loser','winner_rank_points':'WPts','loser_rank_points':'LPts'}, inplace = True)
    pd.set_option('display.max_rows', 10)  # or 1000
    pd.set_option('display.max_columns', 10)  # or 1000
    dataset1['Loser'] = dataset1['Loser'].str.replace('-',' ')
    dataset1['Winner'] = dataset1['Winner'].str.replace('-',' ')
    dataset2['Loser'] = dataset2['Loser'].str.replace('-',' ')
    dataset2['Winner'] = dataset2['Winner'].str.replace('-',' ')

    df_cd1 = dataset1[dataset1['Winner'].isin(['Nadal R.']) ]
    list_of_names4 = df_cd1['Loser'].to_list()
    list_of_names = dataset2['Winner'].to_list()
    list_of_names2 = dataset2['Loser'].to_list()
    list_of_names3 = dataset1['Winner'].to_list()
    list_of_names4 = dataset1['Loser'].to_list()
    list_of_names = dataset1['Winner'].to_list()
    list_of_names2 = dataset1['Loser'].to_list()
    final_list = list_of_names + list_of_names2
    dict2 ={'w_ace':'w number of aces',
    'w_df':'w number of doubles faults',
    'w_svpt':'w number of serve points',
    'w_1stIn':'w number of first serves made',
    'w_1stWon':'w number of first-serve points won',
    'w_2ndWon':'w number of second-serve points won',
    'w_SvGms':'w number of serve games',
    'w_bpSaved':'w number of break points saved',
    'w_bpFaced':'w number of break points faced'}
    #pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', 10)  # or 1000

    df_merged = pd.merge(dataset1, dataset2, on=['Winner', 'Loser','WPts','LPts'], how='inner')
    df_merged=df_merged.drop(['W1','W2','W3','W4','W5',
                            'L1','L2','L3','L4','L5',
                            'Comment','draw_size','tourney_id','PSW','PSL','MaxW','MaxL','AvgW','AvgL'],axis=1)
    df_merged = df_merged.sort_values(['Date_x'],
                ascending = [True])
    df_merged=df_merged.drop(['Date_y','match_num','best_of','round','winner_rank','loser_rank'],axis=1)
    df_merged = df_merged.reset_index()
    df_merged.rename(columns = {'index':'match_id'}, inplace = True)
    pd.set_option('display.max_rows', 650)  # or 1000

    df_v2 = df_merged.drop(['Winner',"WRank", 
                            "WPts", 
                            "Wsets",
                            "B365W",
                            "winner_hand",
                            "winner_age",
                            "w_ace",
                            'w_df',
                            'w_svpt',
                            'w_1stIn',
                            'w_1stWon',
                            'w_2ndWon',
                            'w_SvGms',
                            'w_bpSaved',
                            'w_bpFaced'],axis=1)
    df_v2['group'] = 'Loser'

    df_v1 = df_merged.drop(['Loser',"LRank","LPts", 
                            "Lsets",
                            "B365L",
                            "loser_hand",
                            "loser_age",
                            "l_ace",
                            'l_df',
                            'l_svpt',
                            'l_1stIn',
                            'l_1stWon',
                            'l_2ndWon',
                            'l_SvGms',
                            'l_bpSaved',
                            'l_bpFaced'],axis=1)
    df_v1['group'] = 'Winner'

    df_v1.rename(columns = {'Winner':'player',
                            "WRank":'Rank',
                            "WPts":'Pts', 
                            "Wsets":'Sets',
                            "B365W":'Odds',
                            "winner_hand":'Hand',
                            "winner_age":'Age',
                            "w_ace":'Ace',
                            'w_df':'Df',
                            'w_svpt':'Svpt',
                            'w_1stIn':'1stIn',
                            'w_1stWon':'1stWon',
                            'w_2ndWon':'2ndWon',
                            'w_SvGms':'SvGms',
                            'w_bpSaved':'BpSaved',
                            'w_bpFaced':'BpFaced'}, inplace = True)

    df_v2.rename(columns = {'Loser':'player',
                            "LRank":'Rank',
                            "LPts":'Pts', 
                            "Lsets":'Sets',
                            "B365L":'Odds',
                            "loser_hand":'Hand',
                            "loser_age":'Age',
                            "l_ace":'Ace',
                            'l_df':'Df',
                            'l_svpt':'Svpt',
                            'l_1stIn':'1stIn',
                            'l_1stWon':'1stWon',
                            'l_2ndWon':'2ndWon',
                            'l_SvGms':'SvGms',
                            'l_bpSaved':'BpSaved',
                            'l_bpFaced':'BpFaced'}, inplace = True)
    df_v3 = pd.concat([df_v1,df_v2])
    df_v3.to_csv('ATP/df_v3.csv')

    #pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', 10)  # or 1000

    df_merged = pd.merge(dataset1, dataset2, on=['Winner', 'Loser','WPts','LPts'], how='inner')
    df_merged=df_merged.drop(['W1','W2','W3','W4','W5',
                            'L1','L2','L3','L4','L5',
                            'Comment','draw_size','tourney_id','PSW','PSL','MaxW','MaxL','AvgW','AvgL'],axis=1)
    df_merged = df_merged.sort_values(['Date_x'],
                ascending = [True])
    df_merged=df_merged.drop(['Date_y','match_num','best_of','round','winner_rank','loser_rank'],axis=1)
    df_merged = df_merged.reset_index()
    df_merged.rename(columns = {'index':'match_id'}, inplace = True)
    df_merged.to_csv('ATP/df_versus.csv')
    pd.set_option('display.max_rows', 650)  # or 1000

    df_v2 = df_merged.drop(['Winner',"WRank", 
                            "WPts", 
                            "Wsets",
                            "B365W",
                            "winner_hand",
                            "winner_age",
                            "w_ace",
                            'w_df',
                            'w_svpt',
                            'w_1stIn',
                            'w_1stWon',
                            'w_2ndWon',
                            'w_SvGms',
                            'w_bpSaved',
                            'w_bpFaced'],axis=1)
    df_v2['group'] = 'Loser'

    df_v1 = df_merged.drop(['Loser',"LRank","LPts", 
                            "Lsets",
                            "B365L",
                            "loser_hand",
                            "loser_age",
                            "l_ace",
                            'l_df',
                            'l_svpt',
                            'l_1stIn',
                            'l_1stWon',
                            'l_2ndWon',
                            'l_SvGms',
                            'l_bpSaved',
                            'l_bpFaced'],axis=1)
    df_v1['group'] = 'Winner'

    df_v1.rename(columns = {'Winner':'player',
                            "WRank":'Rank',
                            "WPts":'Pts', 
                            "Wsets":'Sets',
                            "B365W":'Odds',
                            "winner_hand":'Hand',
                            "winner_age":'Age',
                            "w_ace":'Ace',
                            'w_df':'Df',
                            'w_svpt':'Svpt',
                            'w_1stIn':'1stIn',
                            'w_1stWon':'1stWon',
                            'w_2ndWon':'2ndWon',
                            'w_SvGms':'SvGms',
                            'w_bpSaved':'BpSaved',
                            'w_bpFaced':'BpFaced'}, inplace = True)

    df_v2.rename(columns = {'Loser':'player',
                            "LRank":'Rank',
                            "LPts":'Pts', 
                            "Lsets":'Sets',
                            "B365L":'Odds',
                            "loser_hand":'Hand',
                            "loser_age":'Age',
                            "l_ace":'Ace',
                            'l_df':'Df',
                            'l_svpt':'Svpt',
                            'l_1stIn':'1stIn',
                            'l_1stWon':'1stWon',
                            'l_2ndWon':'2ndWon',
                            'l_SvGms':'SvGms',
                            'l_bpSaved':'BpSaved',
                            'l_bpFaced':'BpFaced'}, inplace = True)
    df_v3 = pd.concat([df_v1,df_v2])
    df_v3["group"].replace({"Winner": "1",
                            "Loser": "0"}, inplace=True)

    df_v3 = df_v3.sort_values(['player', 'Date_x'],
                ascending = [True, True])
    test3 = df_v3.groupby(['player'])['group'].rolling(5).sum()
    test3 = test3.reset_index()
    df_v3 = df_v3.reset_index()

    df_v3['series'] = test3['group']

    pd.set_option('display.max_rows', 10)  # or 1000
    df_v3['forme'] = df_v3['series'] / 5 * 100
    df_v3 = df_v3.sort_values(['player','surface', 'Date_x'],
                ascending = [True, True,True])
    test3 = df_v3.groupby(['player','surface'])['group'].rolling(5).sum()
    test3 = test3.reset_index()
    df_v3 = df_v3.reset_index()
    df_v3['series_surf'] = test3['group']
    df_v3['forme_surf'] = df_v3['series_surf'] / 5 * 100
    df_v3 = df_v3.drop(['index','level_0'],axis=1)
    df_v3 = df_v3.drop(['BpFaced','BpSaved','SvGms','2ndWon','1stWon','1stIn','Svpt','Df','Ace','minutes','score','surface'],axis=1)
    df_v3 = df_v3.sort_values(['Date_x'],
                ascending = [True])
    df_cd1 = df_v3[df_v3['player'].isin(['Nadal R.']) ]
    df_v3 = df_v3.sort_values(by=['match_id'])
    df_v3.to_csv('ATP/df_v4.csv')

    df_new_1 = df_v3[df_v3['group'] == '1']
    df_new_2 = df_v3[df_v3['group'] == '0']

    df_new_1.rename(columns = {'forme_surf':'winner_forme_surf',
                            'series_surf':'winner_series_surf',
                            'forme':'winner_forme',
                            'series':'winner_series',
                            'player':'winner_player',
                            'Rank':'winner_Rank',
                            'Pts':'winner_Pts',
                            'Sets':'winner_Sets',
                            'Odds':'winner_Odds',
                            'Hand':'winner_Hand',
                            'Age':'winner_Age'
                            }, inplace = True)



    df_new_2.rename(columns = {'forme_surf':'loser_forme_surf',
                            'series_surf':'loser_series_surf',
                            'forme':'loser_forme',
                            'series':'loser_series',
                            'player':'loser_player',
                            'Rank':'loser_Rank',
                            'Pts':'loser_Pts',
                            'Sets':'loser_Sets',
                            'Odds':'loser_Odds',
                            'Hand':'loser_Hand',
                            'Age':'loser_Age'
                            }, inplace = True)


    df_merged2 = pd.merge(df_new_1, df_new_2, on=['match_id'], how='inner')
    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', 50)  # or 1000
    df_merged2.to_csv('ATP/df.csv')
