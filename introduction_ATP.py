
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
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# TENNIS STATISTICS AND PREDICTION")


    st.markdown("## Application pour le pronostic de matchs de tennis (ATP Tennis)")

    
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

    concatenated = pd.concat([a_2010,a_2011,a_2012,a_2013,a_2014,a_2015,a_2016,a_2017,a_2018,a_2019,a_2020,a_2021, a_2022])
    concatenated.to_csv('base.csv')

    concatenated.drop(['AvgW','AvgL','Comment'], axis=1,inplace=True)

    concatenated.rename(columns = {
                              'B365W':'WB365',
                              'B365L':'LB365',
                              'PSW':'WPS',
                              'PSL':'LPS',
                              'MaxW':'WMax',
                              'MaxL':'LMax'}, inplace = True)




    WL_extensions = ['Rank', 'Pts', '1', '2', '3', '4', '5', 'sets','B365','PS','Max']
    def obscure_features(DF):
        '''
        We replace 'winner' and 'loser' with 'player_1' and 'player_2' (not necessarily in that order)
        'player_1' replaces the name of the player that comes first alphabetically
        The purpose of this is to predict the winner of a match without the data being tied to
        the known winner or loser.
        '''
        DF['player_1'] = pd.concat([DF['Winner'], DF['Loser']], axis = 1).min(axis = 1)
        DF['player_2'] = pd.concat([DF['Winner'], DF['Loser']], axis = 1).max(axis = 1)
    
        for ext in WL_extensions:
            p1_feature = np.where(DF['player_1'] == DF['Winner'],
                     DF['W' + ext],
                     DF['L' + ext])
    
            p2_feature = np.where(DF['player_2'] == DF['Winner'],
                     DF['W' + ext],
                     DF['L' + ext])
    
            DF['player_1_' + ext] = p1_feature
            DF['player_2_' + ext] = p2_feature
        
        winner_cols = list(filter(lambda x: x.startswith('W'), DF.columns))
        loser_cols = list(filter(lambda x: x.startswith('L'), DF.columns))
        cols_to_drop = winner_cols + loser_cols
    
        target = DF['Winner']
    
        DF.drop(cols_to_drop, axis = 1, inplace = True)
    
        DF['target'] = target
    
        return DF

    obscure_features(concatenated)
    concatenated.drop(['player_2_2','player_1_3','player_2_3','player_1_4','player_2_4','player_1_5','player_1_2','player_1_1','player_2_1','player_2_5'], axis=1,inplace=True)
    concatenated['target'] = np.where(concatenated['player_1_sets'] >= concatenated['player_2_sets'], 1, 2)
    
    visua = pd.concat([a_2010,a_2011,a_2012,a_2013,a_2014,a_2015,a_2016,a_2017,a_2018,a_2019,a_2020,a_2021, a_2022])
    visua.drop(['AvgW','AvgL','Comment','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','AvgW','AvgL'], axis=1,inplace=True)
    
    df_v2 = visua.drop(['Winner', 'WRank','WPts','Wsets','B365W','PSW','MaxW'],axis=1)
    df_v2['group'] = 'Loser'

    df_v1 = visua.drop(['Loser', 'LRank','LPts','Lsets','B365L','PSL','MaxL'],axis=1)
    df_v1['group'] = 'Winner'

    df_v1.rename(columns = {'Winner':'player',
                        'WRank':'Rank',
                        'WPts':'Pts',
                        'Wsets':'sets',
                        'B365W':'B365',
                        'PSW':'PS',
                        'MaxW':'Max'}, inplace = True)

    df_v2.rename(columns = {'Loser':'player',
                        'LRank':'Rank',
                        'LPts':'Pts',
                        'Lsets':'sets',
                        'B365L':'B365',
                        'PSL':'PS',
                        'MaxL':'Max'}, inplace = True)


    df_v3 = pd.concat([df_v1,df_v2])
    df_v3['groupv2'] = np.where(df_v3['group'] == 'Winner', 1, 0)
    
    df_v3.to_csv('df_v3.csv')
    concatenated.to_csv('concatenated.csv')
 
          

    

