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
from tensorflow.keras.models import  load_model
import tensorflow as tf
import plotly.express as px


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# PLAYER CHOICE")
    concatenated = pd.read_csv("concatenated.csv")
    df_v3 = pd.read_csv("df_v3.csv")
    df_v3['Date'] = pd.to_datetime(df_v3['Date'])
    df_v3 = df_v3.sort_values(by='Date') 

    P1_list = list(df_v3['player'].unique())
    with st.sidebar:
        P1 = st.selectbox(label = "Player 1", options = P1_list)
        P2 = st.selectbox(label = "Player 2", options = P1_list)
    st.markdown('## RANK')


    query = f"player=='{P1}' | player=='{P2}'"

    df_filtered = df_v3.query(query)

    fig = px.line(df_filtered, x='Date', y='Rank',color='player',title = 'Player rank')
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    base = pd.read_csv("base.csv")
    query = f"Winner=='{P1}' & Loser=='{P2}' | Loser=='{P1}' & Winner=='{P2}'"
    base.drop(['AvgW','AvgL','Comment','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','AvgW','AvgL'], axis=1,inplace=True)

    df_filtered = base.query(query)

    df_filtered['BW_L']= df_filtered['B365W']-df_filtered['B365L']
    df_filtered['B3652'] = np.where(df_filtered['B365W'] <= df_filtered['B365L'], 'good', 'ngood')

    df_v2 = df_filtered.drop(['Winner', 'WRank','WPts','Wsets','B365W','PSW','MaxW'],axis=1)
    df_v2['group'] = 'Loser'

    df_v1 = df_filtered.drop(['Loser', 'LRank','LPts','Lsets','B365L','PSL','MaxL'],axis=1)
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
    if df_v3.empty:
        st.markdown('## Pas de confrontations entre les joueurs')
    else:
        st.markdown('## % VICTORY')

        st.markdown('### % VICTORY par tournoi')
        pro = df_v3.groupby(['player', 'Series','group']).size()
        pro = pro.reset_index()
        pro.columns = ['player', 'Series', 'group', 'value']
        pro2 = df_v3.groupby(['player', 'Series','group']).sum()

        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        pro['percent'] = pro2['ATP']

        fig = px.bar(pro, x="player", y="value", color="group",title="Nombre Victoires par Series", text_auto=True, facet_col="Series")


        fig.update_layout(barmode='relative')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown('### % VICTORY par surface')

        pro = df_v3.groupby(['player', 'Surface','group']).size()
        pro = pro.reset_index()
        pro.columns = ['player', 'Surface', 'group', 'value']
        pro2 = df_v3.groupby(['player', 'Surface','group']).sum()

        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        pro['percent'] = pro2['ATP']

        fig = px.bar(pro, x="player", y="value", color="group",title="Nombre Victoires par surface", text_auto=True, facet_col="Surface")
        fig.update_layout(barmode='relative')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('### % VICTORY par round')

        pro = df_v3.groupby(['player', 'Round','group']).size()
        pro = pro.reset_index()
        pro.columns = ['player', 'Round', 'group', 'value']
        pro2 = df_v3.groupby(['player', 'Round','group']).sum()

        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        pro['percent'] = pro2['ATP']

        fig = px.bar(pro, x="player", y="value", color="group",title="Nombre Victoires par round", text_auto=True, facet_col="Round")

        fig.update_layout(barmode='relative')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown('### % de réussite')
        pro = df_v3.groupby(['player','Surface','Series', 'B3652','group']).size()
        pro = pro.reset_index()
        pro.columns = ['player','Surface','Series', 'B3652', 'group', 'value']
        pro2 = df_v3.groupby(['player','Surface','Series', 'B3652','group']).sum()

        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        pro['percent'] = pro2['ATP']
    
        col = pro.columns
        pro3 = pro.head(len(col))
        fig = px.pie(pro3, values='value', names='B3652',color=None,facet_col='Series',facet_row='Surface')

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('### Five last results')
        df_filtered = base.query(query)
        df_filtered = df_filtered.tail(5)
        df_filtered = df_filtered.drop(['ATP', 'Court','Tournament','PSW','PSL','MaxL','MaxW','WPts','LPts','Location'],axis=1)
        df_filtered = df_filtered.iloc[: , 1:]
        st.dataframe(df_filtered)

    

