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
import graphs_bokeh
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

    ##########################
    # PREPARATION DES FICHIERS
    ##########################
    concatenated = pd.read_csv("ATP/df_merged.csv")
    df_v3 = pd.read_csv("ATP/df_v3.csv")
    df_v3['Date_x'] = pd.to_datetime(df_v3['Date_x'])
    df_v3 = df_v3.sort_values(by='Date_x') 

    P1_list = list(df_v3['player'].unique())
    with st.sidebar:
        P1 = st.selectbox(label = "Player 1", options = P1_list)
        P2 = st.selectbox(label = "Player 2", options = P1_list)

    #############
    # RANK PLAYER
    #############
    st.markdown('## RANK')
    query = f"player=='{P1}' | player=='{P2}'"

    df_filtered = df_v3.query(query)

    fig = px.line(df_filtered, x='Date_x', y='Rank',color='player',title = 'Player rank')
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    base = pd.read_csv("ATP/df_versus.csv")
    query = f"Winner=='{P1}' & Loser=='{P2}' | Loser=='{P1}' & Winner=='{P2}'"

    df_filtered = base.query(query)

    df_filtered['BW_L']= df_filtered['B365W']-df_filtered['B365L']
    df_filtered['B3652'] = np.where(df_filtered['B365W'] <= df_filtered['B365L'], 'good', 'ngood')

    df_v2 = df_filtered.drop(['Winner',"WRank", 
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

    df_v1 = df_filtered.drop(['Loser',"LRank","LPts", 
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
    df_v4 = pd.concat([df_v1,df_v2])


    if df_v4.empty:
        st.markdown('## Pas de confrontations entre les joueurs')
    else:
        ###########
        # % VICTORY
        ###########
        st.markdown('## % VICTORY')
        st.markdown('### % VICTORY par tournoi')
        pro = df_v4.groupby(['player', 'Series','group']).size()
        pro = pro.reset_index()
        pro.columns = ['player', 'Series', 'group', 'value']
        pro2 = df_v4.groupby(['player', 'Series','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        pro['percent'] = pro2['ATP']
        fig = px.pie(pro, values='value', names='group',color='group',facet_col='Series',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.update_layout(barmode='relative')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown('### % VICTORY par surface')
        pro = df_v4.groupby(['player', 'Surface','group']).size()
        pro = pro.reset_index()
        pro.columns = ['player', 'Surface', 'group', 'value']
        pro2 = df_v4.groupby(['player', 'Surface','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        pro['percent'] = pro2['ATP']
        fig = px.bar(pro, x="player", y="value", color="group",title="Nombre Victoires par surface", text_auto=True, facet_col="Surface",color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.update_layout(barmode='relative')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('### % VICTORY par round')
        pro = df_v4.groupby(['player', 'Round','group']).size()
        pro = pro.reset_index()
        pro.columns = ['player', 'Round', 'group', 'value']
        pro2 = df_v4.groupby(['player', 'Round','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        pro['percent'] = pro2['ATP']
        fig = px.bar(pro, x="player", y="value", color="group",title="Nombre Victoires par round", text_auto=True, facet_col="Round",color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.update_layout(barmode='relative')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)
    
        #########################
        # % DE REUSSITE BOOKMAKER
        #########################
        st.markdown('## % de réussite bookmaker')
        pro = df_v4.groupby(['player','Surface','Series', 'B3652','group']).size()
        pro = pro.reset_index()
        pro.columns = ['player','Surface','Series', 'B3652', 'group', 'value']
        pro2 = df_v4.groupby(['player','Surface','Series', 'B3652','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        pro['percent'] = pro2['ATP']
        col = pro.columns
        pro3 = pro.head(len(col))
        fig = px.pie(pro3, values='value', names='B3652',color='B3652',facet_col='Series',facet_row='Surface',color_discrete_map={'good': 'green','ngood': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        ###################
        # FIVE LAST RESULTS
        ###################
        st.markdown('### Five last results')
        df_filtered = base.query(query)
        df_filtered = df_filtered.tail(5)
        df_filtered = df_filtered.drop(['ATP', 'Court','Tournament','WPts','LPts','Location'],axis=1)
        df_filtered = df_filtered.iloc[: , 1:]
        st.dataframe(df_filtered)

        #############
        # ODDS MOYENS
        #############
        st.markdown('### Odds moyens')
        test = df_v4.groupby(['player','Series', 'Surface']).mean()
        test = test.reset_index()
        fig = px.bar(test, x="player", y="Odds",title='Odds par series et surface',color='player',facet_col='Series',facet_row='Surface')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        ################################################
        # DUREE MOYENNE EN MINUTES PAS SURFACE ET SERIES
        ################################################
        st.markdown('### Moyenne durée en Minutes par surface et series')
        test = df_v4.groupby(['Series', 'Surface']).mean()
        test = test.reset_index()
        fig = px.bar(test, x="Surface", y="minutes",title='Durée match en minutes par series et surface',color='Surface',facet_col='Series')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        ########################
        # % VICTOIRE PAR SURFACE
        ########################
        st.markdown('### % de victoire par surface')
        test = df_v4.groupby(['player', 'Surface','group']).mean()
        test = test.reset_index()
        pro2 = df_v4.groupby(['player', 'Surface','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        test['percent'] = pro2['1stIn']
        fig = px.bar(test, x="player", y="percent",title="% de victoire par surface",color='group',facet_col='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        ################################
        # NOMBRE MOYEN D'ACE PAR SURFACE
        ################################
        st.markdown('### Moyenne ace par surface')
        test = df_v4.groupby(['player', 'Surface']).mean()
        test = test.reset_index()
        fig = px.bar(test, x="player", y="Ace",title="Nombre moyen d'ace par surface",color='player',facet_col='Surface')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)


        ############################################
        # NOMBRE MOYEN DE DOUBLES FAUTES PAR SURFACE
        ############################################
        st.markdown('### Nombre moyen doubles fautes par surface')
        test = df_v4.groupby(['player', 'Surface','group']).mean()
        test = test.reset_index()
        fig = px.bar(test, x="player", y="Df",title="Nombre moyen doubles fautes par surface",color='group',facet_col='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        #############################################
        # NOMBRE POINTS MOYENS AU SERVICE PAR SURFACE
        #############################################
        st.markdown('### Moyenne points au service par surface')
        test = df_v4.groupby(['player', 'Surface','group']).mean()
        test = test.reset_index()
        fig = px.bar(test, x="player", y="Svpt",title="Nombre moyen points sur service par series",color='group',facet_col='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        #####################################
        # % DE 1ER SERVICE REUSSI PAR SURFACE
        #####################################
        st.markdown('### % de 1er service réussi par surface')
        test = df_v4.groupby(['player', 'Surface','group']).mean()
        test = test.reset_index()
        pro2 = df_v4.groupby(['player', 'Surface','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        test['percent'] = pro2['1stIn']
        fig = px.bar(test, x="player", y="percent",title="% de 1er service réussi par surface",color='group',facet_col='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        ###############################################
        # % DE POINTS GAGNES AU 1ER SERVICE PAR SURFACE
        ###############################################
        st.markdown('### % de points gagnés au 1er service par surface')
        test = df_v4.groupby(['player', 'Surface','group']).mean()
        test = test.reset_index()
        pro2 = df_v4.groupby(['player', 'Surface','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        test['percent'] = pro2['1stWon']
        fig = px.bar(test, x="player", y="percent",title="% de points gagnés au 1er service par surface",color='group',facet_col='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        ###############################################
        # % DE POINTS GAGNES AU 2ND SERVICE PAR SURFACE
        ###############################################
        st.markdown('### % de points gagnés au 2nd service par surface')
        test = df_v4.groupby(['player', 'Surface','group']).mean()
        test = test.reset_index()
        pro2 = df_v4.groupby(['player', 'Surface','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        test['percent'] = pro2['2ndWon']
        fig = px.bar(test, x="player", y="percent",title="% de points gagnés au 2nd service par surface",color='group',facet_col='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        #############################################
        # NOMBRE DE BALLES DE BREK SAUVES PAR SURFACE
        #############################################
        st.markdown('### Nombre de balles de break sauvés par surface')
        test = df_v4.groupby(['player', 'Surface','group']).mean()
        test = test.reset_index()
        pro2 = df_v4.groupby(['player', 'Surface','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        test['percent'] = pro2['2ndWon']
        fig = px.bar(test, x="player", y="BpSaved",title="Nombre de balles de break sauvés par surface",color='group',facet_col='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)

        #######################################
        # NOMBRE DE BALLES DE BREAK PAR SURFACE
        #######################################
        st.markdown('### Nombre de balles de break par surface')
        test = df_v4.groupby(['player', 'Surface','group']).mean()
        test = test.reset_index()
        pro2 = df_v4.groupby(['player', 'Surface','group']).sum()
        pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
        pro2 = pro2.reset_index()
        test['percent'] = pro2['2ndWon']
        fig = px.bar(test, x="player", y="BpFaced",title="Nombre de balles de break par surface",color='group',facet_col='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, use_container_width=True)


    

