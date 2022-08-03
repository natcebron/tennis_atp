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
import cv2 as cv
import plotly.express as px

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)

    ##########################
    # IMPORTATION DES FICHIERS
    ##########################
    base = pd.read_csv("WTA/df_versus.csv")
    d_exploration = pd.read_csv("WTA/df_v3.csv")
    d_exploration['Date_x'] = pd.to_datetime(d_exploration['Date_x'])
    d_exploration = d_exploration.sort_values(by='Date_x') 


    #############
    # RANK PLAYER
    #############
    st.markdown('## Rank player')
    P1_list = list(d_exploration['player'].unique())
    with st.sidebar:
        P1 = st.selectbox(label = "Player", options = P1_list)
    query = f"player=='{P1}'"
    query2= f"Winner=='{P1}' | Loser=='{P1}'"
    df_filtered = d_exploration.query(query)
    df_filtered2 = base.query(query2)
    fig = px.line(df_filtered, x='Date_x', y='Rank',color='player',title = 'Player rank')
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    #######################
    # LISTE DES ADVERSAIRES
    #######################
    st.markdown('## Liste des adversaires')
    list_of_names = df_filtered2['Loser'].to_list()
    list_of_names2 = df_filtered2['Winner'].to_list()
    final_list = list_of_names + list_of_names2
    adv = pd.DataFrame(final_list)
    adv.columns = ['test']
    adv.drop(adv[adv.test == P1].index, inplace=True)
    adv_2 = adv['test'].value_counts()
    adv_2 = adv_2.reset_index()
    adv_2.columns = ['index', 'last_results']
    adv_f = adv_2[:10]
    fig = px.bar(adv_f, x="last_results", y="index",title='Top 10 adversaires')
    fig.update_layout(width=1000)
    fig.update_layout(height=700)
    st.plotly_chart(fig)

    ###################################
    # % VICTOIRE PAR SURFACE ET TOURNOI
    ###################################
    st.markdown('## % victoire par surface et tournoi')
    pro = df_filtered.groupby(['Tier', 'Surface','group']).size()
    pro = pro.reset_index()
    pro.columns = ['Tier', 'Surface', 'group', 'value']
    pro2 = df_filtered.groupby(['Tier', 'Surface','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['WTA']
    fig = px.pie(pro, values='value', names='group',color='group',facet_col='Tier',facet_row='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)
    # fig = px.bar(pro, x="Tier", y="value", color="group",title="Nombre Victoires par surface",text_auto=True, facet_col="Surface")
    # fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    # st.plotly_chart(fig, use_container_width=True)

    ################################################
    # % VICTOIRE PAR ROUND EN FONCTION DE LA SURFACE
    ################################################
    st.markdown('## % victoire par round en fonction de la surface')
    pro = df_filtered.groupby(['Round', 'Surface','group']).size()
    pro = pro.reset_index()
    pro.columns = ['Round','Surface', 'group', 'value']
    pro2 = df_filtered.groupby(['Round', 'Surface','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['WTA']
    fig = px.pie(pro, values='value', names='group',color='group',facet_col='Round',facet_row='Surface',color_discrete_map={'Winner': 'green','Loser': 'red'})
    #fig = px.bar(pro, x="Round", y="value", color="group",title="Nombre Victoires par round et surface",text_auto=True,facet_col="Surface")
    fig.update_layout(barmode='relative')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    #####################################################
    # % VICTOIRE PAR ROUND EN FONCTION DU TYPE DE TOURNOI
    #####################################################
    st.markdown('## % victoire par round en fonction du type de tournoi')
    pro = df_filtered.groupby(['Round','Tier','group']).size()
    pro = pro.reset_index()
    pro.columns = ['Round','Tier', 'group', 'value']
    pro2 = df_filtered.groupby(['Round','Tier','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['WTA']
    fig = px.pie(pro, values='value', names='group',color='group',facet_col='Round',facet_row='Tier',color_discrete_map={'Winner': 'green','Loser': 'red'})
    # fig = px.bar(pro, x="Round", y="value", color="group",title="Nombre Victoires par round et tournoi",text_auto=True,facet_col="Tier")
    fig.update_layout(barmode='relative')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)


    ######################
    # % VICTOIRE PAR ANNEE
    ######################
    st.markdown('## % victoire par année')
    df_filtered['year'] = pd.to_datetime(df_filtered['Date_x']).dt.year
    pro = df_filtered.groupby(['year','group']).size()
    pro = pro.reset_index()
    pro.columns = ['year', 'group', 'value']
    pro2 = df_filtered.groupby(['year','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['WTA']
    pro[['year']] = pro[['year']].astype(str)
    fig = px.bar(pro, x="year", y="value", color="group",title="Nombre Victoires par année",text_auto=True,color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.update_layout(barmode='relative')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ###############################
    # NOMBRE DE VICTOIRES PAR ANNEE
    ###############################
    pro = df_filtered.groupby(['year','Surface','group']).size()
    pro = pro.reset_index()
    pro.columns = ['year','Surface', 'group', 'value']
    pro2 = df_filtered.groupby(['year','Surface','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['WTA']
    pro[['year']] = pro[['year']].astype(str)
    fig = px.bar(pro, x="Surface", y="value", color="group",title="Nombre Victoires par année et surface",text_auto=True, facet_col="year",color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.update_layout(barmode='relative')
    fig.update_xaxes(matches=None, showticklabels=True,title_text=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    #########################################
    # NOMBRE DE VICTOIRES PAR ANNEE ET SERIES
    #########################################
    pro = df_filtered.groupby(['year','Tier','group']).size()
    pro = pro.reset_index()
    pro.columns = ['year','Tier', 'group', 'value']
    pro2 = df_filtered.groupby(['year','Tier','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['WTA']
    pro[['year']] = pro[['year']].astype(str)
    fig = px.bar(pro, x="Tier", y="value", color="group",title="Nombre Victoires par année et Tier",text_auto=True, facet_col="year",color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.update_layout(barmode='relative')
    fig.update_xaxes(matches=None, showticklabels=True,title_text=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ###########################
    # TEN AND FIVE LAST RESULTS
    ###########################
    st.markdown('## Ten and Five last results')
    df_filtered2 = df_filtered2.tail(10)
    #df_filtered2 = df_filtered2.drop(['WTA', 'Location','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','Comment','AvgW','AvgL','Court','Tournament','PSW','PSL','MaxL','MaxW','WPts','LPts','Location'],axis=1)
    df_filtered2 = df_filtered2.iloc[: , 1:]
    st.dataframe(df_filtered2)
    df_filtered2 = df_filtered2.tail(5)
    st.dataframe(df_filtered2)

    ##################################
    # % PRONOSTIC REUSSI PAR BOOKMAKER
    ##################################
    st.markdown('## % pronostic reussi par bookmaker')
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
    fig = px.pie(pro, values='value', names='prono',color='prono',facet_col='Tier',facet_row='Surface',color_discrete_map={'b_prono': 'green','m_prono': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ############
    # ODDS MOYEN
    ############
    st.markdown('## Odds moyen')
    test = df_filtered.groupby(['Tier', 'Surface']).mean()
    test = test.reset_index()
    fig = px.bar(test, x="Surface", y="Odds",title='Odds par Tier et surface',color='Surface',facet_col='Tier')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ##############################################
    # TEMPS MOYEN EN MINUTES PAR SURFACE ET SERIES
    ##############################################
    st.markdown('## Moyenne durée en Minutes par surface et Tier')
    est = df_filtered.groupby(['Tier', 'Surface']).mean()
    test = test.reset_index()
    fig = px.bar(test, x="Surface", y="minutes",title='Durée match en minutes par Tier et surface',color='Surface',facet_col='Tier')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ###################################
    # MOYENNE ACE PAR SURFACE ET SERIES
    ###################################
    st.markdown('## Moyenne ace par surface et Tier')
    test = df_filtered.groupby(['Tier', 'Surface']).mean()
    test = test.reset_index()
    fig = px.bar(test, x="Surface", y="Ace",title="Nombre moyen d'ace par Tier et surface",color='Surface',facet_col='Tier')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)


    ###################################################
    # NOMBRE MOYEN DOUBLES FAUTES PAR SURFACE ET SERIES
    ###################################################
    st.markdown('## Nombre moyenne de doubles fautes par surface et Tier')
    test = df_filtered.groupby(['Tier', 'Surface','group']).mean()
    test = test.reset_index()
    fig = px.bar(test, x="Surface", y="Df",title="Nombre moyen doubles fautes par Tier et surface",color='group',facet_col='Tier',color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ###########################################################
    # NOMBRE MOYENNE DE POINTS AU SERVICE PAR SURFACE ET SERIES
    ###########################################################
    st.markdown('## Moyenne points au service par surface et Tier')
    test = df_filtered.groupby(['Tier', 'Surface','group']).mean()
    test = test.reset_index()
    fig = px.bar(test, x="Surface", y="Svpt",title="Nombre moyen points sur service par Tier et surface",color='group',facet_col='Tier',color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ###############################################
    # % DE 1ER SERVICE REUSSI PAR SURFACE ET SERIES
    ###############################################
    st.markdown('## % de 1er service réussi par surface et Tier')
    test = df_filtered.groupby(['Tier', 'Surface','group']).mean()
    test = test.reset_index()
    pro2 = df_filtered.groupby(['Tier', 'Surface','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    test['percent'] = pro2['1stIn']
    fig = px.bar(test, x="Surface", y="percent",title="% de 1er service réussi par surface et Tier",color='group',facet_col='Tier',color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    #########################################################
    # % DE POINTS GAGNES AU 1ER SERVICE PAR SURFACE ET SERIES
    #########################################################
    st.markdown('## % de points gagnés au 1er service par surface et Tier')
    test = df_filtered.groupby(['Tier', 'Surface','group']).mean()
    test = test.reset_index()
    pro2 = df_filtered.groupby(['Tier', 'Surface','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    test['percent'] = pro2['1stWon']
    fig = px.bar(test, x="Surface", y="percent",title="% de points gagnés au 1er service par surface et Tier",color='group',facet_col='Tier',color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    #######################################################
    # % DE POINTS GAGNES AU 2ND SERVICE PAR SURFACE ET TIER
    #######################################################
    st.markdown("## % de points gagnés au 2nd service par surface et Tier")
    test = df_filtered.groupby(['Tier', 'Surface','group']).mean()
    test = test.reset_index()
    pro2 = df_filtered.groupby(['Tier', 'Surface','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    test['percent'] = pro2['2ndWon']
    fig = px.bar(test, x="Surface", y="percent",title="% de points gagnés au 2nd service par surface et Tier",color='group',facet_col='Tier',color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ######################################################
    # NOMBRE DE BALLES DE BREAK SAUVES PAR SURFACE ET TIER
    ######################################################
    st.markdown('## Nombre de balles de break sauvés par surface et Tier')
    test = df_filtered.groupby(['Tier', 'Surface','group']).mean()
    test = test.reset_index()
    pro2 = df_filtered.groupby(['Tier', 'Surface','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    test['percent'] = pro2['2ndWon']
    fig = px.bar(test, x="Surface", y="BpSaved",title="Nombre de balles de break sauvés par surface et Tier",color='group',facet_col='Tier',color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    ###############################################
    # NOMBRE DE BALLES DE BREAK PAR SURFACE ET TIER
    ###############################################
    st.markdown('## Nombre de balles de break par surface et Tier')
    test = df_filtered.groupby(['Tier', 'Surface','group']).mean()
    test = test.reset_index()
    pro2 = df_filtered.groupby(['Tier', 'Surface','group']).sum()
    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    test['percent'] = pro2['2ndWon']
    fig = px.bar(test, x="Surface", y="BpFaced",title="Nombre de balles de break par surface et Tier",color='group',facet_col='Tier',color_discrete_map={'Winner': 'green','Loser': 'red'})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

