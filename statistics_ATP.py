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

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# PLAYER STATISTICS")

    concatenated = pd.read_csv("concatenated.csv")
    df_v3 = pd.read_csv("df_v3.csv")
    base = pd.read_csv("base.csv")
    df_v3['Date'] = pd.to_datetime(df_v3['Date'])
    df_v3 = df_v3.sort_values(by='Date') 

    st.markdown('## Rank player')

    P1_list = list(df_v3['player'].unique())
    with st.sidebar:
        P1 = st.selectbox(label = "Player", options = P1_list)

    query = f"player=='{P1}'"
    query2= f"Winner=='{P1}' | Loser=='{P1}'"
    df_filtered = df_v3.query(query)
    df_filtered2 = base.query(query2)
    fig = px.line(df_filtered, x='Date', y='Rank',color='player',title = 'Player rank')
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


    st.markdown('## Liste des adversaires')

    list_of_names = df_filtered2['Loser'].to_list()
    list_of_names2 = df_filtered2['Winner'].to_list()
    final_list = list_of_names + list_of_names2
    df10 = pd.DataFrame(final_list)
    df10.columns = ['test']
    df10.drop(df10[df10.test == P1].index, inplace=True)
    df11 = df10['test'].value_counts()
    df11 = df11.reset_index()
    df11.columns = ['index', 'last_results']

    df15 = df11[:10]
    fig = px.bar(df15, x="last_results", y="index",title='Top 10 adversaires')
    fig.update_layout(width=1000)
    fig.update_layout(height=700)


    st.plotly_chart(fig)

    st.markdown('## % victoire par surface et tournoi')

    pro = df_filtered.groupby(['Series', 'Surface','group']).size()
    pro = pro.reset_index()
    pro.columns = ['Series', 'Surface', 'group', 'value']
    pro2 = df_filtered.groupby(['Series', 'Surface','group']).sum()

    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['ATP']
    fig = px.pie(pro, values='value', names='group',color='group',facet_col='Series',facet_row='Surface')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    st.plotly_chart(fig, use_container_width=True)

    # fig = px.bar(pro, x="Series", y="value", color="group",title="Nombre Victoires par surface",text_auto=True, facet_col="Surface")
    # fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    # st.plotly_chart(fig, use_container_width=True)

    st.markdown('## % victoire par round en fonction de la surface')

    pro = df_filtered.groupby(['Round', 'Surface','group']).size()
    pro = pro.reset_index()
    pro.columns = ['Round','Surface', 'group', 'value']
    pro2 = df_filtered.groupby(['Round', 'Surface','group']).sum()

    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['ATP']
    fig = px.pie(pro, values='value', names='group',color='group',facet_col='Round',facet_row='Surface')

    #fig = px.bar(pro, x="Round", y="value", color="group",title="Nombre Victoires par round et surface",text_auto=True,facet_col="Surface")
    fig.update_layout(barmode='relative')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('## % victoire par round en fonction du type de tournoi')

    pro = df_filtered.groupby(['Round','Series','group']).size()
    pro = pro.reset_index()
    pro.columns = ['Round','Series', 'group', 'value']
    pro2 = df_filtered.groupby(['Round','Series','group']).sum()

    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['ATP']

    fig = px.pie(pro, values='value', names='group',color='group',facet_col='Round',facet_row='Series')

    # fig = px.bar(pro, x="Round", y="value", color="group",title="Nombre Victoires par round et tournoi",text_auto=True,facet_col="Series")
    fig.update_layout(barmode='relative')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('## % victoire par année')

    df_filtered['year'] = pd.to_datetime(df_filtered['Date']).dt.year

    pro = df_filtered.groupby(['year','group']).size()
    pro = pro.reset_index()
    pro.columns = ['year', 'group', 'value']
    pro2 = df_filtered.groupby(['year','group']).sum()

    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['ATP']
    pro[['year']] = pro[['year']].astype(str)

    fig = px.bar(pro, x="year", y="value", color="group",title="Nombre Victoires par année",text_auto=True)
    fig.update_layout(barmode='relative')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)


    pro = df_filtered.groupby(['year','Surface','group']).size()
    pro = pro.reset_index()
    pro.columns = ['year','Surface', 'group', 'value']
    pro2 = df_filtered.groupby(['year','Surface','group']).sum()

    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['ATP']
    pro[['year']] = pro[['year']].astype(str)

    fig = px.bar(pro, x="Surface", y="value", color="group",title="Nombre Victoires par année et surface",text_auto=True, facet_col="year")
    fig.update_layout(barmode='relative')
    fig.update_xaxes(matches=None, showticklabels=True,title_text=None)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    pro = df_filtered.groupby(['year','Series','group']).size()
    pro = pro.reset_index()
    pro.columns = ['year','Series', 'group', 'value']
    pro2 = df_filtered.groupby(['year','Series','group']).sum()

    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['ATP']
    pro[['year']] = pro[['year']].astype(str)

    fig = px.bar(pro, x="Series", y="value", color="group",title="Nombre Victoires par année et series",text_auto=True, facet_col="year")
    fig.update_layout(barmode='relative')
    fig.update_xaxes(matches=None, showticklabels=True,title_text=None)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('## Ten and Five last results')

    df_filtered2 = df_filtered2.tail(10)
    df_filtered2 = df_filtered2.drop(['ATP', 'Location','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','Comment','AvgW','AvgL','Court','Tournament','PSW','PSL','MaxL','MaxW','WPts','LPts','Location'],axis=1)
    df_filtered2 = df_filtered2.iloc[: , 1:]
    st.dataframe(df_filtered2)
    df_filtered2 = df_filtered2.tail(5)
    st.dataframe(df_filtered2)

    st.markdown('## % pronostic reussi par bookmaker')

    df_filtered2 = base.query(query2)
    df_filtered2 = df_filtered2.drop(['Location','W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','Comment','AvgW','AvgL','Court','Tournament','PSW','PSL','MaxL','MaxW','WPts','LPts','Location'],axis=1)
    df_filtered2 = df_filtered2.iloc[: , 1:]
    df_filtered2['prono'] = np.where(df_filtered2['B365W'] <= df_filtered2['B365L'], 'b_prono', 'm_prono')
    pro = df_filtered2.groupby(['Series','prono','Surface']).size()
    pro = pro.reset_index()
    pro.columns = ['Series','prono','Surface', 'value']
    pro2 = df_filtered2.groupby(['Series','Surface','prono']).sum()

    pro2 = pro2.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
    pro2 = pro2.reset_index()
    pro['percent'] = pro2['ATP']
    fig = px.pie(pro, values='value', names='prono',color=None,facet_col='Series',facet_row='Surface')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    st.plotly_chart(fig, use_container_width=True)



  