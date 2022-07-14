import streamlit as st 

import os 

import mise_a_jour_ATP
import introduction_ATP                     #+Deployment
import statistics_ATP
import versus_ATP
import preprocessing_ATP
import group_ATP
import inspect
import mise_a_jour_WTA
import introduction_WTA                    #+Deployment
import statistics_WTA
import versus_WTA
import preprocessing_WTA
import group_WTA


from collections import OrderedDict
import jour_ATP
import jour_WTA
import subprocess
import sys
from streamlit_option_menu import option_menu

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
logo = os.path.join(currentdir, 'data/covid_1.png')
import streamlit as st
import hydralit_components as hc
import datetime

st.set_page_config(layout="wide")

st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 1.8rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 1rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
# 1. as sidebar menu
with st.sidebar:
    selected = option_menu(
    menu_title='Menu',
    options=['ATP','WTA','Double'],
    icons=['file-earmark-easel','file-earmark-code','file-earmark-slides'],
    menu_icon='grid-1x2',default_index=0,    styles={
        "container": {"padding": "0!important", "background-color": "#0E1117"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "red"},
    })
    

# 3. CSS style definitions
selected2 = option_menu(None, ["Introduction","Data exploration","Player statistics","Player versus" ,"Match prediction",'Pronostic du jour','Mise à jour'], 
    icons=['house', 'bar-chart', "list-task", 'clipboard-data','cloud-arrow-up','collection-play','gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0E1117"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "red"},
    }
)


if selected =="ATP":
    if selected2 == "Introduction":
        introduction_ATP.app()
    if selected2 == "Data exploration":
        preprocessing_ATP.app()
    if selected2 == "Player statistics":
        statistics_ATP.app()
    if selected2 == "Player versus":
        versus_ATP.app()
    if selected2 == "Match prediction":
        group_ATP.app()
    if selected2 == "Mise à jour":
        mise_a_jour_ATP.app()
    if selected2 =="Pronostic du jour":
        jour_ATP.app()
if selected =="WTA":
    if selected2 == "Introduction":
        introduction_WTA.app()
    if selected2 == "Data exploration":
        preprocessing_WTA.app()
    if selected2 == "Player statistics":
        statistics_WTA.app()
    if selected2 == "Player versus":
        versus_WTA.app()
    if selected2 == "Match prediction":
        group_WTA.app()
    if selected2 == "Mise à jour":
        mise_a_jour_WTA.app()
    if selected2 =="Pronostic du jour":
        jour_WTA.app()



