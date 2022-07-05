import streamlit as st 

import os 


import introduction_ATP                     #+Deployment
import statistics_ATP
import versus_ATP
import preprocessing_ATP
import group_ATP
import config
import member
from collections import OrderedDict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
logo = os.path.join(currentdir, 'data/covid_1.png')
PAGE_CONFIG = {"page_title":"DeMACIA-RX.io","page_icon": logo,"layout":"wide"}
st.set_page_config(**PAGE_CONFIG)

ATP = {
    "Introduction" : introduction_ATP,
    "Data exploration" : preprocessing_ATP,
    "Player statistics" : statistics_ATP,
    "Player versus" : versus_ATP,
    "Match prediction" : group_ATP}

st.sidebar.title('SOMMAIRE')
selection_page = st.sidebar.radio("",list(ATP.keys()))
page = ATP[selection_page]
page.app()


st.sidebar.markdown("### DEVELOPER:")
for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)


