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
from PIL import Image, ImageStat,ImageOps
import matplotlib.image as mpimg
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models,utils
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils
import keras
import matplotlib.cm as cm
import tensorflow

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# Data exploration")
    concatenated = pd.read_csv("concatenated.csv")
    df_v3 = pd.read_csv("df_v3.csv")
    base = pd.read_csv("base.csv")
    df_v3['Date'] = pd.to_datetime(df_v3['Date'])
    df_v3 = df_v3.sort_values(by='Date') 



    series_list = list(df_v3['Series'].unique())
    player_list = list(df_v3['player'].unique())
    surface_list = list(df_v3['Surface'].unique())
    round_list = list(df_v3['Round'].unique())
    col1,col2 = st.columns(2)
    with col1:
        series = st.selectbox(label = "Choose a series", options = series_list)
        surface = st.selectbox(label = "Choose a surface", options = surface_list)

    with col2:
        player = st.selectbox(label = "Choose a player", options = player_list)
        round2 = st.selectbox(label = "Choose a round", options = round_list)


    query = f"Series=='{series}' & player=='{player}'& Surface=='{surface}'& Round=='{round2}'"
    df_filtered = df_v3.query(query)
    st.dataframe(df_filtered)

            
         

    
