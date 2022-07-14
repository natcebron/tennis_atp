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

    d_exploration = pd.read_csv("WTA/df_v3.csv")
    base = pd.read_csv("base.csv")
    d_exploration['Date_x'] = pd.to_datetime(d_exploration['Date_x'])
    d_exploration = d_exploration.sort_values(by='Date_x') 
    d_exploration = d_exploration.iloc[: , 1:]



    series_list = list(d_exploration['Tier'].unique())
    player_list = list(d_exploration['player'].unique())
    surface_list = list(d_exploration['Surface'].unique())
    round_list = list(d_exploration['Round'].unique())
    col1,col2 = st.columns(2)
    with col1:
        series = st.selectbox(label = "Choose a series", options = series_list)
        surface = st.selectbox(label = "Choose a surface", options = surface_list)

    with col2:
        player = st.selectbox(label = "Choose a player", options = player_list)
        round2 = st.selectbox(label = "Choose a round", options = round_list)


    query = f"Tier=='{series}' & player=='{player}'& Surface=='{surface}'& Round=='{round2}'"
    df_filtered = d_exploration.query(query)
    st.dataframe(df_filtered)

            
         

    
