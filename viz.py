import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

st.title('Cricket Analytics')

def load_data():
    df = pd.read_csv("Clean_data.csv")
    return df

data = load_data()
columns = data.select_dtypes(['float64', 'int64', 'object']).columns
#teams = data.team

checkbox = st.sidebar.checkbox("Reveal data.")

if checkbox:
    # st.write(data)
    st.dataframe(data=data)

#Histogram

st.sidebar.subheader("Count_plot")
new_df = data.loc[data['team'] == 'India']
#select_box = st.sidebar.selectbox(label='Teams', options=team)
select_box1 = st.sidebar.selectbox(label='X axis', options=columns)
select_box2 = st.sidebar.selectbox(label="Hue", options=columns)
sns.countplot(x=select_box1,data=new_df,hue=select_box2)
st.pyplot()

