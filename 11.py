import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('CRICKET ANALYTICS')

def load_data():
    df = pd.read_csv("Clean_data.csv")
    return df

data = load_data()

teams = st.multiselect(
    "Select Any of The Following TEAMS For Analysis", list(data.team.unique()), ['Australia']
)

toss = st.multiselect(
    "Select TOSS RESULT For Analysis", list(data.toss.unique()),
)

bat = st.multiselect(
    "Select BATTING or BOWLING For Analysis", list(data.bat.unique()),
)

ground = st.multiselect(
    "Select Any of The Following GROUNDS For Analysis", list(data.ground.unique()),
)

result = st.multiselect(
    "Select Any of The Following RESULT For Analysis", list(data.result.unique()),
)

date = st.multiselect(
    "Select Any of The Following YEAR For Analysis", list(data.date.unique()), [2021]
)

if not teams:
    st.error("Please select a team")
    
else:
    df = data[data['team'].isin(teams)]
    #st.write('### Team Data', df)
    
if not bat:
    st.error("Please select a team")
    
else:
    df = data[data['bat'].isin(bat)]
    
    #team_df = []
    #for t in teams:
    #    d = df.copy()
    #    d = d[d['team'] == t]
    #    t_df = pd.DataFrame(
    #        {
    #            t : d.ground.values
    #            }
    #        )
    #    team_df.append(t_df)
    #    
    #chart_data = pd.concat(team_df, axis = 1).dropna()
    #print(chart_data.head())
    #st.write(chart_data)
    




columns = data.select_dtypes(['float64', 'int64', 'object']).columns
#teams = data.team

checkbox = st.sidebar.checkbox("Reveal data.")

if checkbox:
     st.write(data)
     st.dataframe(data=data)

#Histogram

st.sidebar.subheader("Count_plot")
#new_df = data.loc[data['team'] == 'India']
select_box = st.sidebar.selectbox(label='Teams', options=teams)
select_box3 = st.sidebar.selectbox(label='toss', options=toss)
select_box1 = st.sidebar.selectbox(label='X axis', options=columns)
select_box2 = st.sidebar.selectbox(label="Hue", options=columns)
plt.figure(figsize=(20,10), facecolor='w')
#plt.xlabel(xlab, size=10)
#plt.ylabel(ylab, size=10)
plt.xticks(size=20)
plt.yticks(size=20)
sns.countplot(x=select_box1,data=df,hue=select_box2)
st.pyplot()
#################

