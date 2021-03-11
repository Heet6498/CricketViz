import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.title('CRICKET ANALYTICS')
st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
    df = pd.read_csv("Clean_data.csv")
    return df

data = load_data()

teams = st.multiselect(
    "Select Any of The Following TEAMS For Analysis", list(data.team.unique()), ['Australia', 'India', 'England']
)

toss = st.multiselect(
    "Select TOSS RESULT For Analysis", list(data.toss.unique()), [1.0, 0.0]
)

bat = st.multiselect(
    "Select BATTING or BOWLING For Analysis", list(data.bat.unique()), [1.0]
)

ground = st.multiselect(
    "Select Any of The Following GROUNDS For Analysis", list(data.ground.unique()), ['Sharjah']
)

result = st.multiselect(
    "Select Any of The Following RESULT For Analysis", list(data.result.unique()), [1.0]
)

date = st.multiselect(
    "Select Any of The Following YEAR For Analysis", list(data.date.unique()), [2020 ]
)


columns = data.select_dtypes(['float64', 'int64', 'object']).columns


checkbox = st.sidebar.checkbox("Reveal data.")

if checkbox:
     st.write(data)
     st.dataframe(data=data)



st.sidebar.subheader("Count_plot")

select_box = st.sidebar.selectbox(label='Team', options=teams)
select_box1 = st.sidebar.selectbox(label='X axis', options=columns)
select_box2 = st.sidebar.selectbox(label="Hue", options=columns)
select_box3 = st.sidebar.selectbox(label='Toss', options=toss)
select_box4 = st.sidebar.selectbox(label='bat', options=bat)
select_box5 = st.sidebar.selectbox(label='Groung', options=ground)
select_box6 = st.sidebar.selectbox(label='result', options=result)
select_box7 = st.sidebar.selectbox(label='date', options=date)



def plot_chart(df, title_list):
    #plt.figure(figsize=(20,10), facecolor='w')
    fig, ax = plt.subplots(3, 2, figsize=(10,10), facecolor='w' )
    plt.subplots_adjust(hspace = 0.4)
    
    ax_array = ax.reshape(-1)
    plt.xticks(size=10)
    plt.yticks(size=10)
    
    for i in range(0, len(df)):
        ax_array[i].set_title(title_list[i])
        sns.countplot(x=select_box1,data=df[i],hue=select_box2, ax=ax_array[i])
    st.pyplot(fig)
    


plot_df = []
title_list = []

df = data[data['team'].isin(teams)]
df = df.loc[df['opposition'] == select_box]   

title_list.append('One vs Many Analysis')


new_df = df[df['toss'].isin([select_box3])]
title_list.append('Toss Result')

new_df1 = df[df['bat'].isin([select_box4])]
title_list.append('Batting Order')


new_df2 = df[df['ground'].isin([select_box5])]
title_list.append('Ground')


new_df3 = df[df['result'].isin([select_box6])]
title_list.append('Match Outcome')


new_df4 = df[df['date'].isin([select_box7])]
title_list.append('Historic Year')

plot_df.append(df)
plot_df.append(new_df)
plot_df.append(new_df1)
plot_df.append(new_df2)
plot_df.append(new_df3)
plot_df.append(new_df4)

plot_chart(plot_df, title_list)



html = ''' 
<style>
footer {visibility:hidden;}
#MainMenu {visibility: hidden;}
           

body{background-color: #FCFFDE;}
     

</style>
'''

st.markdown(html,unsafe_allow_html=True)

