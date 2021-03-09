import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import catboost as cb
import catboost.utils as cbu
import numpy as np
import pandas as pd
import hyperopt
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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

#if not teams:
#    st.error("Please select a team")
    
#if teams:
#    df = data[data['team'].isin(teams)]
    #st.write('### Team Data', df)
#else:
#    st.error("Please select a team")    
#if not bat:
#    st.error("Please select a bat")
    
#if bat:
#    new_df = df[df['bat'].isin(bat)]
    
#else:
#    st.error("Please select a Bat")
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


select_box = st.sidebar.selectbox(label='Team', options=teams)
select_box1 = st.sidebar.selectbox(label='X axis', options=columns)
select_box2 = st.sidebar.selectbox(label="Hue", options=columns)
select_box3 = st.sidebar.selectbox(label='Toss', options=toss)
select_box4 = st.sidebar.selectbox(label='bat', options=bat)
select_box5 = st.sidebar.selectbox(label='Groung', options=ground)
select_box6 = st.sidebar.selectbox(label='result', options=result)
select_box7 = st.sidebar.selectbox(label='date', options=date)

#    st.error("Please select a bat")

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
# plot_chart(new_df, 'toss')
title_list.append('Toss Result')

new_df1 = df[df['bat'].isin([select_box4])]
# plot_chart(new_df1, 'bat')
title_list.append('Batting Order')


new_df2 = df[df['ground'].isin([select_box5])]
# plot_chart(new_df2, 'Ground')
title_list.append('Ground')


new_df3 = df[df['result'].isin([select_box6])]
# plot_chart(new_df3, 'Result')
title_list.append('Match Outcome')


new_df4 = df[df['date'].isin([select_box7])]
# plot_chart(new_df4, 'Date')
title_list.append('Historic Year')

plot_df.append(df)
plot_df.append(new_df)
plot_df.append(new_df1)
plot_df.append(new_df2)
plot_df.append(new_df3)
plot_df.append(new_df4)

plot_chart(plot_df, title_list)



#  Preprocessing

df.to_csv('./raw_data.csv')

raw_df = df.copy()
raw_df.drop(columns=['margin'], inplace=True)

import catboost as cb
X= raw_df.drop(columns='result')
y= raw_df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train(X, y):
    dataset = cb.Pool(X, y, cat_features=np.where(X.dtypes != np.float)[0])
    model = cb.CatBoostClassifier(iterations=70)
    model.fit(dataset, verbose=2)
    return model
    
model = train(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

st.text("Classification Report on Training Dataset")
st.text(classification_report(y_train, model.predict(X_train)))

cf_matrix = confusion_matrix(y_train, model.predict(X_train))



st.text("Classification Report on Test Dataset")
st.text(classification_report(y_test, model.predict(X_test)))

cf_matrix_test = confusion_matrix(y_test, model.predict(X_test))


# fig_cf, ax_cf = plt.subplots(1, 2)
# ax_cf[0].set_title("Training Confusion Matrix")
# ax_cf[0] = sns.heatmap(cf_matrix, annot=True, ax=ax_cf[0])

# ax_cf[1].set_title("Test Confusion Matrix")
# ax_cf[1] = sns.heatmap(confusion_matrix(y_test, model.predict(X_test)),
#                        annot=True, ax= ax_cf[1])
# st.pyplot(fig_cf)

