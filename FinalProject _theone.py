#!/usr/bin/env python
# coding: utf-8

# In[263]:


import pandas as pd
from IPython.display import FileLink
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[264]:


NFL = pd.read_csv("Data/out.csv")


# In[265]:


QBR_weekly = pd.read_csv('Data/qbr_week_level.csv')


# In[266]:


QBR_weekly = QBR_weekly.replace(['Chicago Bears'], 'Bears')
QBR_weekly = QBR_weekly.replace(['Philadelphia Eagles'], 'Eagles')
QBR_weekly = QBR_weekly.replace(['New York Jets'], 'Jets')
QBR_weekly = QBR_weekly.replace(['Indianapolis Colts'], 'Colts')
QBR_weekly = QBR_weekly.replace(['Atlanta Falcons'], 'Falcons')
QBR_weekly = QBR_weekly.replace(['Pittsburgh Steelers'], 'Steelers')
QBR_weekly = QBR_weekly.replace(['Baltimore Ravens'], 'Ravens')
QBR_weekly = QBR_weekly.replace(['Jacksonville Jaguars'], 'Jaguars')
QBR_weekly = QBR_weekly.replace(['Minnesota Vikings'], 'Vikings')
QBR_weekly = QBR_weekly.replace(['Buffalo Bills'], 'Bills')
QBR_weekly = QBR_weekly.replace(['Washington Redskins'], 'Redskins')
QBR_weekly = QBR_weekly.replace(['New York Giants'], 'Giants')
QBR_weekly = QBR_weekly.replace(['Dallas Cowboys'], 'Cowboys')
QBR_weekly = QBR_weekly.replace(['New England Patriots'], 'Patriots')
QBR_weekly = QBR_weekly.replace(['New Orleans Saints'], 'Saints')
QBR_weekly = QBR_weekly.replace(['Houston Texans'], 'Texans')
QBR_weekly = QBR_weekly.replace(['San Francisco 49ers'], '49ers')
QBR_weekly = QBR_weekly.replace(['Arizona Cardinals'], 'Cardinals')
QBR_weekly = QBR_weekly.replace(['Detroit Lions'], 'Lions')
QBR_weekly = QBR_weekly.replace(['Cleveland Browns'], 'Browns')
QBR_weekly = QBR_weekly.replace(['Cincinati Bengals'], 'Bengals')
QBR_weekly = QBR_weekly.replace(['Seattle Seahawks'], 'Seahawks')
QBR_weekly = QBR_weekly.replace(['Green Bay Packers'], 'Packers')
QBR_weekly = QBR_weekly.replace(['Miami Dolphins'], 'Dolphins')
QBR_weekly = QBR_weekly.replace(['Kansas City Chiefs'], 'Cheifs')
QBR_weekly = QBR_weekly.replace(['Oakland Raiders'], 'Raiders')
QBR_weekly = QBR_weekly.replace(['Tennessee Titans'], 'Titans')
QBR_weekly = QBR_weekly.replace(['Tampa Bay Buccaneers'], 'Buccaneers')
QBR_weekly = QBR_weekly.replace(['Carolina Panthers'], 'Panthers')
QBR_weekly = QBR_weekly.replace(['Denver Broncos'], 'Broncos')
QBR_weekly = QBR_weekly.replace(['San Diego Chargers'], 'Chargers')
QBR_weekly = QBR_weekly.replace(['Los Angeles Rams'], 'Rams')
QBR_weekly = QBR_weekly.replace(['Los Angeles Chargers'], 'Chargers')
QBR_weekly = QBR_weekly.replace(['St. Louis Rams'], 'Rams')
QBR_weekly = QBR_weekly.replace(['Las Vegas Raiders'], 'Raiders')
QBR_weekly = QBR_weekly.replace(['Washington Commanders'], 'Commanders')


# In[267]:


QBR_weekly1 = ['season', 'season_type', 'game_id', 'game_week', 'week_text', 'rank', 'qbr_total', 'pass', 'run','sack','name_first','name_last','team', 'opp_team']
QBR_weekly_df = pd.DataFrame(QBR_weekly, columns=QBR_weekly1)

qbr_sorted = QBR_weekly_df.sort_values(by=['season', 'game_id'])
qbr_sorted['Name'] = qbr_sorted['name_first'] + ' ' + qbr_sorted['name_last']
qbr_sorted.drop(['name_last', 'name_first'], axis=1, inplace=True)
qbr_sorted.reset_index(inplace=True)
qbr_sorted.drop(columns=['index'], inplace=True)


# In[268]:


NFL = NFL.replace(['Chicago Bears'], 'Bears')
NFL = NFL.replace(['Philadelphia Eagles'], 'Eagles')
NFL = NFL.replace(['New York Jets'], 'Jets')
NFL = NFL.replace(['Indianapolis Colts'], 'Colts')
NFL = NFL.replace(['Atlanta Falcons'], 'Falcons')
NFL = NFL.replace(['Pittsburgh Steelers'], 'Steelers')
NFL = NFL.replace(['Baltimore Ravens'], 'Ravens')
NFL = NFL.replace(['Jacksonville Jaguars'], 'Jaguars')
NFL = NFL.replace(['Minnesota Vikings'], 'Vikings')
NFL = NFL.replace(['Buffalo Bills'], 'Bills')
NFL = NFL.replace(['Washington Redskins'], 'Redskins')
NFL = NFL.replace(['New York Giants'], 'Giants')
NFL = NFL.replace(['Dallas Cowboys'], 'Cowboys')
NFL = NFL.replace(['New England Patriots'], 'Patriots')
NFL = NFL.replace(['New Orleans Saints'], 'Saints')
NFL = NFL.replace(['Houston Texans'], 'Texans')
NFL = NFL.replace(['San Francisco 49ers'], '49ers')
NFL = NFL.replace(['Arizona Cardinals'], 'Cardinals')
NFL = NFL.replace(['Detroit Lions'], 'Lions')
NFL = NFL.replace(['Cleveland Browns'], 'Browns')
NFL = NFL.replace(['Cincinnati Bengals'], 'Bengals')
NFL = NFL.replace(['Seattle Seahawks'], 'Seahawks')
NFL = NFL.replace(['Green Bay Packers'], 'Packers')
NFL = NFL.replace(['Miami Dolphins'], 'Dolphins')
NFL = NFL.replace(['Kansas City Chiefs'], 'Cheifs')
NFL = NFL.replace(['Oakland Raiders'], 'Raiders')
NFL = NFL.replace(['Tennessee Titans'], 'Titans')
NFL = NFL.replace(['Tampa Bay Buccaneers'], 'Buccaneers')
NFL = NFL.replace(['Carolina Panthers'], 'Panthers')
NFL = NFL.replace(['Denver Broncos'], 'Broncos')
NFL = NFL.replace(['San Diego Chargers'], 'Chargers')
NFL = NFL.replace(['Los Angeles Rams'], 'Rams')
NFL = NFL.replace(['Los Angeles Chargers'], 'Chargers')
NFL = NFL.replace(['St. Louis Rams'], 'Rams')
NFL = NFL.replace(['Las Vegas Raiders'], 'Raiders')
NFL = NFL.replace(['Washington Commanders'], 'Commanders')
NFL = NFL.replace(['Washington Football Team'], 'Commanders')


# In[269]:


NFL['team_home'].value_counts()


# In[270]:


import pandas as pd


merged_rows = []

prev_row = None

for index, row in qbr_sorted.iterrows():
    if index % 2 == 0:
        prev_row = row
    else:
        merged_row = prev_row.copy()
        merged_row['rank_2'] = row['rank']
        merged_row['qbr_total_2'] = row['qbr_total']
        merged_row['pass_2'] = row['pass']
        merged_row['run_2'] = row['run']
        merged_row['sack_2'] = row['sack']
        merged_row['team_2'] = row['team']
        merged_row['opp_team_2'] = row['opp_team']
        merged_row['Name_2'] = row['Name']
        merged_rows.append(merged_row)

merged_df = pd.DataFrame(merged_rows)
merged_df.reset_index(drop=True, inplace=True)


# In[271]:


merged_df.head()


# In[272]:


merged_df['friendly_rank'] = merged_df['rank']
merged_df['friendly_qbr_total'] = merged_df['qbr_total']
merged_df['friendly_name'] = merged_df['Name']
merged_df['Opposing_rank'] = merged_df['rank_2']
merged_df['Opposing_qbr_total'] = merged_df['qbr_total_2']
merged_df['Opposing_QB_Name'] = merged_df['Name_2']
merged_df.drop(['pass', 'Name','qbr_total', 'run', 'sack', 'rank_2', 'qbr_total_2', 'pass_2', 'run_2', 'sack_2', 'team_2', 'opp_team_2', 'Name_2'], axis=1, inplace=True)


# In[273]:


df_clean = merged_df.dropna()


# In[274]:


df_clean.tail()


# In[13]:


df_clean.to_csv('Data/df_clean.csv')


# In[275]:


NFL.drop(['Unnamed: 0'], axis =1 , inplace = True)


# In[276]:


NFL.head()


# In[277]:


NFL['Merge_column'] = NFL['schedule_season'].astype(str) + ',' + NFL['schedule_week'].astype(str) + ',' + NFL['team_home'] + ',' + NFL['team_away']
NFL['Merge_column'] = NFL['Merge_column'].apply(lambda x: ','.join(sorted(x.split(','))))
df_clean['Merge_column'] = df_clean['season'].astype(str) + ',' + df_clean['game_week'].astype(str) + ',' + df_clean['team'] + ',' + df_clean['opp_team']
df_clean['Merge_column'] = df_clean['Merge_column'].apply(lambda x: ','.join(sorted(x.split(','))))


# In[278]:


merged_df = NFL.merge(df_clean, on='Merge_column', how='inner')
merged_df['Over/Under'].value_counts()


# In[18]:


merged_df.to_csv("Data/Merged_df.csv")


# In[279]:


merged_df['Underdog_Cover']


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


merged_df['Over/Under'] = merged_df['Over/Under'].apply(lambda x: 1 if x == 'Over' else 0)

numeric_feats = merged_df[['weather_temperature', 'weather_wind_mph', 'weather_humidity', 'rank',
                          'friendly_rank', 'friendly_qbr_total', 'Opposing_rank', 'Opposing_qbr_total']]

categorical_columns = ['team_home', 'team_away', 'stadium']
encoded_df = pd.get_dummies(merged_df[categorical_columns], columns=categorical_columns)

X = pd.concat([numeric_feats, encoded_df], axis=1)
y = merged_df['Over/Under']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


merged_df['Underdog_Cover'] = merged_df['Underdog_Cover'].apply(lambda x: 1 if x == 'Favorite Cover' else 0)

numeric_feats = merged_df[['weather_temperature', 'weather_wind_mph', 'weather_humidity', 'rank',
                          'friendly_rank', 'friendly_qbr_total', 'Opposing_rank', 'Opposing_qbr_total']]

categorical_columns = ['team_home', 'team_away', 'stadium']
encoded_df = pd.get_dummies(merged_df[categorical_columns], columns=categorical_columns)

X = pd.concat([numeric_feats, encoded_df], axis=1)
y = merged_df['Underdog_Cover']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")


# In[227]:


#Use a numpy where to create a column that shows how often the home team or away team covers the spread 


# ## Used SQL in SAS to make engineer new columns

# In[321]:


New_NFL = pd.read_csv('Data/NFL_merge.csv')
New_NFL.head()


# In[322]:


New_NFL['Merge_column'] = New_NFL['season'].astype(str) + ',' + New_NFL['schedule_week'].astype(str) + ',' + New_NFL['team_home'] + ',' + New_NFL['team_away']
merged_df['Merge_column'] = merged_df['schedule_season'].astype(str) + ',' + merged_df['schedule_week'].astype(str) + ',' + merged_df['team_home'] + ',' + merged_df['team_away']


# In[323]:


merged_df = merged_df.merge(New_NFL, on='Merge_column', how='inner')
merged_df['Over/Under'].value_counts()


# In[324]:


columns = [col for col in merged_df.columns if not col.endswith("_y")]
merged_df = merged_df[columns]
merged_df = merged_df.drop_duplicates()


# In[325]:


merged_df['Over/Under'].value_counts()


# In[328]:


merged_df.columns


# In[327]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))


sns.countplot(x='season_x', hue='Over/Under', data=merged_df)

plt.xlabel('Season')
plt.ylabel('Count')
plt.title('Distribution of Over/Under in Each Season')


plt.legend(title='Over/Under')


plt.show()


# In[334]:


season_stats = merged_df.groupby('schedule_season').agg({
    'Total_Score': 'mean',
    'over_under_line': 'mean'
}).reset_index()


plt.figure(figsize=(12, 6))

sns.barplot(x='schedule_season', y='Total_Score', label='Average Score', color='blue', data=season_stats, alpha=0.7)
sns.barplot(x='schedule_season', y='over_under_line', label='Average Over/Under line', color='orange', data=season_stats, alpha=0.5)


plt.xlabel('Season')
plt.ylabel('Mean Score')
plt.title('Average Score vs Average Over/Under Line Per Season')


plt.legend()

plt.show()


# In[337]:


team_overunder_count = merged_df.groupby('team_home')['Over/Under'].value_counts().unstack().fillna(0)
team_overunder_count['Over/Under_Count'] = team_overunder_count['Over'] - team_overunder_count['Under']
team_overunder_count


# In[338]:


merged_df.groupby('team_home').agg({
    'score_away': 'mean',
}).reset_index()


# In[345]:


team_overunder_count = merged_df.groupby('team_home')['Over/Under'].value_counts().unstack().fillna(0)
team_overunder_count['Over/Under_Count'] = team_overunder_count['Over'] - team_overunder_count['Under']

away_stats = merged_df.groupby('team_home').agg({
    'score_away': 'mean',
}).reset_index()


merged_data = pd.merge(away_stats, team_overunder_count, left_on='team_home', right_index=True)


plt.figure(figsize=(12, 8), facecolor='#f3f3f3') 


sns.scatterplot(x='score_away', y='Over/Under_Count', size='Over', data=merged_data, hue='team_home', palette='Set1', sizes=(30, 300), legend=False)


for line in range(0, merged_data.shape[0]):
    plt.text(merged_data['score_away'][line] + 0.01, merged_data['Over/Under_Count'][line], merged_data['team_home'][line], horizontalalignment='left', size='small', color='black')

# Set plot labels and title
plt.xlabel('Does Defense Matter')
plt.ylabel('Count of Over')
plt.title('Average Points Allowed by Away Team vs. Count of Over/Under by Team')

# Show the plot
plt.show()


# In[299]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Only run this line below if you have not ran the models above
#merged_df['Over/Under'] = merged_df['Over/Under'].apply(lambda x: 1 if x == 'Over' else 0)

numeric_feats = merged_df[['weather_temperature', 'weather_wind_mph', 'weather_humidity',
                           'friendly_qbr_total', 'Opposing_qbr_total','over_under_line','Avg_Homescore_season',
                          'Avg_PPG_Scoredon_athome','Avg_Awayscore_season','Avg_PPG_Scoredon_ataway']]

categorical_columns = ['team_home_x', 'team_away_x', 'stadium','season_x']
encoded_df = pd.get_dummies(merged_df[categorical_columns], columns=categorical_columns)

X = pd.concat([numeric_feats, encoded_df], axis=1)
y = merged_df['Over/Under']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")


# In[310]:


X_train.columns


# In[318]:


coef = model.coef_
a = pd.DataFrame(coef, columns = X_train.columns)
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': abs(a.iloc[0])}).reset_index(drop=True)
feature_importance_df.sort_values(by='Coefficient', ascending=False)[:5]


# In[286]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

confusion_matrix(y_test, y_pred)


# In[287]:


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.show()


# ## Creating a Calculator like this

# In[288]:


X_test = X[-50:]
y_test = y[-50:]


# In[289]:


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")


# In[290]:


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.show()


# In[291]:


Calculator = confusion_matrix(y_test, y_pred)
Calculator


# In[292]:


confusion_matrix(y_test, y_pred)


# In[293]:


(Calculator[0, 0] - Calculator[0, 1]) + Calculator[1, 1] - Calculator[1, 0]


# In[294]:


def Calculator(Games_played,UnitSize):
    
    X_test = X[-Games_played:]
    y_test = y[-Games_played:]
    y_pred = model.predict(X_test)
    Calculator = confusion_matrix(y_test, y_pred)
    a = ((Calculator[0, 0] - Calculator[0, 1]) + (Calculator[1, 1] - Calculator[1, 0]))*UnitSize
    
    print("Total Amount Up/Down: $",a)


# In[295]:


Calculator(50,10)


# In[296]:


Calculator(25,100)


# In[297]:


Calculator(16,100)


# In[298]:


Calculator(100,100)

