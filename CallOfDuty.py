#!/usr/bin/env python
# coding: utf-8

# Link to data repo:
#     

# In[21]:


import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# # Read in data

# In[3]:


one = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2017-12-10-dallas.csv")
two = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-01-14-neworleans.csv")
three= pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-03-11-atlanta.csv")
four = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-04-01-birmingham.csv")
five = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-04-08-proleague1.csv")
six = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-04-22-seattle.csv")
seven = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-05-05-london.csv")
eight = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-06-16-anaheim.csv")
nine = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-07-05-proleague.csv")
ten = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2017-08-13-champs.csv")
onea = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-06-17-anaheim.csv")
twoa = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-07-29-proleague2.csv")
threea = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2018-08-19-champs.csv")
foura = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-01-20-proleague-qual.csv")
fivea = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-03-17-fortworth.csv")
sixa = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-07-21-proleague-finals.csv")
sevena = pd.read_csv("https://raw.githubusercontent.com/Activision/cwl-data/master/data/data-2019-08-18-champs.csv")

df = pd.concat([one, two, three, four, five, six, seven, eight, nine, ten, onea, twoa, threea, foura, fivea, sixa, sevena])


# # Initial Cleaning

# ## Drop inconsistent data

# In[4]:


df.drop(['fave payload', 'fave rig', 'fave trait'],axis =1, inplace = True)


# # Number of players

# *** Methodz and MethodZsixk is the same person***

# In[5]:


df['player'].nunique()


# # Number of Teams

# In[6]:


df['team'].nunique()


# # Base EDA

# In[7]:


g = sns.barplot(x = df['mode'].value_counts().index, y = df['mode'].value_counts())
g.tick_params(axis='x', rotation=90)


# ### Seperate the values based on matched

# In[8]:


hp = df[df['mode'] == "Hardpoint"].copy()
snd = df[df['mode'] == "Search & Destroy"].copy()
ctf = df[df['mode'] == "Capture The Flag"].copy()
ctrl = df[df['mode'] == "Control"].copy()
up = df[df['mode'] == "Uplink"].copy()


# # Clean data 

# ## Remove match specific data from other match types

# In[9]:


uplink_specific = ['uplink dunks', 'uplink throws','uplink points']
ctf_specific = ['ctf captures', 'ctf returns','ctf pickups', 'ctf defends', 'ctf kill carriers','ctf flag carry time (s)']
snd_specific = ['snd rounds', 'snd firstbloods', 'snd firstdeaths','snd survives', 'bomb pickups', 'bomb plants', 'bomb defuses','bomb sneak defuses', 'snd 1-kill round', 'snd 2-kill round','snd 3-kill round', 'snd 4-kill round']
ctrl_specific = ['ctrl rounds', 'ctrl firstbloods','ctrl firstdeaths', 'ctrl captures']
hp_specific = []


# In[10]:


up.drop(ctf_specific+snd_specific+ctrl_specific, axis = 1, inplace = True)
ctrl.drop(uplink_specific+ctf_specific+snd_specific, axis = 1, inplace = True)
ctf.drop(uplink_specific+ctrl_specific+snd_specific, axis = 1, inplace = True)
hp.drop(uplink_specific+ctf_specific+snd_specific+ctrl_specific, axis = 1, inplace = True)
snd.drop(uplink_specific+ctf_specific+ctrl_specific, axis = 1, inplace = True)


# ## Who is the best?
# In pop culture you may have heard of optic, faze, and many others, but from those clans... who's the top dog?
# A quick search to the internet would say that... is the best?
# But lets see if the data supports that!!

# ### Best Team Overall

# In[11]:


df.groupby('team').count().sort_values('win?',ascending = False)[['win?']].head()


# #### Best Team per game mode

# In[12]:


hp.groupby('team').count().sort_values('win?',ascending = False)[['win?']].head()


# In[45]:


snd.groupby('team').count().sort_values('win?',ascending = False)[['win?']].head()


# In[46]:


ctf.groupby('team').count().sort_values('win?',ascending = False)[['win?']].head()


# In[47]:


ctrl.groupby('team').count().sort_values('win?',ascending = False)[['win?']].head()


# In[48]:


up.groupby('team').count().sort_values('win?',ascending = False)[['win?']].head()


# ### Best Player Overall

# #### Best Player per game mode

# In[49]:


df.groupby('player').count().sort_values('win?',ascending = False)[['win?']].head()


# In[50]:


hp.groupby('player').count().sort_values('win?',ascending = False)[['win?']].head()


# In[51]:


snd.groupby('player').count().sort_values('win?',ascending = False)[['win?']].head()


# In[52]:


ctf.groupby('player').count().sort_values('win?',ascending = False)[['win?']].head()


# In[53]:


ctrl.groupby('player').count().sort_values('win?',ascending = False)[['win?']].head()


# In[54]:


up.groupby('player').count().sort_values('win?',ascending = False)[['win?']].head()


# In[55]:


df['suicides'].count()


# # What's a call of duty match without a post match summary and accolades?

# ## Accolades

# ### Most likely to kill themselves

# In[56]:


df.groupby('player').count().sort_values('suicides',ascending = False)[['suicides']].head()


# ### Worst KD

# In[57]:


df.groupby('player').mean().sort_values(['k/d'], ascending = True)[['k/d']].head()


# ## Start Small (Uplink)
# Match Type Description:
# The objective is to bring a satellite drone, which is placed in the middle of the map, to the enemy uplink station. Each team can score one point if the drone is thrown in or two points if the player enters the uplink station with the drone. Teams switch sides at half-time once the timer runs out or if one team scores 10 points. Teams must score 20 points to win the game.
# 
# Rounds vary a bit differently than other gametypes. The second round time limit is based on how long the first round lasted. For example, if one team scores all 10 points in the first round in two minutes, the second round will be two minutes instead of the usual five. This time limit is shown during the half-time under "Time to Beat."
# 
# While holding possession of the drone, the player will earn extra health and faster movement speed. They can throw the ball by pressing the fire button. The player can also pass the ball directly to another teammate using the aim button once they are locked on by seeing "pass" above the targeted player; the passing range is very large. Friendly players are visible as outlines through walls while carrying the ball. Players can strategically throw the drone to an enemy, rendering him weaponless and easy to take down.
# 
# A loose ball can bounce off surfaces, and potentially score a goal through rebounds. Balls that are thrown outside the map will be reset to its starting position. Throwing a ball away can prevent enemies from possessing it too close to one's own uplink station. If a ball lands in an inaccessible location, or is idle for several seconds, it will automatically reset.

# In[87]:


up.head()


# In[88]:


removelist = []
for column in up.columns:
    #print(sum(up[up[column].isna() == True][column])
    if sum(up[column].isna()) > int(up.shape[0]/2):
        removelist.append(column)
up.drop(removelist,axis = 1, inplace = True)


# ### Remove Match and Player logistics and Target data

# In[ ]:





# In[89]:


y = up['win?']
up.drop(['match id', 'series id', 'end time', 'mode', 'team', 'player'],axis = 1, inplace = True)


# In[96]:


up_cat.head()


# ### Split numeric and categorical data

# In[97]:


up_numeric = up._get_numeric_data()
up_cat = up.select_dtypes(include=['object']).copy()
up_numeric['avg kill dist'] = up_cat['avg kill dist (m)'].str.replace('m','').astype('float64')
up_numeric['accuracy'] = up_cat['accuracy (%)'].str.replace('%','').astype('float64')
up_cat.drop(['avg kill dist (m)', 'accuracy (%)', 'win?'], axis = 1, inplace = True)
X = pd.concat([up_numeric,pd.get_dummies(up_cat)], axis = 1)


# ### Test Train split

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# ### Fit Baseline Regression

# In[99]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
clf = LogisticRegression(solver = 'liblinear')
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores,scores.mean())
clf.fit(X_train,y_train)


# In[100]:


clf.score(X_test,y_test)


# In[101]:


for name, rank in zip(X.columns,clf.coef_[0]):
    print(name,rank)


# # Repeat process for Other game modes

# ## Capture The Flag
# The objective of Capture the Flag is to capture the opposing team's flag and bring it back to the player's team's flag. Points will only be added if an enemy has not got hold of the team's flag.

# In[457]:


removelist = []
for column in ctf.columns:
    if sum(ctf[column].isna()) > int(ctf.shape[0]/2):
        removelist.append(column)
ctf.drop(removelist,axis = 1, inplace = True)
y = ctf['win?']
ctf.drop(['match id', 'series id', 'end time', 'mode', 'team', 'player','win?','avg time per life (s)','map'],axis = 1, inplace = True)
ctf_numeric = ctf._get_numeric_data()
ctf_cat = ctf.select_dtypes(include=['object']).copy()
ctf_numeric['accuracy'] = ctf_cat['accuracy (%)'].str.replace('%','').astype('float64')
ctf_cat.drop(['accuracy (%)'], axis = 1, inplace = True)
X = pd.concat([ctf_numeric,pd.get_dummies(ctf_cat)], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = LogisticRegression(solver = 'liblinear')
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores,scores.mean())
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
for name, rank in zip(X.columns,clf.coef_[0]):
    print(name,rank)


# In[460]:


logreg = LogisticRegression()
import matplotlib.pyplot as plt
coefs = clf.coef_[0]

# Create a pandas dataframe to store the coefficients and their corresponding feature names
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': abs(coefs)})

# Sort the coefficients by their absolute values
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
pd.set_option('display.max_rows', 80)
feature_importances


# In[459]:


range_of_features = list(range(1,70,1))
from sklearn.metrics import accuracy_score

def accuracy_top_n_features(n):  # Gonna use the function from last class
       
        
        clf = LogisticRegression(solver = 'liblinear')
        clf.fit(X_train[list(feature_importances.feature[:n])],y_train)
        clf_predict = clf.predict(X_test[list(feature_importances.feature[:n])])
        return accuracy_score(y_test, clf_predict)
    
    
accuracy_scores = []
for n in range_of_features:
    current_accuracy = accuracy_top_n_features(n)
    accuracy_scores.append(current_accuracy)
accuracy_scores


# ## Search & Destroy
# A one-sided game mode, the goal is for an attacking side to either eliminate the defending team or detonate either one of two bomb sites. Players only get one life per round except for Call of Duty: Finest Hour, where players get unlimited lives, with most versions of the mode going to a best-of-seven rounds (first to four rounds wins). There is an intermission/half when two or three rounds are completed.

# In[274]:


removelist = []
for column in snd.columns:
    if sum(snd[column].isna()) > int(snd.shape[0]/2.5):
        removelist.append(column)
snd.drop(removelist,axis = 1, inplace = True)


# In[244]:


y = snd['win?']
snd.drop(['match id','score' 'series id', 'end time','win?', 'mode', 'team', 'player','avg time per life (s)','map'],axis = 1, inplace = True)
snd_numeric = snd._get_numeric_data()
snd_cat = snd.select_dtypes(include=['object']).copy()
snd_numeric['accuracy'] = snd_cat['accuracy (%)'].str.replace('%','').astype('float64')
snd_cat.drop(['accuracy (%)'], axis = 1, inplace = True)
X = pd.concat([snd_numeric,pd.get_dummies(snd_cat)], axis = 1)


# In[245]:


sizes = set()
for column in X.columns[X.isnull().any()]:
    sizes.add(X[column].isna().sum())
if len(sizes) == 1 and list(sizes)[0] < int(X.shape[0] /15):
    newset = pd.concat([X,y],axis =1)
    newset.dropna(inplace = True)
    X = newset.drop(['win?'], axis = 1)
    y = newset['win?']


# In[246]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = LogisticRegression(solver = 'liblinear')
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores,scores.mean())
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
#for name, rank in zip(X.columns,clf.coef_[0]):
print(name,rank)


# In[247]:


logreg = LogisticRegression()
import matplotlib.pyplot as plt
coefs = clf.coef_[0]

# Create a pandas dataframe to store the coefficients and their corresponding feature names
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': abs(coefs)})

# Sort the coefficients by their absolute values
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
pd.set_option('display.max_rows', 80)
feature_importances


# In[166]:


range_of_features = list(range(1,64,1))
from sklearn.metrics import accuracy_score

def accuracy_top_n_features(n):  # Gonna use the function from last class
       
        
        clf = LogisticRegression(solver = 'liblinear')
        clf.fit(X_train[list(feature_importances.feature[:n])],y_train)
        clf_predict = clf.predict(X_test[list(feature_importances.feature[:n])])
        return accuracy_score(y_test, clf_predict)
    
    
accuracy_scores = []
for n in range_of_features:
    current_accuracy = accuracy_top_n_features(n)
    accuracy_scores.append(current_accuracy)
accuracy_scores


# ## Hardpoint
# The teams must rush to secure a "hardpoint" on the map and repel the area from the enemies. Holding the hardpoint increases the teams' score, but if no players are in the hardpoint, no points are gained. The Hardpoint switches to different set rotational positions on the map several times during the match. A point is granted every second a single team is present in the Hardpoint.If players from both teams are present in the Hardpoint, it will be marked as "contested" and neither team will gain points until remaining enemies are removed from the objective. Players who arrive in a new Hardpoint or one that was previously occupied by enemies gains 200 score for "Securing the Hardpoint" and will receive a "Hardpoint Secure" medal.

# In[462]:


#hp


# In[470]:


removelist = []
for column in hp.columns:
    if sum(hp[column].isna()) > int(hp.shape[0]/2.5):
        removelist.append(column)
hp.drop(removelist,axis = 1, inplace = True)


# In[471]:


y = hp['win?']
hp.drop(['match id', 'series id', 'end time', 'mode','map','team', 'player','win?','avg time per life (s)'],axis = 1, inplace = True)
hp_numeric = hp._get_numeric_data()
hp_cat = hp.select_dtypes(include=['object']).copy()
hp_numeric['accuracy'] = hp_cat['accuracy (%)'].str.replace('%','').astype('float64')
hp_cat.drop(['accuracy (%)'], axis = 1, inplace = True)
X = pd.concat([hp_numeric,pd.get_dummies(hp_cat)], axis = 1)


# In[472]:


sizes = set()
for column in X.columns[X.isnull().any()]:
    sizes.add(X[column].isna().sum())
if len(sizes) == 1 and list(sizes)[0] < int(X.shape[0] /15):
    newset = pd.concat([X,y],axis =1)
    newset.dropna(inplace = True)
    X = newset.drop(['win?'], axis = 1)
    y = newset['win?']


# In[473]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = LogisticRegression(solver = 'liblinear')
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores,scores.mean())
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
for name, rank in zip(X.columns,clf.coef_[0]):
    print(name,rank)


# In[474]:


logreg = LogisticRegression()
import matplotlib.pyplot as plt
coefs = clf.coef_[0]

# Create a pandas dataframe to store the coefficients and their corresponding feature names
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': abs(coefs)})

# Sort the coefficients by their absolute values
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
pd.set_option('display.max_rows', 80)
feature_importances


# In[475]:


range_of_features = list(range(1,64,1))
from sklearn.metrics import accuracy_score

def accuracy_top_n_features(n):  # Gonna use the function from last class
       
        
        clf = LogisticRegression(solver = 'liblinear')
        clf.fit(X_train[list(feature_importances.feature[:n])],y_train)
        clf_predict = clf.predict(X_test[list(feature_importances.feature[:n])])
        return accuracy_score(y_test, clf_predict)
    
    
accuracy_scores = []
for n in range_of_features:
    current_accuracy = accuracy_top_n_features(n)
    accuracy_scores.append(current_accuracy)
accuracy_scores

