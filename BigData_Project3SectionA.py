
# Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

# Reading and exploring the dataset
titanic_df=pd.read_csv(r"Titanic.csv")
test_df=pd.read_csv(r"test.csv")
titanic_df.head
titanic_df.describe
titanic_df.info()
titanic_df.columns.values

# Q1: Find out the overall chance of survival for a Titanic passenger.
print("Total number of passengers survived are",titanic_df['survived'].value_counts()[1])
print("Percentage passengers survived are",titanic_df['survived'].value_counts(normalize=True)[1]*100)


# Q2: Find out the chance of survival for a Titanic passenger based on their sex and plot it.
sns.barplot(x="sex", y="survived", data=titanic_df)
print("Percentage of females who survived is", titanic_df["survived"][titanic_df["sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived is", titanic_df["survived"][titanic_df["sex"] == 'male'].value_counts(normalize = True)[1]*100)


# Q3: Find out the chance of survival for a Titanic passenger by traveling class wise and plot it.

sns.barplot(x="pclass", y="survived", data=titanic_df)
print("Percentage of Pclass 1 who survived is", titanic_df["survived"][titanic_df["pclass"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass 2 who survived is", titanic_df["survived"][titanic_df["pclass"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass 3 who survived:", titanic_df["survived"][titanic_df["pclass"] == 3].value_counts(normalize = True)[1]*100)


# Q4: Find out the average age for a Titanic passenger who survived by passenger class and sex. 

fig = plt.figure(figsize=(12,5))
fig.add_subplot(121)
plt.title('TRAIN - Age/Sex per Passenger Class')
sns.barplot(data=titanic_df, x='pclass',y='age',hue='sex')
meanAgeTrnMale = round(titanic_df[(titanic_df['sex'] == "male")]['age'].groupby(titanic_df['pclass']).mean(),2)
meanAgeTrnFeMale = round(titanic_df[(titanic_df['sex'] == "female")]['age'].groupby(titanic_df['pclass']).mean(),2)
print('\n\t\tMEAN AGE PER SEX PER PCLASS')
print(pd.concat([meanAgeTrnMale, meanAgeTrnFeMale], axis = 1,keys= ['Male','Female']))


# Q5: Find out the chance of survival for a Titanic passenger based on number of siblings the passenger had on the ship and plot it.

sns.barplot(x="sibsp", y="survived", data=titanic_df)
print("Percentage of SibSp 0 who survived is", titanic_df["survived"][titanic_df["sibsp"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp 1 who survived is", titanic_df["survived"][titanic_df["sibsp"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp 2 who survived is", titanic_df["survived"][titanic_df["sibsp"] == 2].value_counts(normalize = True)[1]*100)

# Q6: Find out the chance of survival for a Titanic passenger based on number of parents/children the passenger had on the ship and plot it.

sns.barplot(x="parch", y="survived", data=titanic_df)
plt.show()
print("Percentage of parch 0 who survived is", titanic_df["survived"][titanic_df["parch"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of parch 1 who survived is", titanic_df["survived"][titanic_df["parch"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of parch 2 who survived is", titanic_df["survived"][titanic_df["parch"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of parch 3 who survived is", titanic_df["survived"][titanic_df["parch"] == 3].value_counts(normalize = True)[1]*100)


# Q7: Plot out the variation of survival and death amongst passengers of different age.

titanic_df["age"] = titanic_df["age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
titanic_df['agegroup'] = pd.cut(titanic_df['age'], bins, labels = labels)
sns.barplot(x="agegroup", y="survived", data=titanic_df)
plt.show()
g = sns.FacetGrid(titanic_df, col='survived')
g.map(plt.hist, 'age', bins=20)


# Q8: Plot out the variation of survival and death with age amongst passengers of different passenger classes.

grid = sns.FacetGrid(titanic_df, col='survived', row='pclass', size=3, aspect=2)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# Q9: Find out the survival probability for a Titanic passenger based on title from the name of passenger.
combine = [titanic_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic_df['Title'],titanic_df['sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

titanic_df[['Title', 'survived']].groupby(['Title'], as_index=False).mean()
