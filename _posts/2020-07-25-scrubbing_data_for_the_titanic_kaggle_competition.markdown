---
layout: post
title:      "Scrubbing Data for the Titanic Kaggle Competition"
date:       2020-07-25 22:46:39 +0000
permalink:  scrubbing_data_for_the_titanic_kaggle_competition
---


To keep my data science skills sharp, I decided to try my hand at one of the most popular machine learning competitions on [Kaggle: Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic). The competition challenges machine learning practitioners to develop a classification algorithm to predict whether a passenger on the Titanic survived or perished, with the following information available for each individual: ID number, Class (1, 2, or 3), Name, Sex, Age, number of Siblings/Spouses on board, number of Parents/Children on board, Ticket Number, Ticket Fare, Cabin, and Port of Embarkation. Kaggle splits the dataset into a training set with 891 labeled observations and a test set with 418 unlabeled observations. The test set labels are unknown to the competitor - only after submitting test predictions is your prediction accuracy score revealed.

This post will focus on my process for loading and scrubbing the data in Python. I will address exploring, modeling, and interpretation of the results in future posts. Disclaimer: I will probably wish to refine the data scrubbing process outlined here as I come across new insights in later steps. Ok, let’s jump in!

For these initial steps, I import numpy, pandas, and matplotlib.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

I can then import the test and training data from my current directory using pandas. ‘train.csv’ and ‘test.csv’ can be downloaded from the [Kaggle competition data page](https://www.kaggle.com/c/titanic/data).

```
# Import training/testing data using pandas

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

It makes most sense to process the train and test data at once, so I combine them into one dataframe while ignoring the target variable (‘Survived’).

```
train_features = train.drop('Survived', axis=1)
df = pd.concat([train_features, test], axis=0, ignore_index=True)
```

Now let’s preview ‘df’ to see what the raw data look like.

```
df.head()
df.info()
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_1.png)

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 11 columns):
PassengerId    1309 non-null int64
Pclass         1309 non-null int64
Name           1309 non-null object
Sex            1309 non-null object
Age            1046 non-null float64
SibSp          1309 non-null int64
Parch          1309 non-null int64
Ticket         1309 non-null object
Fare           1308 non-null float64
Cabin          295 non-null object
Embarked       1307 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 112.6+ KB
```

Since our goal is to use these features for machine learning classification, there are several immediate issues with these data.

* Missing (null) values in the 'Age', 'Cabin', 'Embarked', and ‘Fare’  features. 
* Non-numeric data is present in the 'Name', 'Sex', 'Ticket', 'Cabin', and 'Embarked' features.
* Since ‘Pclass’ is a qualitative description, it is probably more appropriate to one-hot encode as a categorical variable.

###### Handling missing ‘Age’ values

There are so many missing age values, that I don't want to simply fill these with the overall median or mean. So, I will impute age values using median values when grouping by a couple of other categories: ‘Pclass’ and ‘Sex’. First, let's find these median values:

```
grouped_df = df.groupby(['Sex', 'Pclass'])
grouped_median_df = grouped_df.median()
grouped_median_df = grouped_median_df.reset_index()[['Sex', 'Pclass', 'Age']]

grouped_median_df
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_2.png)

Now I can define a function to impute ‘Age’ where they are missing using each instance’s ‘Sex’ and ‘Pclass’:

```
def impute_ages(df):
    
    df.Age.fillna(value=0, inplace=True)
    
    for i in range(len(df)):
        if df.iloc[i]['Age'] == 0:
            if df.iloc[i]['Sex'] == 'female' and df.iloc[i]['Pclass'] == 1:
                df.set_value(i, 'Age', grouped_median_df.iloc[0]['Age'])
                
            elif df.iloc[i]['Sex'] == 'female' and df.iloc[i]['Pclass'] == 2:
                df.set_value(i, 'Age', grouped_median_df.iloc[1]['Age'])
                
            elif df.iloc[i]['Sex'] == 'female' and df.iloc[i]['Pclass'] == 3:
                df.set_value(i, 'Age', grouped_median_df.iloc[2]['Age'])
                
            elif df.iloc[i]['Sex'] == 'male' and df.iloc[i]['Pclass'] == 1:
                df.set_value(i, 'Age', grouped_median_df.iloc[3]['Age'])
                
            elif df.iloc[i]['Sex'] == 'male' and df.iloc[i]['Pclass'] == 2:
                df.set_value(i, 'Age', grouped_median_df.iloc[4]['Age'])
                
            elif df.iloc[i]['Sex'] == 'male' and df.iloc[i]['Pclass'] == 3:
                df.set_value(i, 'Age', grouped_median_df.iloc[5]['Age'])

impute_ages(df)
```

###### Handling missing ‘Cabin’ values

Since most of the data is missing, I am going to opt to drop this feature entirely.

```
df.drop('Cabin', axis=1, inplace=True)
```

###### Handling ‘Embarked’ missing values and data type

Since there are only two missing values, I will fill these with the most commonly occurring value for ‘Embarked’.

```
# Let's see the distribution of values for 'Embarked'

df.Embarked.value_counts()
```

```
S    914
C    270
Q    123
Name: Embarked, dtype: int64
```

```
# Since most people embarked at 'S', we will fill the two null values with 'S'

df.Embarked.fillna('S', inplace=True)

df.Embarked.value_counts()
```

Since 'Embarked' is stored categorically, I also want to convert this feature to one-hot encoded dummy variables.

```
dummies = pd.get_dummies(df.Embarked, prefix='Emb')
df = pd.concat([df, dummies], axis=1)
df.drop('Embarked', axis=1, inplace=True)
```

###### Handling ‘Sex’ data type

As with ‘Embarked’, ‘Sex’ is stored categorically and should be converted to dummy variables. Since there are only two options for ‘Sex’, I will drop one of the dummy variables, as no information will be added by keeping them both.

```
s_dummies = pd.get_dummies(df.Sex, prefix='sex', drop_first=True)
df = pd.concat([df, s_dummies], axis=1)
df.drop('Sex', axis=1, inplace=True)
```

###### Handling ‘Name’ and ‘Ticket’ features

For the sake of simplicity, I will drop these features from the dataset. There may be some way to extract meaningful information from these features, and I will probably come back to this later after some initial modeling.

```
df.drop(['Name', 'Ticket'], axis=1, inplace=True)
```

###### Handling ‘Fare’ missing value

There is only one missing value for ‘Fare’, and we can make a pretty good guess based on ‘Pclass’. So, let’s take a look at the row where ‘Fare’ is missing.

```
df[df.isnull().any(axis=1)]
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_3.png)

This passenger is in the 3rd class. Now I need to find the median ‘Fare’ value for 3rd class and use it to fill the missing value.

```
grouped_df = df.groupby(['Pclass'])
grouped_median_df = grouped_df.median()
grouped_median_df = grouped_median_df.reset_index()[['Pclass', 'Fare']]
grouped_median_df
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_4.png)

```
df.Fare.fillna(8.05, inplace=True)
```

###### One-hot encoding of ‘Pclass’

Since ‘Pclass’ isn’t really a quantitative measure, I would prefer to one-hot encode this feature into dummy variables.

```
c_dummies = pd.get_dummies(df.Pclass, prefix='class', drop_first=False)
df = pd.concat([df, c_dummies], axis=1)
df.drop('Pclass', axis=1, inplace=True)
```

Ok, let’s see a preview of the ‘scrubbed’ dataframe and make sure there are no issues with data types or missing values.

```
df.head()
df.info()
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_5.png)

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 12 columns):
PassengerId    1309 non-null int64
Age            1309 non-null float64
SibSp          1309 non-null int64
Parch          1309 non-null int64
Fare           1309 non-null float64
Emb_C          1309 non-null uint8
Emb_Q          1309 non-null uint8
Emb_S          1309 non-null uint8
sex_male       1309 non-null uint8
class_1        1309 non-null uint8
class_2        1309 non-null uint8
class_3        1309 non-null uint8
dtypes: float64(2), int64(3), uint8(7)
memory usage: 60.2 KB
```

Perfect. None of the features have missing values, and all data types are either float or integers. Qualitative variables (Port of Embarkation, Sex, and PClass) have successfully been converted to dummy variables. Since I plan to use tree-based algorithms for modeling, I will refrain from doing any feature scaling here. 

My next steps will be to explore and visualize the data distributions and perform some feature engineering to assist the classification algorithms. Tune in to the next post to check it out!


