---
layout: post
title:      "EDA and Feature Engineering on the Titanic Kaggle Dataset"
date:       2020-08-01 19:29:28 -0400
permalink:  eda_and_feature_engineering_on_the_titanic_kaggle_dataset
---


Last week I wrote about my initial data cleaning process for the Titanic Kaggle competition (see that post [here](https://douglaspsteen.github.io/scrubbing_data_for_the_titanic_kaggle_competition)). This week, my focus is on exploratory data analysis (EDA) and some feature engineering using the Titanic dataset. 

To begin my EDA, I’d like to visualize how some of the individual features relate to the target variable, ‘Survived’. Below, concatenate the scrubbed training data features with the training data labels (‘Survived’) and add a complementary column (‘Died’) to assist with plotting.

```
# Imports
import numpy as np
import pandas as pd
port matplotlib.pyplot as plt
```

```
train_data = pd.concat([df[:len(train)], train.Survived], axis=1)
train_data['Died'] = 1 - train['Survived']
```

### Survival by Sex

Many people recall the adage “Women and children first!” when imagining the evacuation of the Titanic as it was sinking in 1912. We can then assume that, since women would have been given preference on the lifeboats, that women were more likely to survive the tragedy than men. Let’s use pandas .plot() method to visualize this and see if it’s true.

```
ax = train_data.groupby('male').agg('sum')[['Survived', 'Died']].plot(
    kind='bar', figsize=(8,6), stacked=True, color=['g', 'r'])
ax.set_xlabel('Sex')
ax.set_ylabel('Count')
ax.set_xticklabels(['Female', 'Male']);
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_EDA_sex.png)

Just as we might’ve guessed, being a man on the Titanic meant your chances of survival were much lower - most men aboard died in the disaster. However, most female passengers actually survived. Sex will likely be one of the most important predictors in the final classification algorithm.

### Survival by Passenger Class

Sadly, one might guess that wealthier passengers had a better chance of escaping the Titanic safely. 

```
ax = train_data.groupby('Pclass').agg('sum')[['Survived', 'Died']].plot(
    kind='bar', figsize=(8,6), stacked=True, color=['g', 'r'])
ax.set_xlabel('Pclass')
ax.set_ylabel('Count')
ax.set_xticklabels(['1', '2', '3']);
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_EDA_pclass.png)

It appears that, although there were more than twice as many passengers in 3rd class than 1st class, more passengers from 1st class survived. This wealth inequality, for whatever reason, clearly had a role in determining which passengers survived the sinking.

### Survival by Family Size

In a transformation not shown here, I added the ‘SibSp’ (# Siblings or spouses) and ‘Parch’ (# Parents and children) features to create a new feature called ‘family’ (family size aboard) for each passenger. As you can see below, the relationship between family size and survival is not as straightforward as in sex or passenger class.

```
ax = train_data.groupby('family').agg('sum')[['Survived', 'Died']].plot(
    kind='bar', figsize=(8,6), stacked=True, color=['g', 'r'])
ax.set_xlabel('# Family Members')
ax.set_ylabel('Count');
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_EDA_family.png)

It appears that passengers alone on the Titanic died at a higher rate than those with one, two, or three family members on board. However, in family sizes of four and above, the survival rate is generally quite low. Perhaps being in a small, close family was an advantage, as those family members could look after each other. On the other hand, being alone or in a very large family may have meant getting lost in the crowd and not finding a lifeboat. Additionally, many of those traveling alone were likely to be men, and would already be at a higher risk of death.

### Survival by Port of Embarkation

The Titanic picked up passengers at three European ports before heading across the Atlantic: Southampton, England (S), Cherbourg, France (C), and Queenstown, Ireland (Q). Did a passenger’s port of embarkation have any impact on passenger survival?

```
ax = train_data.groupby('Embarked').agg('sum')[['Survived', 'Died']].plot(
    kind='bar', figsize=(8,6), stacked=True, color=['g', 'r'])
ax.set_xlabel('Port of Embarkation')
ax.set_ylabel('Count');
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_EDA_embark.png)

The survival rate of passengers who got on at Cherbourg seems to be slightly higher than those from Queenstown or Southampton, but not by much.

### Survival by Age

If the women and children truly were saved first, then survival rate should be higher among the youngest passengers.

```
ax = train_data.pivot(columns='Survived', values='Age').plot.hist(bins=15, 
                                                                  alpha=0.5,
                                                                 color=['r', 'g'],
                                                                 figsize=(8,6))
ax.set_xlabel('Age');
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_EDA_age.png)

This seems to hold true - passengers about 0 - 5 years old have a much higher survival rate, and even teenagers appear to have had better odds than the rest of the ship’s population.

### Survival by Fare

Similar to passenger class, I expect that the higher paying passengers would somehow have a better chance for survival than lower paying passengers. The wealthier passengers were likely more favorably located in the ship, and probably had more means to bribe their way into lifeboat spots.

```
ax = train_data.pivot(columns='Survived', values='Fare').plot.hist(bins=50, 
                                                                  alpha=0.5,
                                                                 color=['r', 'g'],
                                                                 figsize=(8,6))
ax.set_xlabel('Fare');
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_EDA_fare.png)

As expected, who paid the lowest fares had a much lower survival rate. Interestingly, at fares of 50 and above, this wealthy minority seemed to survive more often than not.

### Feature Engineering: Extracting title from ‘Name’

Since each passenger name is unique and non-numeric, it is difficult to gain much information from this feature without some engineering. With a little research, we can see that each passenger name includes a title. The typical titles that appear are ‘Mr.’, ‘Mrs.’, ‘Miss.’, ‘Master.’, and ‘Rev.’. Below is a function to extract the titles from each passenger name, and store them in a new feature called ‘title’.

```
def title_from_name(df):
    df['title'] = None
    
    for i in range(len(df)):
        
        if 'Mr.' in df.iloc[i]['Name']:
            df.set_value(i, 'title', 'Mr')
        elif 'Mrs.' in df.iloc[i]['Name']:
            df.set_value(i, 'title', 'Mrs')
        elif 'Miss' in df.iloc[i]['Name']:
            df.set_value(i, 'title', 'Miss')
        elif 'Master' in df.iloc[i]['Name']:
            df.set_value(i, 'title', 'Master')
        elif 'Rev.' in df.iloc[i]['Name']:
            df.set_value(i, 'title', 'Rev')
        else:
            df.set_value(i, 'title', 'Other')
            
title_from_name(df)

ax = train_data.groupby('title').agg('sum')[['Survived', 'Died']].plot(
    kind='bar', figsize=(8,6), stacked=True, color=['g', 'r'])
ax.set_xlabel('Title')
ax.set_ylabel('Count');
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_EDA_title.png)

Passengers with the ‘Mr.’ title were very likely to die, while those with ‘Miss.’ or ‘Mrs.’ had much higher survival rates. The title ‘Master’ was used at the time to identify male minors, which explains why the survival rate of this small group was higher than in the ‘Mr.’ group.

### Baseline Classification Using Sex Only

Based on the EDA performed above, I think that using ‘Sex’ (male or female) as a single predictor of survival on the Titanic would actually produce some pretty good results. We can use the labeled training dataset of 891 instances provided by Kaggle to test this theory. The code for the ‘Gender Predictor’ below simply predicts that if a passenger is male, they would have died in the wreck.

```
from sklearn.metrics import accuracy_score

X = df[:len(train)]
y = train.Survived

gender_pred = 1 - X.male
gender_acc = round(accuracy_score(y, gender_pred), 2)
print(f'Gender Predictor Accuracy: {gender_acc}')
```

```
Gender Predictor Accuracy: 0.79
```

On the training dataset, predicting that all males died leads to 79% accuracy - not bad! 

But, how does the Gender Predictor do when applied to the unlabeled test data? Kaggle provides a gender_submission.csv data file that allows us to check this.

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/img_EDA_submission.png)

The Gender Predictor achieves 77% accuracy on the test data set. While this is a very simple classifier using only one Titanic dataset feature, it will serve as a great baseline to indicate whether more complex models provide an actual improvement. As many Kaggle Titanic participants will attest, it is surprisingly challenging to improve significantly on this score.

That’s what I’ll attempt to do in my next post, when I try out several machine learning classification algorithms!


