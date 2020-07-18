---
layout: post
title:      "SMOTE: Handling Imbalanced Classes in Machine Learning Classification"
date:       2020-07-18 23:03:06 +0000
permalink:  smote_handling_imbalanced_classes_in_machine_learning_classification
---


In this post I’ll be discussing one method for dealing with a common problem in machine learning classification: imbalanced datasets.

First, let’s consider why this situation is a problem. An imbalanced dataset, with respect to classification, is one in which there are many more members of one target class than another. This can happen in either binary or multi-class classification problems. Since supervised machine learning algorithms ‘learn’ from labeled training data, it can be an issue when the algorithm trains on a disproportionate number of one class of labels. 

Today I’ll be demonstrating this concept using the Haberman’s Survival Dataset available on Kaggle (https://www.kaggle.com/gilsousa/habermans-survival-data-set) to illustrate these points. This dataset contains information for patients who had undergone surgery for breast cancer between 1958 and 1970, and contains 308 observations with four attributes: (1) age of patient at time of operation, (2) year of the operation, (3) the number of positive axillary nodes detected, and (4) whether the patient died within five years. The Haberman’s dataset is an example of class imbalance - of the 306 patients, 226 (74%) survived at least five years, while only 81 (26%) died within 5 years.

Clearly, what we would like to try to predict with this dataset is whether a patient died within 5 years (i.e., the target variable), based on the information in the remaining three attributes. This exercise has clear practical implications, as it could identify future patients who are immediately at risk.  The code below walks through a basic machine learning classification algorithm for this problem.

```
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, plot_confusion_matrix
from imblearn.over_sampling import SMOTE

# Import haberman dataset from local file
df = pd.read_csv('haberman.csv')
```


For a baseline model, I will perform a 75%-25% train-test split on the data, and fit a Logistic Regression classifier to make predictions on the test data.

```
# Define X and y variables
X = df.drop('target', axis=1)
y = df.target

# Perform train test split on data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Fit a logistic regression model and make predictions
clf_1 = LogisticRegression()
clf_1.fit(X_train, y_train)
y_hat_train = clf_1.predict(X_train)
y_hat_test = clf_1.predict(X_test)
```

To assess how well this classifier performs, I will look at two metrics: the recall score, and a confusion matrix. I opt for the recall score, in lieu of accuracy or precision, due to the nature of the problem itself: since this classification algorithm is trying to predict whether someone will die in the next 5 years, I think that it is more important to minimize false negative results (overlooking a patient that may die within five years) than it is to achieve overall accuracy. In other cases, it might be more important to ensure that precision is maximized, which would incentivize minimizing the number of false positive results (e.g., an algorithm that automatically removes email labeled as spam from an inbox). The recall score can be calculated using the following equation:

![](https://help.relativity.com/9.5/Content/Resources/Images/Recipes/How_to_Calculate_Precision_and_Recall_without_a_Control_Set/SCR_RecallEquation.png)

In addition, a confusion matrix will tell us exactly where the algorithm is performing well, and where it comes up short.

```
# Check train and test recall score

train_recall = round(recall_score(y_train, y_hat_train, pos_label=2), 2)
test_recall = round(recall_score(y_test, y_hat_test,pos_label=2), 2)

print(f'Train Recall Score: {train_recall}')
print(f'Test Recall Score: {test_recall}')
```

```
Train Recall Score: 0.15
Test Recall Score: 0.21
```

So, in this baseline model, the train and test recall are quite low (15% and 21%, respectively). This means that of all the patients that died within 5 years, the classifier could only correctly predict 21% of these cases. Let’s see what the confusion matrix shows:

```
plot_confusion_matrix(clf_1, X_test, y_test, cmap='Blues',
                      display_labels = ['Live > 5 Years', 'Live < 5 Years'],
                     normalize='true');
```

![](https://raw.githubusercontent.com/douglaspsteen/smote/master/cm_1.png?token=AH6UCI6CHA3U3YJZC7IZEIC7CN6X2)

While the classifier performs very well when predicting cases where the patient lived greater than five years, it predicts very poorly with patients that lived less than five years. This is likely because, as the classifier was fit to the training dataset, it saw many more instances of patients who lived more than five years, and very few instances of patients who lived less than five years. The result is a classifier that is good at predicting the most common result, and very bad at predicting the most dire circumstance: when a person has less than five years to live.

How can we address this problem and improve our classifier? There are many methods for addressing class imbalance in supervised learning, but here we will explore just one: Synthetic Minority Oversampling TEchnique, or SMOTE.

##### SMOTE

When a minority class is present in a dataset, one method to deal with this is by oversampling that minority class to produce a balanced dataset. SMOTE allows for this by creating new ‘synthetic’ observations of the minority class. SMOTE generally works like this: 

1) A feature vector of the minority class and its nearest neighbor is identified;
2) The difference between the vector and neighbor is calculated;
3) That difference is multiplied by a random number between 0 and 1;
4) A ‘synthetic’ point is created by adding the random number to the feature vector along the line segment connecting to the nearest neighbor;
5) Steps 1 - 4 repeated for other feature vectors until the desired number of ‘synthetic’ points are added.
6) 
(The steps described above are explained in detail in this video on SMOTE: [SMOTE - Synthetic Minority Oversampling Technique - Jitesh Khurkhuriya](https://www.youtube.com/watch?v=FheTDyCwRdE&t=1s))

Here’s how this looks in our Haberman’s dataset example in Python. The process is the same as our baseline classifier, but with an added step in which SMOTE is performed on the training data. 

NOTE: SMOTE should be performed on your training data only - since our test data is what we’ve set aside for ‘ground-truthing’ our model, we don’t want to introduce any ‘synthetic’ data into it. 

```
# Define X and y variables
X = df.drop('target', axis=1)
y = df.target

# Perform train test split on data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,  random_state=1)

# Perform SMOTE to balance classes
smote = SMOTE(random_state=1)
smote.fit(X_train, y_train)
X_train, y_train = smote.sample(X_train, y_train)

# Fit a logistic regression model and make predictions
clf_2 = LogisticRegression()
clf_2.fit(X_train, y_train)
y_hat_train = clf_2.predict(X_train)
y_hat_test = clf_2.predict(X_test)

# Check train and test recall score
train_recall = round(recall_score(y_train, y_hat_train, pos_label=2), 2)
test_recall = round(recall_score(y_test, y_hat_test, pos_label=2), 2)
print(f'Train Recall Score: {train_recall}')
print(f'Test Recall Score: {test_recall}')
```

```
Train Recall Score: 0.46
Test Recall Score: 0.68
```

After performing SMOTE, our test recall score has improved dramatically. Remember, this represents the percentage of positive cases (patient died within 5 years) that were correctly identified by the model. This is the same Logistic Regression classifier we used previously - the only thing that changed was balancing the training data classes with SMOTE.

The confusion matrix also shows how SMOTE has paid off:

```
plot_confusion_matrix(clf_2, X_test, y_test, cmap='Blues',
                      display_labels = ['Live > 5 Years', 'Live < 5 Years'],
                     normalize='true');
```

![](https://raw.githubusercontent.com/douglaspsteen/smote/master/cm_2.png?token=AH6UCI2FO7KLIGAZ5ODQDE27CN7NO)


This model correctly classified more than two-thirds of patients that had less than 5 years to live, and is a dramatic improvement over the first model in this regard. However, you can see that the classifier is slightly worse at correctly predicting when a person will live more than five years. In this situation, it seems worthwhile to risk increasing the rate of false positives (top right on confusion matrix) to decrease the rate of false negatives (bottom left). This will not always be the case - the metrics used to evaluate a classifier will always depend on the specific domain and problem to be solved. For example, you wouldn’t want an email spam filter so aggressive that it files away important messages from your boss - those would be some costly false positives!

While there are a variety of ways to deal with class imbalance in machine learning, I hope I’ve shown that SMOTE is a great option to have in the toolbox.

References:

https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

https://www.kaggle.com/gilsousa/habermans-survival-data-set

[SMOTE - Synthetic Minority Oversampling Technique - Jitesh Khurkhuriya](https://www.youtube.com/watch?v=FheTDyCwRdE&t=1s)

