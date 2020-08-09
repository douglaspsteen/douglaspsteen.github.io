---
layout: post
title:      "Applying the KNN Algorithm to the Titanic Kaggle Dataset"
date:       2020-08-09 03:35:26 +0000
permalink:  applying_the_knn_algorithm_to_the_titanic_kaggle_dataset
---


In this post I’m going to be exploring one of my favorite machine learning classification algorithms - k-nearest neighbors, or k-NN. k-NN has the advantage of being quite intuitive to understand, and it performs exceptionally well on certain problems. k-NN can be used for either regression or classification problems - here I will focus on classification only.

As in my recent posts, here I am working with the Titanic Kaggle Dataset. This famous dataset contains information on the Titanic’s passengers (Name, Fare, Sex, etc.), as well as which passengers survived the disaster. Today I will be applying the k-NN algorithm to determine how accurately a passenger’s survival can be predicted from the other information provided in the dataset. 

k-NN is a non-parametric algorithm, meaning that it does not require any assumptions about the underlying data distributions. When used for classification, an unlabeled query point is classified based on the k labeled neighbor points that are in closest proximity to that unlabeled point. Proximity is determined by calculating distance in n-dimensional feature space using one of several metrics.

## Selecting features for k-NN

As k-NN suffers from the curse of dimensionality, it would be unwise to simply use all of the features in the model. For the purpose of this demonstration, I arbitrarily select four features that I think might be helpful in predicting survival: Sex (‘male’), passenger class (‘Pclass’), fare (‘Fare’), and whether a passenger has the title ‘Master’ in their name (‘title_Master’). Boys and young men aboard the Titanic in 1912 would have held the title ‘Master’, as was the convention of the time. Therefore this feature could help the model differentiate between boys and adult men - two groups with very different survival probabilities. Data pre-processing for these features is not shown in this post, but can be viewed in my previous posts here: [Scrubbing Data for the Titanic Kaggle Competition](https://douglaspsteen.github.io/scrubbing_data_for_the_titanic_kaggle_competition), [EDA and Feature Engineering on the Titanic Kaggle Dataset](https://douglaspsteen.github.io/eda_and_feature_engineering_on_the_titanic_kaggle_dataset).

Another common approach for performing k-NN on a high-dimensionality dataset might be to perform dimensionality reduction (e.g., PCA) - I don’t explore this technique here, but it is certainly a possibility, and one that I may explore in the future.

## Fitting a k-NN Model

First I fit a vanilla k-NN classifier on the training dataset. I use sklearn’s `KNeighborsClassifier` instead of coding this from scratch. I also employ `train_test_split` from sklearn.model_selection, `accuracy_score` from sklearn.metrics, and `StandardScaler` from sklearn.preprocessing. I also import `GridSearchCV`, which I will use later for hyperparameter tuning.

It is important to scale features AFTER performing the `train_test_split`, in order to avoid data leakage from the test set to the training set.

```
# Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



# Subset data for k-NN classifier
X = X[['male', 'Pclass', 'Fare', 'title_Master']]
y = train.Survived

# Check shapes of features and target
X.shape, y.shape

((891, 4), (891,))


# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                       random_state=1)

# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
    
# Fit k-NN classifier and make predictions
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# Print results
print(f'Train Accuracy: {train_acc}')
print(f'Test Accuracy: {test_acc}')
```

*Train Accuracy: 0.8712574850299402
Test Accuracy: 0.8340807174887892*

So, the vanilla k-NN classifier actually performs pretty well for this task, with an 83% classification accuracy on the test set. Also, 87% accuracy on the training set indicates that this classifier is probably not overfitting by very much. Now, let’s see if we can improve this performance by adjusting the parameters of `KNeighborsClassifier`.

## Tuning the n_neighbors parameter

One of the adjustable parameters in sklearn’s `KNeighborsClassifier` is `n_neighbors` (default = 5). This parameter represents the number of nearby labeled data points the classifier uses to classify a new instance. Each of these ‘neighbors’ can be thought of as having a ‘vote’ toward the classification of the new instance. Here, I test how the classifier performance changes for n_neighbors values of 1 through 200.

```
train_acc = []
test_acc = []

for i in range(1, 200):
    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=1)

    # Perform feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit k-NN classifier and make predictions
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_acc.append(accuracy_score(y_train, y_pred_train))
    test_acc.append(accuracy_score(y_test, y_pred_test))

# Plot accuracy by n_neighbors
plt.figure(figsize=(8, 6))
plt.plot(range(1, 200), train_acc, label='Train Accuracy')
plt.plot(range(1, 200), test_acc, label='Test Accuracy')
plt.ylabel('Accuracy Score')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/n_neighbors.png)

The k-NN classifier seems to overfit the data for n_neighbors values below 50, where the train and test accuracy appear to converge. When n_neighbors exceeds 120, additional increases in `n_neighbors` do not appear to have any significant effect on classification accuracy.

## Tuning the weights parameter

Another adjustable parameter in KNeighborsClassifier is `weights`. Two options for the `weights` parameter are ‘uniform’ and ‘distance’. 

‘uniform’ gives each of the neighboring points equal ‘voting power’ toward classifying a new point. For example, if n_neighbors=50, the 50th closest data point would have just as much influence as the very closest data point in making a classification.

When using ‘distance’, the neighbors are weighted by the inverse of their distance from the point to be classified. So, closer neighbors to a query point will have more influence than neighbors that are farther away.

```
train_acc = []
test_acc = []

# Weights parameter options
weights = ['uniform', 'distance']

for weight in weights:
    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                       random_state=1)
    
    # Perform feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit k-NN classifier and make predictions
    knn = KNeighborsClassifier(weights=weight)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_acc.append(accuracy_score(y_train, y_pred_train))
    test_acc.append(accuracy_score(y_test, y_pred_test))

# Print results
print(f'Uniform-Weighted Train Acc: {train_acc[0]}')
print(f'Uniform-Weighted Test Acc: {test_acc[0]}')
print('\n')
print(f'Distance-Weighted Train Acc: {train_acc[1]}')
print(f'Distance-Weighted Test Acc: {test_acc[1]}')
```

*Uniform-Weighted Train Acc: 0.8712574850299402
Uniform-Weighted Test Acc: 0.8340807174887892*

*Distance-Weighted Train Acc: 0.9161676646706587
Distance-Weighted Test Acc: 0.8565022421524664*

Using distance-based weighting improves the test accuracy score by over 2%!

## Tuning the Minkowski metric (p)

The last parameter I’ll look at is `p`, the Minkowski metric. This parameter handles how the distance between points is calculated in the k-NN algorithm. When p=1 and p=2, the manhattan and euclidean distance calculations are used, respectively. The minkowski distance is used for an arbitrary value of `p`, and is a generalization of the manhattan and euclidean distances in normed vector space. 

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/distance_functions.png)
Source: [https://www.saedsayad.com/k_nearest_neighbors_reg.htm](https://www.saedsayad.com/k_nearest_neighbors_reg.htm)

```
train_acc = []
test_acc = []

# Minkowski power parameter options
p_metrics = [1,2,3,4,5]

for p in p_metrics:
    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                       random_state=1)
    
    # Perform feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit k-NN classifier and make predictions
    knn = KNeighborsClassifier(p=p)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_acc.append(accuracy_score(y_train, y_pred_train))
    test_acc.append(accuracy_score(y_test, y_pred_test))
    
# Plot accuracy by power parameter
plt.figure(figsize=(8,6))
plt.plot(p_metrics, train_acc, label='Train Accuracy')
plt.plot(p_metrics, test_acc, label='Test Accuracy')
plt.ylabel('Accuracy Score')
plt.xlabel('Minkowski Power Parameter (p)')
plt.legend(loc='center right')
plt.show();
```

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/minkowski.png)

Interestingly, both the train and test accuracy improve slightly at p=3 and above. However, varying the value of `p` does not appear to make much of a difference.

## Hyperparameter Tuning with GridSearchCV

Now that we’ve taken a look at how adjusting single parameters affects k-NN performance, let’s see if we can find the optimal parameters for this specific problem. For this, I’ll use sklearn.model_selection’s `GridSearchCV`, which will allow me to test different combinations of k-NN parameters and validate the classifiers using 10-fold cross validation.

When selecting the values of `n_neighbors` for the grid search, I choose only odd numbers. This is often done in k-NN so that there are no ties in the ‘vote’ to classify a new instance.

```
# Define grid search parameters
grid_params = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 51, 101],
              'weights': ['uniform', 'distance'],
              'p': [1, 2, 3, 4, 5]}

# Instantiate grid search for k-NN, with 10-fold cross validation
gs = GridSearchCV(KNeighborsClassifier(), grid_params, cv=10)

# Fit model using titanic data
gs.fit(X, y)

# View best grid search parameters
best_knn = gs.best_params_
print(best_knn)
```

*{'nneighbors': 51, 'p': 1, 'weights': 'distance'}*

These best parameter values are not too surprising - in the individual tests, training and test accuracy seemed to converge at `n_neighbors`=50, ‘p’ didn’t seem to matter too much, and ‘distance’ outperformed ‘uniform’. So these values make sense.

```
# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                       random_state=1)

# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
    
# Fit k-NN classifier and make predictions
knn = KNeighborsClassifier(**best_knn)
knn.fit(X_train, y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# Print results
print(f'Train Accuracy: {train_acc}')
print(f'Test Accuracy: {test_acc}')
```

*Train Accuracy: 0.9161676646706587
Test Accuracy: 0.8565022421524664*

Hyperparameter tuning results in an accuracy score of almost 86% - more than a 2% improvement over the k-NN model. However, we must remember that this is simply one possible train-test-split of the data. Would this classifier still perform as well over many more random splits? Let’s see!

```
train_acc = []
test_acc = []

for i in range(0,100):
    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                       random_state=i)
    
    # Perform feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit k-NN classifier and predict
    knn = KNeighborsClassifier(**best_knn)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_acc.append(accuracy_score(y_train, y_pred_train))
    test_acc.append(accuracy_score(y_test, y_pred_test))

print(f'KNN Mean Train Accuracy: {np.mean(train_acc)}')
print(f'KNN Mean Test Accuracy: {np.mean(test_acc)}')

plt.hist(test_acc)
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title('Test Accuracy for 100 Random Train-Test Splits')
plt.show();
```

*KNN Mean Train Accuracy: 0.9167365269461079
KNN Mean Test Accuracy: 0.8458295964125562*

![](https://raw.githubusercontent.com/douglaspsteen/titanic/master/blog_images/knn_100_hist.png)

Over 100 random train-test splits, the tuned k-NN classifier still performs quite well, with a mean test accuracy of 84.6%!

I used this k-NN classifier to submit test predictions for the Titanic Kaggle competition, and achieved 79.5% accuracy. This may seem like a let-down, but it is a 3% improvement over a simple ‘Gender Classifier’ - predicting that every male passenger died results in a 76.5% classification accuracy. This means that some of the additional features (passenger class, fare, passenger title) are being successfully used by this k-NN model to refine classification.







