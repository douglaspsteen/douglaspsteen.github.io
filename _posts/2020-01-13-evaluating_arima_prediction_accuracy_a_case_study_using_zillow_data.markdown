---
layout: post
title:      "Evaluating Time Series Model Predictions"
date:       2020-01-13 16:16:29 -0500
permalink:  evaluating_arima_prediction_accuracy_a_case_study_using_zillow_data
---

                                                                                                                                                                                                                                                                                                                                                 
For the third project in the Flatiron School Online Data Science Bootcamp, we were tasked with determining the five “best” zip codes for investment based on historical mean home values provided by Zillow. I chose to focus my analysis on Tarrant County, Texas, which spans the western portion of the Dallas – Fort Worth Metroplex and contains approximately 60 zip codes. The project involved the use Autoregressive Integrated Moving Average (ARIMA) modeling techniques, which essentially use the past patterns and variability of a time series to make predictions about future values of that variable. 

The general ARIMA modeling process for each individual zip code was performed as follows:

1. Fit an ARIMA model to the training portion of the monthly time series
2. Generate predictions for the test portion of the time series, and evaluate the quality of the predictions
3. Make forecast predictions beyond the range of the time series data

Clearly, the ability of an ARIMA model to “predict” the test portion of the time series data (Step 2) must be considered when determining which zip codes are “best” for investment. I refer to this from here on as “predictability” of a model. For example, if an ARIMA model forecasts an 80% increase in Return on Investment (ROI) for a zip code over the next three years, but the model can't predict test data with any degree of accuracy (low “predictability”), how confident can we really be in that rosy forecast? 

To assess “predictability” of the ARIMA models fit to each zip code, plenty of model evaluation metrics are available. Here, I discuss three (MSE, RMSE, and MAPE) and provide some justification for my preferred choice in this case.

## MSE
Mean Squared Error (MSE) is a very common loss function used in assessing the quality of model predictions and is especially popular for regression problems. MSE can be calculated as:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e258221518869aa1c6561bb75b99476c4734108e)

where n is the number of data points, y is the vector of observed values of the variable being predicted, and y_hat is the vector of predicted values. The MSE calculation will therefore be in (units of y)^2.

## RMSE
Root-Mean-Square Error (RMSE) essentially provides the same information as MSE; however, the square root of the MSE is taken so that the result is in the same units as variable y. RMSE can be calculated as follows:

![](http://statweb.stanford.edu/~susan/courses/s60/split/img29.png)

where n is the number of data points, y is the vector of observed values of the variable being predicted, and y ̂_hat is the vector of predicted values. Since RMSE is in the same units as variable y, it is often easier to interpret than MSE. For example, if you are trying to test a model that predicts home prices, an RMSE of $10,500 will tell you that your test predictions, on average, are $10,500 away from the observed values.

## MAPE
Mean Absolute Percentage Error (MAPE) is another method for evaluating the accuracy of model predictions, and is commonly used in time series forecasting. MAPE can be calculated using the following equation:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/4cf2158513b0345211300fe585cc88a05488b451)

where n is the number of data points, A is the vector of observed values of the variable being predicted, and F is the vector of predicted values. MAPE is expressed as a % error of predictions relative to observations, and is therefore unitless. 

While comparing time series data from different zip codes (with very different mean home values) for the Zillow project, it became clear that I needed a unitless metric to assess model test predictions. To understand my reasoning, consider the following scenario: over a 3 year testing period, Zip Code A has a mean home value of $200,000, a model prediction RMSE of $2,000, and a model prediction MSE of $4,000,000. Zip Code B has a mean home value of $750,000, a model prediction RMSE of $2,000, and a model prediction MSE of $4,000,000. Since the MSE and RMSE of both zip codes are equal, is it fair to conclude that both zip code time series models have equal prediction accuracy? Of course not, because the *scale* (mean home value) of each time series is very different ($200,000 vs $750,000). It could be argued that the theoretical model for Zip Code B has much better prediction accuracy.

To compare the “predictability” of the models for Zip Code A and Zip Code B, a unitless metric like MAPE is a more appropriate choice, since the value of this metric is not influenced by the *scale* of the values in each time series.

Below is an example of a zip code with high “predictability”. An ARIMA model was fit to the portion of the time series highlighted in blue. The red line / shaded region show the model predictions for the “test” portion of the time series. In this case, MAPE = 1.9% for the model test predictions.

![](https://raw.githubusercontent.com/douglaspsteen/dsc-mod-4-project-online-ds-ft-100719/master/blogimage1.png)

The graph below shows a zip code with low “predictability”. In this case, MAPE = 14.3% for the model test predictions in the red line / shaded region. 

![](https://raw.githubusercontent.com/douglaspsteen/dsc-mod-4-project-online-ds-ft-100719/master/blogimage2.png)

Though the actual mean home values for these two zip codes are very different in *scale*, the unitless MAPE metric allows us to compare the quality of model predictions between the two. In this case, model test predictions for 76126 are much more accurate than for 76132, and 76126 therefore has a much higher model “predictability”.

Though I chose MAPE as a model evaluation metric for this project, there are certainly situations where other metrics, such as MSE and RMSE, might be a better choice. For example, if your goal is to compare the performance of multiple models on the *same time series*, and you wanted to visualize your error in the same units as your target variable, RMSE would be an appropriate option. On the other hand, if you are performing linear regression and want to see how much variance in dependent variable y can be explained by independent variable x, then the coefficient of determination (R^2) will be your best bet. There is no one-size-fits-all model evaluation metric in data science; the most appropriate choice will vary depending on the characteristics of the dataset and the problem to be solved.




