---
layout: post
title:      "Handling categorical variables with statsmodels' OLS"
date:       2019-10-28 03:50:07 +0000
permalink:  handling_categorical_variables_with_statsmodels_ols
---


I am just now finishing up my first project of the Flatiron data science bootcamp, which includes predicting house sale prices through linear regression using the King County housing dataset. This project has helped clarify many fresh concepts in my mind, not least of which is the creation of an efficient data science workflow. For this project, my workflow was guided by OSEMiN approach, an acronym for ‘Obtain, Scrub, Explore, Model, and iNterpret’.

During the ‘Scrub’ portion of my work on the King County data, I was left scratching my head at how to handle the ‘Zip Code’ feature as an independent variable. I knew that it should be treated as categorical, since the ~70 unique zip codes clearly did not have an ordinal relationship. So, I performed label encoding on the column with help from pandas, using the code below:

```
df_zip.zipcodes = df_zip.zipcodes.astype('category')
df_zip['zip_coded'] = df_zip.zipcodes.cat.codes
```

However, remembering our lesson on ‘Dealing with Categorical Variables’, I knew that this would still not allow me to use the ‘Zip Code’ feature in a linear regression model – this would require one-hot encoding of the variable.

For those unfamiliar with the concept, one-hot encoding involves the creation of a new 'dummy' variable for each value present in the original categorical variable. The resulting new variables become ‘binary’, with a value of ‘1’ indicating presence of a specific categorical value, and ‘0’ representing its absence (hence the name, ‘one-hot’). Additionally, when using one-hot encoding for linear regression, it is standard practice to drop the first of these ‘dummy’ variables to prevent multicollinearity in the model.

So, in the case of the ‘Zip Code’ feature in the King County dataset, one-hot encoding would leave me with about seventy (70) new dummy variables to deal with. During my initial ‘Scrub’ phase, I then decided that the cumbersome zip codes probably wouldn’t be very important to my regression model, and dropped them from my dataframe. I figured that this information might also be sufficiently captured by latitude and longitude. So, I moved on and kept scrubbing.

When I finally fit the initial linear regression model, my r-squared value of 0.59 left a lot to be desired. I had selected the five most important features using recursive feature elimination (RFE) with the help of sklearn. My five selected features were: 1) living area of neighborhood homes, 2) distance from downtown Seattle, 3) home size (above ground), 4) view, and 5) construction/design grade. 

![](https://github.com/douglaspsteen/dsc-v2-mod1-final-project-online-ds-ft-100719/blob/master/initial%20regression%20model%20output.png?raw=true)

Now I had a feeling that my decision to scrap the zip codes had been a bit too rash, and I decided to see how they would affect my revised model. Luckily, this same day my instructor James Irving had provided some guidance on how to perform one-hot encoding of categorical variables within statsmodels’ ordinary least squares (OLS) class, thus avoiding the need to manually create ~70 dummy variables! Below is an example of how this can be performed for the zip codes variable in the King County data set:

```
import statsmodels.formula.api as smf

f_rev = 'price~C(zip_coded)+grade+sqft_living15_log+dist_dt+sqft_above_log'
model_rev = smf.ols(formula=f_rev, data=df_rev).fit()
model_rev.summary()
```

And here is the output from my revised linear regression model:

![](https://github.com/douglaspsteen/dsc-v2-mod1-final-project-online-ds-ft-100719/blob/master/revised%20regression%20model%20output.png?raw=true)

Including the zip code information in my regression model improved my r-squared value to 0.77. This adjustment also improved the root mean squared error (RMSE) of my model residuals from $123k to $92k. As you can see above, the interpretation of the zip code variable is not as straightforward as continuous variables – some zip codes produce a positive slope coefficient, some produce a negative one, and some don’t even produce a statistically significant result. However, knowing the zip code of a home appears to be critical to making a more accurate prediction of price. This is further illustrated in the figure below, showing median house sale prices for each zip code in King County:

![](https://github.com/douglaspsteen/dsc-v2-mod1-final-project-online-ds-ft-100719/blob/master/King%20County%20median%20price%20by%20zip.png?raw=true)

So, if you’re like me and don’t like to clutter up your dataframe withan army of dummy variables, you may want to give the category indicator within statsmodels’ OLS a try.
