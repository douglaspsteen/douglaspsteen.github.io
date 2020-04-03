---
layout: post
title:      "Obtaining Raw Data from an API: A Python Tutorial"
date:       2020-04-02 13:22:45 -0400
permalink:  obtaining_raw_data_from_an_api_a_python_tutorial
---


As a data scientist, the ability to obtain data through an API is a critical skill. In this post, I provide a brief tutorial on obtaining soccer data from [API-FOOTBALL](https://www.api-football.com/), a RESTful API hosted by the [RapidAPI](https://rapidapi.com/ )marketplace. This tutorial follows the general method for my fourth project completed during the Flatiron School Online Immersive Data Science Bootcamp. The full GitHub repository for this project can be viewed [here](https://github.com/douglaspsteen/Predict_EPL_Matches).

### Step 1: Create a RapidAPI account and subscribe to API-FOOTBALL

Once you’ve created an account and subscribed, you will be able to access your API key, which you’ll need for making calls to the API.

### Step 2: Import necessary packages for working in Python

For this tutorial, we’ll need to import requests, json, and pandas.

```
import requests
import pandas as pd
import json
```

[Requests](https://requests.readthedocs.io/en/master/) is a library that facilities making http requests with Python. [Json](https://docs.python.org/3/library/json.html) is a module that facilitates working with JSON data (the data returned from our API calls will be in JSON). And of course, [pandas](https://pandas.pydata.org/) will help us display our data in dataframes, making it much easier to work with.

### Step 3: Store your API key in a private folder

Remember, you never want to hard-code your API keys/tokens, since anyone will be able to see and use them if you make your project public on GitHub or elsewhere.

This is my preferred method for storing/using API keys & tokens: I opt to store my personal API key in a folder named ‘.secret’ in my personal user folder on my C: drive. If you go this route, your file path to this folder will look something like this:

"/Users/{your_username}/.secret/"

Using your favorite text editor, you can then create a .json file (I named mine “api_football.json”) that contains a dictionary with a single key-value pair representing your API key:

```
{“api_key”: “paste_your_actual_api_key_here”}
```

There! Now, your API key will be located at the following file path:

"/Users/{your_username}/.secret/api_football.json"

And your actual API key won’t need to appear anywhere in your project code! If the specific API you’re accessing requires more keys/tokens, you can simply add more key-value pairs to the dictionary in your JSON file!

### Step 4: Retrieve your API key from local folder

To make this easier, let’s define a quick function that we can use in python:

```
def get_keys(path):
    with open(path) as f:
        return json.load(f)
```

This function will simply return the contents of the JSON file containing our API key(s).

```
keys = get_keys("/Users/{your_username}/.secret/api_football.json")
api_key = keys['api_key']
```

Now we have our key saved in the variable api_key!

### Step 5: Determine which endpoint URL you need, and make a “GET” request

Figuring out which endpoint to use means you first need to figure out what kind of data you’d like to see. For most APIs, this means you’ll have to review the API documentation itself.

For example, let’s say I wanted to return all seasons available for the English Premier League. API-Football’s documentation says that the correct endpoint URL is:

https://api-football-v1.p.rapidapi.com/v2/leagues/seasonsAvailable/{league_id}

So, now we just need the ‘league_id’ for the English Premier League. Digging a little deeper into the documentation, it looks like we can use ‘524’ as the ‘league_id’ for the English Premier League.

Now we can make our “GET” request. Using the requests module, this can be done as follows:

```
url = "https://api-football-v1.p.rapidapi.com/v2/leagues/seasonsAvailable/524"
headers = {
    'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
    'x-rapidapi-key': api_key
    }
resp = requests.request("GET", url, headers=headers)
```

To check to see if our request was successful, we can do the following:

```
resp.status_code == requests.codes.ok
```

If all went well, the response to the above line of code should be ‘True’.

To check the raw data in the response, simply use:

``` 
print(resp.text)
```

{"api":{"results":10,"leagues":[{"league_id":701,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2010,"season_start":"2010-08-14","season_end":"2011-05-17","logo":"https:\/\/media.api-football.com\/leagues\/56.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":false,"players_statistics":false},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":700,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2011,"season_start":"2011-08-13","season_end":"2012-05-13","logo":"https:\/\/media.api-football.com\/leagues\/56.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":false,"players_statistics":false},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":699,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2012,"season_start":"2012-08-18","season_end":"2013-05-19","logo":"https:\/\/media.api-football.com\/leagues\/56.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":false,"players_statistics":false},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":698,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2013,"season_start":"2013-08-17","season_end":"2014-05-11","logo":"https:\/\/media.api-football.com\/leagues\/56.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":false,"players_statistics":false},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":697,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2014,"season_start":"2014-08-16","season_end":"2015-05-24","logo":"https:\/\/media.api-football.com\/leagues\/56.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":false,"players_statistics":false},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":696,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2015,"season_start":"2015-08-08","season_end":"2016-05-17","logo":"https:\/\/media.api-football.com\/leagues\/56.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":false,"players_statistics":false},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":56,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2016,"season_start":"2016-08-13","season_end":"2017-05-21","logo":"https:\/\/media.api-football.com\/leagues\/56.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":true,"players_statistics":false},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":37,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2017,"season_start":"2017-08-11","season_end":"2018-05-13","logo":"https:\/\/media.api-football.com\/leagues\/37.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":true,"players_statistics":true},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":2,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2018,"season_start":"2018-08-10","season_end":"2019-05-12","logo":"https:\/\/media.api-football.com\/leagues\/2.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":0,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":true,"players_statistics":true},"players":true,"topScorers":true,"predictions":true,"odds":false}},{"league_id":524,"name":"Premier League","type":"League","country":"England","country_code":"GB","season":2019,"season_start":"2019-08-09","season_end":"2020-05-17","logo":"https:\/\/media.api-football.com\/leagues\/2.png","flag":"https:\/\/media.api-football.com\/flags\/gb.svg","standings":1,"is_current":1,"coverage":{"standings":true,"fixtures":{"events":true,"lineups":true,"statistics":true,"players_statistics":true},"players":true,"topScorers":true,"predictions":true,"odds":true}}]}}

This messy JSON response contains the data we need, but it is not the format we want to work with.

### Step 6: Convert JSON data to pandas dataframe

To convert our data to a dataframe, we first need to dig a little into the architecture of the JSON response.


```
# Check keys of response
resp.json().keys()
```

dict_keys(['api'])


```
# Check keys at next level of response
resp.json()['api'].keys()
```


dict_keys(['results', 'leagues'])



```
# Create dictionary of results for 'leagues' key
leagues_dict = resp.json()['api']['leagues']
```


```
# Visualize df for all English Premier league seasons available
leagues_df = pd.DataFrame.from_dict(leagues_dict)
display(leagues_df)
```



![](https://raw.githubusercontent.com/douglaspsteen/dsc-capstone-project-v2-online-ds-ft-100719/master/tweet_classification_files/df.png)

And that’s it! We’ve successfully retrieved data from an API and packaged it into a pandas dataframe. Now you can export that data to a csv, continue working with this data in your notebook, or make more API calls!




