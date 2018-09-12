# airbnb_project
A data science project that explore Airbnb data for Boston and Seattle to answer the question 'How do people profit from becoming an Airbnb host?'

There's also a simplified version published on medium. You can follow the link here:
https://medium.com/@agustinus.thehub/this-is-how-you-get-more-bookings-and-higher-ratings-as-airbnb-host-ff2846834b3e



# Installation
Clone this repo: https://github.com/nugroho1234/airbnb_project.git

I use python 3.6 to create this project and the libraries I used are:
1. Pandas
2. Numpy
3. Matplotlib
4. Collections
5. Seaborn
6. Scikit-Learn

# Project Motivation
I used to work in hospitality industry and Airbnb greatly disrupt this industry. I am really interested in this company since it uses the power / resources of people (which I'm a huge fan of) to accomodate travellers, impacting the sales of hotels. So, I decided to do this project to help people to understand how to profit from becoming an Airbnb host.

In this project, I'm going to answer these following questions:
1. What determines customers' buying decision in Airbnb?
2. What influences customers' rating for a house after they stay on it?
3. What kind of amenities a host should have to be able to get high ratings and get customers to use this host's house?

By doing this, I hope I can give insight to present / future Airbnb host on how to get better ratings, higher bookings, and what amenities to provide if they want better ratings and higher bookings.

# File Descriptions
### Airbnb Data Final.ipnyb
This is the file that describes the data science process I used to answer the main question of this project as well as several sub-questions. I used machine learning techniques mostly to fill missing values just to practice.
### Functions.py
This file stores most of my functions that I used in the notebook. 
### The .csv files
These are the files I used to analyze the Airbnb data. These are the listings file from [Boston](https://www.kaggle.com/airbnb/boston) and [Seattle](https://www.kaggle.com/airbnb/seattle). The original source of these files is Kaggle.

# How to use this project
Actually, this project is analytical in nature. So, one thing to do with this improve my results. As I'm still studying data science and machine learning, I could be wrong in implementing the machine learning techniques to fill the null values. 

## Data Preparation
During my data preparation phase, I conducted several things:
1. Changed datatypes for Price and Percentage related variables
2. Created thumbnail_url column to create thumbnail_available column. If there is a url in the thumbnail_url column, I assumed that there is a thumbnail in the house page. Otherwise, there is no thumbnail in the house page.
4. Filling the missing values on 'city', 'state', and 'cleaning_fee'. I imputed using mode for 'city' and 'state'. For 'cleaning_fee', my assumption is that when the row is empty, the host doesn't charge cleaning fee. 
6. Converted Host verifications into verification_method which is the number of verification method available.
7. Filled The NaN values in weekly_price and monthly_price with 0. I also created 2 new columns, price_per_week with 7 x price as the values if the weekly_price value is NaN and price_per_month with 30 x price as the values if the monthly_price value is NaN.
8. host_since will be converted to host_year, indicating how many years has someone been an Airbnb host.
9. Created zipcode_fix column with the values of zipcode predicted. I used decision tree classifier to predict the zipcode values using neighbourhood_cleansed column.
10. Filled null values in the property_type column. I used grid search with random forest classifier to find the best prediction model, then use it to predict property_type column. The function I used is predict_property_type in the functions.py file
11. Filled null values for numerical variables. I used random forest regressor to predict the values. The function I used is predict_rf in the functions.py file.
12. Changed values in the state column to become uppercase
13. Change values in the host_year column from 2018 to 0.  Since this column is generated from host_since column, the 2018 values are actually generated from NaN values, which I assume is not filled because these houses are from new hosts.

## Modeling and Evaluation
To answer the first 2 questions, I used random forest regressor. For the other one, I used descriptive statistics. 

For the first question, I used random forest regressor and compared 2 models which have number_of_reviews and reviews_per_month as target variables and used reviews_per_month in the end because it has lower MSE score indicating that the error rate is lower than the first model. Then, I created a plot to map the model's feature importance to look at the top 3 variables that have the most influences to reviews_per_month. I found that review_scores_rating, minimum_nights, and instant_bookable_t are the top 3 most influencial variables affecting reviews_per_month.

For the second question, I also used random forest regressor to predict review_scores_rating. I found that reviews_per_month, calculated_host_listings_count, and number_of_reviews to be the top 3 predictors of review_scores_rating.

For the third question, I did several steps. First, I converted the column amenities into a list consisting of each item in the column instead of each line. Then, I created a plot on the count of each item and put it in a descending order. The top 3 items are Heating, Kitchen, and Wireless Internet. Afterwards, I created 6 dataframes using groupby, displaying the mean value reviews_per_month and review_scores_rating. Last, I created 4 tuples related to ratings and reviews. Using these tuples, I created a plot comparing rating and reviews for hosts who provide the top 3 items and those who don't. I found that wifi plays an important role in getting good ratings. In getting property bookings, wifi and heating have important roles.


# Licensing, Author, Acknowledgements
I made this project to help people understand how to profit from becoming Airbnb host. If you can improve this, I will be glad to hear from you! Feel free to use / tweak this and mention me so I can take a look at your work. 

As usual, I benefit greatly from stackoverflow and sklearn documentations. I won't be able to live without them :)
