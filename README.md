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

# File Descriptions
### Airbnb Data Final.ipnyb
This is the file that describes the data science process I used to answer the main question of this project as well as several sub-questions. I used machine learning techniques mostly to fill missing values just to practice.
### Functions.py
This file stores most of my functions that I used in the notebook. 
### The .csv files
These are the files I used to analyze the Airbnb data. These are the listings file from [Boston](https://www.kaggle.com/airbnb/boston) and [Seattle](https://www.kaggle.com/airbnb/seattle). The original source of these files is Kaggle.

# How to use this project
Actually, this project is analytical in nature. So, one thing to do with this improve my results. As I'm still studying data science and machine learning, I could be wrong in implementing the machine learning techniques to fill the null values. 

I chose random forest (both regressor and classifier) to predict the null values mostly. This is because I find it easier to map the feature importance related to the model. I tried to use linear regression for prediction purposes, but the r2_score is a bit weird (more than 1.0). From what I understand, it is supposed to be between 0.0 and 1.0. Therefore, I used MSE score to evaluate the models mostly.

The questions are answered and for me they make sense. I found that number of reviews / reviews per month is influenced by ratings, host experience, minimum nights, and instant bookability. However, I also found out that ratings are influenced by number of reviews, reviews per month, and price. There might be a reciprocal effect here, or perhaps mediation effect.

# Licensing, Author, Acknowledgements
I made this project to help people understand how to profit from becoming Airbnb host. If you can improve this, I will be glad to hear from you! Feel free to use / tweak this and mention me so I can take a look at your work. 

As usual, I benefit greatly from stackoverflow and sklearn documentations. I won't be able to live without them :)
