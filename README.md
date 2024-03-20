
# Project Title

Zomato Restaurant Clustering And Sentiment Analysis


## Summary
This project entailed the utilization of advanced data analytics techniques to gain a deeper understanding of the restaurants and customer feedback on the popular online food delivery platform, Zomato.

The data procured included information such as the restaurant's name, location, cuisines, ratings, and user reviews.

The initial phase of this project involved rigorous data cleaning and preprocessing to ensure the data's suitability for comprehensive analysis. Then conducted the Exploratory Data Analysis (EDA) on both datasets, providing insights into the dataset's composition and features.

The next step in the project was the implementation of clustering on the restaurant data through the use of the k-means algorithm. The objective of the clustering was to group similar restaurants together and discern patterns within the data. The features employed for the clustering process included cuisines, average cost and average rating. The number of clusters was determined by utilizing the elbow method.

Then proceeded to conduct sentiment analysis on the user reviews to gain a comprehensive understanding of the overall sentiment towards the restaurants. Certain libraries were utilized to classify the reviews as positive, negative, or neutral.

The outcome of the analysis revealed that the restaurants within the city were grouped into ten clusters based on their cuisines and average cost. The sentiment analysis uncovered that, generally, customers held a positive sentiment towards the restaurants.

In conclusion, this project exemplifies the utility of clustering and sentiment analysis in gaining a more profound comprehension of restaurant data on Zomato. The insights procured from the analysis can be of immense benefit to both restaurants and customers in making informed decisions. Furthermore, the project can be extended to other cities or even countries to gain insight into the eating habits and preferences of individuals in different regions.
## Deployment

# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px


import warnings
warnings.filterwarnings("ignore")
%matplotlib inline


import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


import string
import re
import nltk
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import roc_auc_score, f1_score, accuracy_score,silhouette_samples, silhouette_score
from sklearn.metrics import roc_curve
from gensim import corpora


from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from textblob import TextBlob


## Business objective
The Project focuses on Customers and Company, you have to analyze the sentiments of the reviews given by the customer in the data and make some useful conclusions in the form of Visualizations. Also, cluster the zomato restaurants into different segments. The data is vizualized as it becomes easy to analyse data at instant. The Analysis also solves some of the business cases that can directly help the customers finding the Best restaurant in their locality and for the company to grow up and work on the fields they are currently lagging in. This could help in clustering the restaurants into segments
## Data inspection
Restaurant Data:

Name: Name of Restaurants
Links: URL Links of Restaurants
Cost: Per person estimated cost of dining
Collection: Tagging of Restaurants w.r.t. Zomato categories
Cuisines: Cuisines served by restaurants
Timings: Restaurant timings
Review Data:

Reviewer: Name of the reviewer
Review: Review text
Rating: Rating provided
MetaData: Reviewer metadata - Number of reviews and followers
Time: Date and Time of Review
Pictures: Number of pictures posted with review



## Conclusion

Clustering and sentiment analysis were performed on a dataset of customer reviews for the food delivery service Zomato. The purpose of this analysis was to understand the customer's experience and gain insights about their feedback.

The clustering technique was applied to group customers based on their review text, and it was found that the customers were grouped into two clusters: positive and negative. This provided a general understanding of customer satisfaction levels, with the positive cluster indicating the highest level of satisfaction and the negative cluster indicating the lowest level of satisfaction.

Sentiment analysis was then applied to classify the review text as positive or negative. This provided a more detailed understanding of customer feedback and helped to identify specific areas where the service could be improved.

Overall, this analysis provided valuable insights into the customer's experience with Zomato, and it could be used to guide future business decisions and improve the service. Additionally, by combining clustering and sentiment analysis techniques, a more comprehensive understanding of customer feedback was achieved.

Other important discoveries during analysis are -

AB's - Absolute Barbecues, show maximum engagement as it has maximum number of rating on average and Hotel Zara Hi-Fi show lowest engagement as has lowest average rating.

Restaurant Collage - Hyatt Hyderabad Gachibowli is the most expensive restaurant in the locality which has a price of 2800 for order. Hotels like Amul and Mohammedia Shawarma are the least expensive with price of 150.

North Indian food followed by Chinese are best or in demand food as sold by most of the restaurants.

Satwinder singh is the most followed reviewer and on an average he gives 3.5 rating.

Anvesh Chowdary is the top reviewer in terms of the number of reviews they have written.

Modern Indian cuisine is the top cuisine by average cost.
## 🔗 Links
https://github.com/Sajeenafiza/ML-capstone
