Authors: Mateo Chapatte, Guillaume Vray, Guillaume Parchet

## Introduction to the Tesco paper:

The paper presents the Tesco Grocery 1.0 dataset: a record of 420 M food items purchased by 1.6 M fidelity card owners who shopped at the 411 Tesco stores in Greater London over the course of the entire year of 2015. The data is aggregated at multiple levels of census areas (borough, ward, MSOA, LSOA) to preserve anonymity.

The paper then describes the derivations of mutiple food characteritics (associated to each area) that mainly concern the ernergy, weight, type and entropy of the bought products.

The paper then aims to establish the validity of the dataset. To do so it compares food purchase volumes to population from census to assess representativeness and match nutrient and energy intake to official statistics of food-related illnesses.


## For our technical extension we chose to propose an additional research questions that can be answered using the dataset used in the paper.

### I. Title:

Impact of wealth on eating habits in Greater London

### II. Abstract:

While the Tesco paper validates its dataset in representativeness and ecologically, we propose to use it in order to study the impact of wealth on eating habits in Greater London at MSOA level in 2015 and see if there are inequalities. We use official geographically-salient MSOA indicators which can be found in the official London site to associate each MSOA a wealth indicator (based on the median household income estimate). We represent eating habits by the consumed weight of each nutrient & food category as well as their entropy. We observe the distribution differences using low dimensional visualization methods and geographic heatmaps. Then we study whether those differences are closely related to social class difference with comparative plots and logistic regression. Using these results, we observe if certain nutrients and food categories are more related to a class than another and conclude whether or not there is a relationship between wealth and healthier food habits.
 
### III. Research questions:

1) At MSOA level in Greater London, are there major differences in eating habits depending on the wealth level?
2) Is there evidence of social class difference in eating habits?
3) Do wealthier MSOA areas buy food that could be judged healthier?
 
 
### IV. Proposed Datasets:
 - ["Area-level grocery purchases"](https://figshare.com/collections/Tesco_Grocery_1_0/4769354/2) datasets from the Tesco paper "Tesco Grocery 1.0, a large-scale dataset of grocery purchases in London". We will especially use the dataset on MSOA area level (msoa-data.csv, 895Ko). This dataset will give all the informations related to bought products, their categories and associated nutrients (food informations). It also presents the representativeness of each area which will allow us to better qualify our results.
 - ["year_msoa_grocery.csv"](https://data.london.gov.uk/dataset/msoa-atlas) dataset (2363Ko) issued by Greater London Authority and shared by London Datastore. It contains a wide range of statistial informations on MSOA's population, this will help us to find an estimate of the wealth level of each MSOA. We will probably use the "median income" to estimate the wealth level but it also contains information like "Employement rate", "Median house price", "Median household income estimate", "% adult with and without qualifications" (which could be used to further refine the wealth estimation).
The main advantage of these two datasets is that they are both indexed by the same MSOA IDs, making the merging task much easier. The two datasets can be found online (see the refered linked above) and come from credible sources (replication of Tesco paper was succesful and the second dataset contain official statistics).
 
### V. Methods:

The main methods we might use are:
 - K-means
 - T-sne visualization
 - Logistic regression
 - Spearman correlation
 - Statistical tests (statmodels)
 
### VI. Organization of the work (notebook):

#### 1) Data preparation
 - Import the different datasets and apply cleaning methods if required
 - Keep only the MSOA with representativeness > 0.1
 - Compute additional required features (the fraction of nutrients)
 - Compute a MSOA population class label using K-means on the median income marker

#### 2) Visualization of eating habits at MSOA level
 - Visualize eating habits with T-sne using with fractions of nutrients and food products categories in order to answer our first question
 - Visualize fraction of each nutrient at MSOA level on a london map using plotly, geopandas
 - Compare the last visualization with a new heatmap visualization that compares the wealth markers and fraction of food categories / nutrients
 - Hypothetize about questions 2 and 3

#### 3) Verify the hypotheses
 - Within each class, compare distribution of nutrient fraction and entropy using histograms plots
 - Analyse if there are significant difference distribution of nutrients / food categories within each population class using statistical tests (Spearman)
 - Use logistic regression to predict high or low wealth using nutrients and entropy ; then observe if there are significant relations
 - Interpret the results to observe if in general "healthy" food habits can be correlated with wealth level

#### 4) Share the results of the study
 - Create a [short video](https://www.youtube.com/) pitching our technical extension
 - Create a [data story](https://chapattemateo.github.io/ADACADABRA/) to visualize and understand the results easily
 
#### 5) Additional information:
Here is an approximate list of each member's participation to the tasks:
 - Mateo Chapatte: heatmap visu, histogram plots, website setup, importing and reworking the graphs into the story, shoot and edit the video
 - Guillaume Parchet: data preparation, k-means clustering, spearman correlations, interpretation of the results, writing data story, new README
 - Guillaume Vray: goepandas & London choropleth map, statistics on wealth classes, piechart visu, logistic regression, writing data story
