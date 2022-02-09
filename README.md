# Data Mining Project
## Introduction
The dataset in this project is scraped from Kickstarter, which is a popular crowdfunding platform. This project aimed to develop a classification model and a clustering model for Kickstarter, so that it can group and predict the status of the incoming project based on their features provided to Kickstarter before launch. In this report, I will first discuss the data preprocessing and feature selections for the Kickstarter dataset, following by the methodologies and rational for each model. I will also discuss the business insights I mined from the dataset using the models I build. 
## Data Preprocessing:
The Kickstarter dataset is comprised of 18,568 projects, where each project has 45 related features’ data. However, most of the features are available only after the project is launched, such as backers counts, and staff pick. Thus, those features should be dropped during data preprocessing. Secondly, I consider “name” and “project id” irrelevant features because models’ performances should not rely on or affect by these two features. Moreover, I dropped the date- related features such as “created at month” and “deadline hr” because these features are again irrelevant to the status of projects. After dropping those features, I found that the feature “category” has around 1500 missing values. Based on the underlying distribution of this variable, I then filled the missing according to the probability.

The final set of candidate features includes 12 predictors. However, in these 12 predictors, there are multiple categorical variables such as “category” and “country”. I first dummified the boolean variable “disable_communication” and the variable “category”. I dummified “category” because I think each of the categories can be used to identify clusters when building the clustering model. For the rest of the category variables: country and currency, I did not simply dummified them. For variable “country”, the country US has over 10,000 counts. Thus, I simply dummified this variable based on US and non-US. The same logic can be applied to the variable “currency”.

Finally, the variable “static_usd_rate” represents the exchange rate between other foreign currencies to USD. I decided to multiply the rate with the goal so that the final goal values are normalized based on USD.
## Classification Model
To build the classification model, I first performed feature selection on the set of pre- processed variables. I tried two feature selection methods: Random Forest and PCA. For PCA, I set the number of components to be 2, because the first two components can explain 99% of the variance. For Random Forest, I set the feature importance score to be 0.01 and obtained 12 predictors. However, the Random Forest method does not take care of correlated variables, I checked the correlation matrix of the 12 features and manually removed the highly correlated terms.

The Random Forest selected and PCA selected predictors were then fed into different classification models to select the best-performed models. The proposed models were: Logistic Regression, KNN, Random Forest, and Gradient Boost. Since KNN and Gradient Boost are required to input the number of nationhood and minimum sample split, I used loops to find the optimal input values for these two models. After performing cross-validation on two datasets using four models, I obtained the following accuracy scores:
<p align="center">
  <img src="https://github.com/yu1inhong/data_mining_and_visualization/blob/main/image/dataframe.png" alt="drawing" width="400"/>
</p>
Based on the result, the best model is the Gradient Boost model and the input predictors for this model are predictors selected by Random Forest method.
  
 ## Clustering Model
The two types of clustering models that I intended to use are K-Means and DBSCAN. The result I obtained from K-Means was reasonable. Thus, I think there is no need to perform DBSCAN again with the same data. Also, the data I fed into the K-Means model is the dataset I obtained after the data preprocessing step.
  
The only parameter I needed to tune is the number of clusters. Utilizing Silhouette Scores, I found that the optimal number of clusters is 2, which is a natural choice because ultimately these projects have two outcome states. However, when clusters‘ number equals 3 the score does not vary much (decrease by 0.005). Hence, I decided to analyze the clustering result with 3 clusters. 
<p align="center">
  <img src="https://github.com/yu1inhong/data_mining_and_visualization/blob/main/image/kmeans_Result.png" alt="drawing" width="400"/>
</p>
<p align="center">
    K-Means Result
</p>
Based on the result, the two variables that affect the clustering outcome the most are normalized goal and US, two predictors I created during the data preprocessing stage. Cluster 2 has the highest average normalized goal value, and all the projects were in the US. Cluster 1 has the second-highest average normalized goal, but none of the projects were in the US. Cluster 3 has the lowest average normalized goal, and it is a mix of US projects and projects from other countries. Moreover, Cluster 2 (highest goal) has the shortest create to launch days, while Cluster 3 has the longest create to launch days.

## Business Insight
These two models could provide rich insights to whoever wants to launch their projects through Kickstarter. For example, some of the categories are more popular and have a greater impact on the state of the projects. Fundraisers should carefully choose their projects’ categories.
