# A MACHINE LEARNING APPROACH TO PREDICTING SPOTIFY SONG POPULARITY USING INTRINSIC AUDIO FEATURES

1.	INTRODUCTION.

1.1	ABOUT SPOTIFY.

Spotify is a music streaming app used worldwide. It provides extensive song data on it’s API. We will be using this data to create models that can be used to predict the popularity of new music.
The music industry has undergone a significant transformation, shifting from physical formats like CDs to digital music streaming platforms such as Spotify which has millions of songs.

1.2	OBJECTIVES.

Main Objective:
○ To build a predictive model that estimates a song's likelihood of success using its audio features.
● Specific Objectives:
○ Analyze how different song features contribute to popularity.
○ Compare various machine learning approaches (like regression, classification) to find the most accurate prediction method.
○ Evaluate our model's performance using key metrics like R-squared and AUC.

1.3	REFRENCES.

○ Duman et al. (2022) specifically studied dance music features (energy, danceability, loudness) and their impact on listener preferences, noting their role in mood regulation.
As studies continued, limitations in purely audio-based predictions surfaced.
○ Studies (e.g., Dong et al., Nijkamp) found out that audio features alone show weak links to popularity, suggesting that other external factors like marketing and social media may be crucial in predicting song popularity.
○ Due to these weak linear links, Xing (2023), began exploring nonlinear approaches for better fit.

2.	DATA ANALYSIS.

2.1 METHODOLOGY

To build our machine learning models, we would employ:
Regression Approach:
○ Objective: To predict a continuous numerical output (e.g., song popularity score).
○ How it works: Establishes and models relationships between independent variables (song features) and a dependent variable (popularity) to forecast its value.
We will use Random Forest, Xgboost and GBM models.
Classification Approach:
○ Objective: To assign data points to one of several predefined categories or classes.
○ How it works: Converts the continuous popularity variable into categorical data (Very Low, Low, High, Very High) and predicts which category a song belongs to based on its attributes.
Here we will use Random Forest and Xgboost models

2.2 DATA CLEANING

Data Source:
- Retrieved from Kaggle, originally extracted from the Spotify Application Programming Interface (API).
- Initial dataset: 1,750,032 records and 25 attributes.
- Processed dataset: 20 key attributes after removing irrelevant ones
(e.g., song names, artist names).
Dependent Variable:
Popularity: - A value from 0-100, where 100 indicates the most popular.
Independent Variables:
The other 19 song features, such as danceability, loudness, energy, key, acousticness, mode, speechiness, instrumentalness, valence, time signature, liveness, and tempo.
Initial Data Cleaning involved:
○ Removing duplicate data entries.
○ Missing values were imputed with the median due to left-skewed data and outliers.
○ Songs with a popularity score of 0 were removed to focus on factors influencing relative success.

2.2 EXPLORATORY ANALYSIS.

About our dependent variable:
● Most frequent popularity scores fall between 60 and 65.
● The distribution is left-skewed, meaning there are more songs with moderately lower popularity scores than extremely high ones.
● There are very few songs with extreme popularity scores.
Correlation Heatmap :
Shows that while some variables exhibit moderate correlations with "popularity," many pairs have absolute correlation values below 0.1, signaling weak or insignificant linear associations.
The Pearson correlation coefficient, measuring linear dependence, struggles to capture the nonlinear relationships.


2.3 REGRESSION

Methodology:
● Data Split: 80% training, 20% testing.
● Robustness: 5-fold cross-validation on training data to ensure reliable model evaluation.
● Models: Random Forest, XGBoost, and Gradient Boosted Machines (GBM).
● Metrics: R-squared and RMSE.
R-squared Comparison :
● Our regression analysis showed comparable R-squared values (about 0.59-0.61) across Random Forest (RF), XGBoost (XGB), and Gradient Boosted Machines (GBM) in predicting song popularity.
● This implies that these features account for more than half of the variation in popularity, but other unmodeled factors likely also have an impact.


2.4 CLASSIFICATION

Methodology:

● Categorization: Continuous popularity converted to these four discrete classes.
● Evaluation Strategy: Used the same 80/20 train-test split and 5-fold cross-validation.
● Metrics: Model performance assessed using ROC curves and AUC for each category (one-vs-rest strategy).
ROC and AUC for Random Forest :
● AUC values for Random Forest and XGBoost exhibit strong predictive power across all categories. They have AUC values ranging from 0.77 to 0.91
● The curve for "very_low" shows a steeper rise towards the top-left corner, suggesting a strong ability to achieve high sensitivity with a low false positive rate for this category.
● The models have almost similar results.


3.	CONCLUSIONS

3.1 SUMMARY

Exploratory Data Analysis:
● Song popularity exhibited a non-normal distribution.
● We observed a complex relationships between audio features and popularity.
Machine Learning Approaches:
● Both regression and classification models showed predictive capabilities.
● Classification was more effective at distinguishing between popularity levels.
Model Performance:
● Regression (XGBoost): Moderate predictive power (R² approx 0.61).
● Classification (Random Forest): Strong performance (high AUC values).

3.2 CONCLUSSIONS

● Our analysis confirmed that audio features are crucial in understanding musical appeal that is providing valuable insights into their contribution to popularity.
● We successfully developed machine learning models that leverage audio features to estimate a song's potential popularity level.
● We compared regression and classification models, finding classification (especially Random Forest with high AUC) offered superior accuracy for discerning popularity levels.
● We built an App that predicts song popularity from Spotify audio features.This tool not only serves as a practical demonstration of the project's findings but also offers artists, producers, and record labels a data-driven glimpse into a song's potential reception in the competitive music market.

3.3 LIMITATIONS

● Dataset Constraints:
○ Some audio features (e.g., acousticness, instrumentalness) may have weak correlations with popularity, adding noise.
● Low Correlations:
○ Weak feature interactions challenge model effectiveness.
● Missing External Factors:
○ Important external influences on popularity (marketing, artist fame, playlisting, social media trends, release timing) were not include.

 3.4 RECOMMENDATIONS

● Expanding Data Sources:
○ Incorporate social media trends, listener demographics, & contextual factors.
● Enhanced Feature Engineering:
○ Investigate interaction effects between independent variables.
● Develop more efficient data cleaning techniques to optimize predictions
● Time Series Analysis: Model popularity changes over time.
● Explore deep learning & hybrid models for better accuracy

