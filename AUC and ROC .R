
##############################
### CODE FOR AUC & ROC ###
##############################

# Load necessary libraries
library(tidyverse)
library(caret)
library(ranger)
library(lubridate)
library(pROC)
library(readr)
library(ggplot2)

# ------------------------
# 1. Load & Clean the Data
# ------------------------

# Load your dataset (adjust the file path accordingly)
spotify_charts_2024 <- read_csv("~/school docs/universal_top_spotify_songs.new.csv")

# Convert date columns and calculate difference in days
spotify_charts_2024 <- spotify_charts_2024 %>%
  mutate(snapshot_date = ymd(snapshot_date),
         album_release_date = ymd(album_release_date),
         days_out = as.numeric(snapshot_date - album_release_date))

# Remove duplicates based on the spotify_id column while retaining all columns
spotify_charts_2024 <- spotify_charts_2024 %>%
  distinct(spotify_id, .keep_all = TRUE)

# Remove unneeded columns
spotify_charts_2024 <- spotify_charts_2024 %>%
  select(-country, -snapshot_date, -name, -artists, -album_name, -album_release_date, -spotify_id)

# Convert 'is_explicit' (boolean) to integer
spotify_charts_2024$is_explicit <- as.integer(spotify_charts_2024$is_explicit)

# Handle missing values in numeric columns only
numeric_cols <- sapply(spotify_charts_2024, is.numeric)
spotify_charts_2024[numeric_cols] <- lapply(spotify_charts_2024[numeric_cols],
                                            function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# Standardize 'duration_ms' to minutes, then remove the original column
spotify_charts_2024 <- spotify_charts_2024 %>%
  mutate(duration_min = duration_ms / 60000) %>%
  select(-duration_ms)

# ------------------------
# 2. Prepare Data for Classification
# ------------------------

# Convert 'popularity' into a binary factor.
# This assigns popularity into two levels: "Low" and "High."
# 'make.names' ensures the levels are valid R variable names.
spotify_charts_2024 <- spotify_charts_2024 %>%
  mutate(popularity = ifelse(popularity >= 50, "High", "Low")) %>%
  mutate(popularity = make.names(popularity))

# Define feature columns (adjust these names if needed)
feature_columns <- c("daily_rank", "duration_min","daily_movement","weekly_movement",
                     "days_out", "is_explicit", "mode", "danceability", "energy", "loudness",
                     "speechiness", "acousticness", "instrumentalness", "time_signature", 
                     "liveness", "valence", "key", "tempo")
View(spotify_charts_2024)
# Create a dataset with predictors and the target variable
class_data <- spotify_charts_2024 %>% 
  select(all_of(feature_columns), popularity)

# Split the dataset into training (80%) and testing sets
set.seed(50)
trainIndex <- createDataPartition(class_data$popularity, p = 0.8, list = FALSE)
train_data  <- class_data[trainIndex, ]
test_data   <- class_data[-trainIndex, ]

str(train_data)
# ------------------------
# 3. Train the Random Forest Classifier
# ------------------------

# Train the classifier using the "ranger" method with cross-validation.
# The 'twoClassSummary' along with 'metric = "ROC"' will use the ROC AUC for tuning.
trainControl(allowParallel=TRUE)
rf_model_class <- train(popularity ~ ., 
                        data = train_data, 
                        method = "ranger",
                        trControl = trainControl(method = "cv", 
                                                 number = 5, 
                                                 classProbs = TRUE, 
                                                 summaryFunction = twoClassSummary),
                        tuneGrid =  expand.grid(mtry = c(5, 7, 9, 11),
                                                min.node.size = c(1, 3, 5),
                                                splitrule = "gini"),
                        num.trees = 200,
                        metric = "ROC")
print(rf_model_class$results)

# ------------------------
# 4. Compute and Plot AUC & ROC Curve
# ------------------------

# Generate predicted probabilities on the test set.
# We request probabilities (type = "prob") for both "Low" and "High" classes.
rf_pred_probs <- predict(rf_model_class, newdata = test_data, type = "prob")
head(rf_pred_probs)
prob_values <- rf_pred_probs$High
# Compute the ROC curve.
# Here, we consider the probability for the "High" class as the predictor.
roc_obj <- roc(response = test_data$popularity, predictor = rf_pred_probs[,"High"])

# Calculate the AUC and print it.
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "\n")

# Plot the ROC curve.
plot(roc_obj, col = "blue", main = "ROC Curve for Popularity Classification")

################################################################################
# Create an empty list to store ROC curves
roc_list <- list()
# Loop through each tuning parameter combination
for(i in 1:nrow(rf_model_class$results)){
  # extract parameters
  mtry_val <- rf_model_class$results$mtry[i]
  node_size_val <- rf_model_class$results$min.node.size[i]
  # make predictions
  predictions <- predict(rf_model_class, newdata = test_data, type= "prob")
  # Compute ROC curve
  roc_curve <- roc(test_data$popularity, predictions[, "High"])
  # Store the ROC curve with parameter labels
  roc_list[[paste("mtry=", mtry_val, " node.size=", node_size_val)]] <- roc_curve
}
###
roc_data <- do.call(rbind, lapply(names(roc_list), function(label) {
  data.frame(
    Specificity = roc_list[[label]]$specificities,
    Sensitivity = roc_list[[label]]$sensitivities,
    Combination = label
  )
}))

# Plot ROC curves with facets for each combination
ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Combination)) +
  geom_line() +
  labs(
    title = "ROC Curves for Different mtry and min.node.size Combinations",
    x = "1-Specificity", y = "Sensitivity"
  ) +
  theme_minimal() +
  facet_wrap(~Combination) +
  scale_x_reverse()
###

####======================================================================######
# Define thresholds
upper_threshold_very_high <- 0.9
upper_threshold_high <- 0.7
lower_threshold_very_low <- 0.1
lower_threshold_low <- 0.3

# Function to classify songs based on probability thresholds
classify_popularity <- function(prob_value) {
  if (prob_value >= upper_threshold_very_high) {
    return("Very High")
  } else if (prob_value >= upper_threshold_high) {
    return("High")
  } else if (prob_value <= lower_threshold_very_low) {
    return("Very Low")
  } else if (prob_value <= lower_threshold_low) {
    return("Low")
  } else {
    return("Uncertain")  # Covers values between 0.3 and 0.7
  }
}

# Apply function to probability values
test_data$Predicted_Popularity <- sapply(rf_pred_probs$High, classify_popularity)

# Combine actual popularity and predicted probabilities
results_df <- data.frame(
  Actual = test_data$popularity,
  High_Probability = rf_pred_probs$High,
  Low_Probability = rf_pred_probs$Low,
  Prediction_probability = test_data$Predicted_Popularity
)

# View the first few rows
print(results_df)

###-------------------------------------
#5.  twst a RANDOM DATASET
#---------------------------------------

# Load necessary libraries
library(tibble)

# Set seed for reproducibility
set.seed(42)

# Generate random dataset
random_dataset <- tibble(
  daily_rank = sample(1:200, 100, replace = TRUE), 
  duration_min = runif(100, 2, 5),  
  daily_movement = rnorm(100, mean = 0, sd = 5),  
  weekly_movement = rnorm(100, mean = 0, sd = 15),
  days_out = sample(1:3650, 100, replace = TRUE),  
  is_explicit = sample(0:1, 100, replace = TRUE),  
  mode = sample(0:1, 100, replace = TRUE),
  danceability = runif(100, 0, 1), 
  energy = runif(100, 0, 1),
  loudness = rnorm(100, mean = -5, sd = 3), 
  speechiness = runif(100, 0, 1), 
  acousticness = runif(100, 0, 1),
  instrumentalness = runif(100, 0, 1),
  time_signature = sample(3:5, 100, replace = TRUE),
  liveness = runif(100, 0, 1), 
  valence = runif(100, 0, 1), 
  key = sample(0:11, 100, replace = TRUE), 
  tempo = runif(100, 60, 180))  

# View first few rows
print(random_dataset)

###test on the model
# We request probabilities (type = "prob") for both "Low" and "High" classes.
rf_pred_probs1 <- predict(rf_model_class, newdata = random_dataset, type = "prob")
head(rf_pred_probs1)
prob_values1 <- rf_pred_probs1$High

####======================================================================######
# Define thresholds
upper_threshold_very_high <- 0.9
upper_threshold_high <- 0.7
lower_threshold_very_low <- 0.1
lower_threshold_low <- 0.3

# Function to classify songs based on probability thresholds
classify_popularity1 <- function(prob_values1) {
  if (prob_values1 >= upper_threshold_very_high) {
    return("Very High")
  } else if (prob_values1 >= upper_threshold_high) {
    return("High")
  } else if (prob_values1 <= lower_threshold_very_low) {
    return("Very Low")
  } else if (prob_values1 <= lower_threshold_low) {
    return("Low")
  } else {
    return("Uncertain")  # Covers values between 0.3 and 0.7
  }
}

# Apply function to probability
random_dataset$PredictedPopularityClass <- sapply(rf_pred_probs1$High, classify_popularity1)
random_dataset <- random_dataset %>%
  mutate(probability= rf_pred_probs1$High)
view(random_dataset)
# Combine actual popularity and predicted probabilities
results_df1 <- data.frame(
  High_Probability = rf_pred_probs1$High,
  Low_Probability = rf_pred_probs1$Low,
  probability_class = random_dataset$PredictedPopularityClass
)

# View the first few rows
print(results_df1)











