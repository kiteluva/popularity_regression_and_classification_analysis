############################################################################################
### COMBINED SPOTIFY POPULARITY ANALYSIS: MULTIPLE REGRESSION MODELS THEN CLASSIFICATION ###
############################################################################################

# Clear environment and set seed for reproducibility
# rm(list = ls())
set.seed(42)

# Load essential libraries
library(tidyverse)
library(caret)
library(corrplot)
library(skimr)
library(DataExplorer)
library(vip)
library(ranger)
library(xgboost)
library(e1071)
library(scales)
library(lubridate)
library(GGally)
library(grid)
library(recipes)
library(pROC)
library(gridExtra)
library(ggpubr)
library(knitr)
library(MLmetrics)
library(RColorBrewer)
library(flextable)
library(officer)
library(dplyr)
library(tidyr)
library(naniar)

# Start PDF device for all outputs
pdf("j-k-combined_reg&classy_spotify_analysis_results_multiple_reg_revised.pdf", width = 15, height = 12)

# ==============================================================================
# Custom Dark Theme Function
# ==============================================================================
theme_dark_custom <- function(title_text = "") {
  theme_dark(base_size = 14) +
    theme(
      plot.background = element_rect(fill = "black"),
      panel.background = element_rect(fill = "gray20"),
      panel.grid.major = element_line(color = "gray40"),
      panel.grid.minor = element_blank(),
      plot.title = element_text(color = "white", size = 18, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(color = "white", size = 14, hjust = 0.5),
      axis.text = element_text(color = "white", size = 12),
      axis.title = element_text(color = "white", size = 12),
      legend.background = element_rect(fill = "gray20"),
      legend.text = element_text(color = "white", size = 16),
      legend.title = element_text(color = "white", size = 16),
      strip.background = element_rect(fill = "gray40"),
      strip.text = element_text(color = "white", size = 14)
    )
}

# =============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION (SHARED)
# =============================================================================

# Load the csv file
spotify_charts <- read_csv("~/school docs/universal_top_spotify_songs.new.csv")

# Initial data exploration (same as before)
# ... (rest of the initial exploration code) ...
cat("====================\n")
cat("Initial Data Exploration\n")
cat("====================\n")

cat("\nSummary:\n")
print(summary(spotify_charts))
cat("\nSkim:\n")
print(skim(spotify_charts))

# Check for missing values
missing_values <- colSums(is.na(spotify_charts))
cat("\nMissing values per column:\n")
print(missing_values[missing_values > 0])

# Create a dark-themed missingness plot manually
missing_heatmap_1 <- gg_miss_var(spotify_charts, show_pct = TRUE) +
  labs(title = "Missing Values per Column (raw data)") +
  theme_dark_custom()

print(missing_heatmap_1)


# =============================================================================
# 2. DATA CLEANING AND FEATURE ENGINEERING (SHARED)
# =============================================================================

# combining all the codes
spotify_charts <- spotify_charts %>%
  group_by(spotify_id) %>%
  mutate(
    market_count = n_distinct(country, na.rm = TRUE),
    other_charted_countries = paste(country[!duplicated(country)], collapse = ", ")
  ) %>%
  slice_max(order_by = popularity, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    artist_count = sapply(strsplit(artists, ","), length),
    snapshot_date = ymd(snapshot_date),
    album_release_date = ymd(album_release_date),
    days_out = as.numeric(snapshot_date - album_release_date),
    is_explicit = as.integer(is_explicit),
    duration_min = duration_ms / 60000
  ) %>%
  select(-duration_ms) # Remove the original duration_ms column

# Verify the column name change (optional)
colnames(spotify_charts)[colnames(spotify_charts) == "duration_min"] <- "duration_min"

view(spotify_charts)
cat("\n====================\n")
cat("Cleaned and Engineered Data (First few rows):\n")
cat("====================\n")
print(head(spotify_charts))

# Prepare dataset for regression modeling
regression_data <- spotify_charts %>%
  select(-country, -other_charted_countries, -snapshot_date, -name, -artists,
         -album_name, -album_release_date, -spotify_id) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.numeric), ~if_else(is.na(.), median(., na.rm = TRUE), .))) %>%
  filter(popularity != 0)
# arrange columns
regression_data <- regression_data %>%
  select(popularity, market_count, daily_rank, days_out, artist_count, daily_movement, weekly_movement,
         duration_min, is_explicit, mode, danceability, energy, loudness, speechiness,
         acousticness, instrumentalness, time_signature, liveness, valence, key, tempo)

cat("\n====================\n")
cat("Regression Modeling Data (Summary):\n")
cat("====================\n")
view(regression_data)

missing_regression <- gg_miss_var(regression_data, show_pct = TRUE) +
  labs(title = "Missing Values per Column (clean data)") +
  theme_dark_custom()

print(missing_regression)
#=====================================================================
## DESCRIPTIVE STATISTICS, SCATTER AND CORRELATION PLOTS
#=====================================================================

# Function to compute basic statistics
spotify_stats <- function(column) {
  stats <- c(
    Mean = mean(column, na.rm = TRUE),
    Median = median(column, na.rm = TRUE),
    SD = sd(column, na.rm = TRUE),
    Variance = var(column, na.rm = TRUE),
    IQR = IQR(column, na.rm = TRUE)
  )
  return(stats)
}

# Loop through columns and compute statistics
stats_results <- lapply(regression_data, spotify_stats)
names(stats_results) <- colnames(regression_data)

# Convert the list of statistics to a data frame for better printing
stats_table <- as.data.frame(stats_results)
print(stats_table)

#------------------------------------------------------------------------------
# Generate the scatter plot matrices
#------------------------------------------------------------------------------

plot1_dark <- ggpairs(
  regression_data,
  columns = 1:21,
  upper = list(continuous = wrap("cor", alpha = 0.7, color = "white")),
  lower = list(continuous = wrap("points", alpha = 0.5, color = "skyblue")),
  diag = list(continuous = wrap("densityDiag", fill = "skyblue", alpha = 0.6))
) +
  theme_dark_custom() +
  labs(title = "Scatter Plot Matrix of Numerical Features")
print(plot1_dark)
#-----------------------------------------------------------------------------
# DESCRIPTIVE STATISTICS
#-----------------------------------------------------------------------------

cat("\n====================\n")
cat("Descriptive Statistics for Key Audio Features (Flextable)\n")
cat("====================\n")

numeric_cols_desc <- regression_data %>%
  select(popularity,danceability, energy, loudness, speechiness,
         acousticness, instrumentalness, liveness, valence, key, tempo)

descriptive_stats <- numeric_cols_desc %>%
  summarise(across(everything(), list(
    Mean = ~round(mean(.), 2),
    SE = ~round(sd(.) / sqrt(n()), 2),
    Median = ~round(median(.), 2),
    SD = ~round(sd(.), 2),
    Min = ~round(min(.), 2),
    Max = ~round(max(.), 2),
    N = ~n()
  ))) %>%
  pivot_longer(cols = everything(),
               names_to = c("Attribute", ".value"),
               names_sep = "_")

if (nrow(descriptive_stats) > 0) {
  table_flex <- flextable(descriptive_stats) %>%
    autofit() %>%
    set_caption(caption = "Descriptive Statistics for Numeric Features") %>%
    fontsize(size = 10, part = "all") %>%
    theme_box() %>%
    bg(bg = "grey20", part = "header") %>%
    color(color = "white", part = "header") %>%
    bg(bg = "grey15", part = "body") %>%
    color(color = "grey85", part = "body") %>%
    border_outer(part = "all", border = fp_border(color = "grey50")) %>%
    border_inner_h(part = "all", border = fp_border(color = "grey40")) %>%
    border_inner_v(part = "all", border = fp_border(color = "grey40"))
  
  print(table_flex)
} else {
  cat("No numeric columns found for descriptive statistics table.\n")
}
# density plots for the features

cat("\n====================\n")
cat("Density Distributions for Numerical Features\n")
cat("====================\n")

numeric_features_for_density <- regression_data %>% 
  select(where(is.numeric))

if (ncol(numeric_features_for_density) > 0) {
  data_long_features <- numeric_features_for_density %>%
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value")
  
  all_features_density_plot <- ggplot(data = data_long_features, aes(x = Value)) +
    geom_density(fill = "#56B4E9", alpha = 0.7, color = "#56B4E9") +
    facet_wrap(~ Feature, scales = "free", ncol = 4) +
    labs(title = "Density Distributions of All Numerical Features",
         x = "Value", y = "Density") +
    theme_dark_custom()
  
  print(all_features_density_plot)
} else {
  cat("No numeric features to plot densities.\n")
}

# Popularity distribution density plot
popularity_dist_density_plot <- ggplot(data = regression_data, aes(x = popularity)) +
  geom_density(fill = "skyblue", alpha = 0.6, color = "white") +
  labs(title = "Popularity Distribution (After Cleaning)",
       x = "Popularity",
       y = "Density") +
  theme_dark_custom()
print(popularity_dist_density_plot)


# Compute correlation matrix
correlation_matrix <- cor(regression_data, use = "complete.obs")

# Create the correlation plot with a dark background
par(bg = "black", mar = c(0, 0, 2, 0)) # Set plotting area background to black
corrplot(
  correlation_matrix,
  method = "color",
  col = colorRampPalette(c("midnightblue", "steelblue", "white", "firebrick", "darkred"))(200),
  tl.col = "white",
  tl.srt = 45,
  type = "upper",
  addCoef.col = "black",
  number.cex = 0.7,
  number.digits = 3,
  main = "Correlation Heatmap"
)
par(bg = "transparent") # Reset plotting area background to transparent for subsequent ggplot2 plots


# Extract the correlations with 'popularity'
popularity_correlations <- correlation_matrix["popularity", ]

# Print the correlations of popularity with each variable
print(popularity_correlations)

# =============================================================================
# 3. MULTIPLE REGRESSION MODEL TRAINING AND EVALUATION
# =============================================================================

cat("\n====================\n")
cat("Multiple Regression Model Training and Evaluation\n")
cat("====================\n")

# Split data for regression
set.seed(42) # Reset seed for consistent split
train_index_reg <- createDataPartition(regression_data$popularity, p = 0.8, list = FALSE)
train_data_reg <- regression_data[train_index_reg, ]
test_data_reg <- regression_data[-train_index_reg, ]

# Create preprocessing recipe for regression
preprocess_recipe_reg <- recipe(popularity ~ ., data = train_data_reg) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

prep_recipe_reg <- prep(preprocess_recipe_reg, training = train_data_reg)
train_processed_reg <- bake(prep_recipe_reg, new_data = train_data_reg)
test_processed_reg <- bake(prep_recipe_reg, new_data = test_data_reg)

# Define resampling strategy
ctrl_reg <- trainControl(method = "cv", number = 5, verboseIter = FALSE, savePredictions = "final")

# Initialize a list to store regression models
regression_models <- list()

# 1. Random Forest
cat("\nTraining Random Forest (Regression)...\n")
regression_models$R.F <- train(
  popularity ~ .,
  data = train_processed_reg,
  method = "ranger",
  trControl = ctrl_reg,
  tuneGrid = expand.grid(mtry = c(5, 7, 9), splitrule = "variance", min.node.size = c(1, 3)),
  importance = 'impurity',
  num.tree = 250
)
cat("Random Forest (Regression) trained.\n")
print(regression_models$R.F)
saveRDS(regression_models$R.F, "RF_model.rds")
# 2. Gradient Boosting
cat("\nTraining Gradient Boosting (Regression)...\n")
regression_models$GBM <- train(
  popularity ~ .,
  data = train_processed_reg,
  method = "gbm",
  trControl = ctrl_reg,
  tuneGrid = expand.grid(n.trees = c(150, 250), interaction.depth = c(3, 5), shrinkage = c(0.1, 0.5), n.minobsinnode = 10),
  verbose = FALSE
)
cat("Gradient Boosting (Regression) trained.\n")
print(regression_models$GBM)
saveRDS(regression_models$GBM, "GBM_model.rds")
# 3. XGBoost
cat("\nTraining XGBoost (Regression)...\n")
regression_models$XGB <- train(
  popularity ~ .,
  data = train_processed_reg,
  method = "xgbTree",
  trControl = ctrl_reg,
  tuneGrid = expand.grid(nrounds = c(150, 250), max_depth = c(3, 6), eta = c(0.1, 0.5), gamma = 0,
                         colsample_bytree = c(0.5, 0.75), min_child_weight = c(1, 3), subsample = 0.75),
  verbose = FALSE
)
cat("XGBoost (Regression) trained.\n")
print(regression_models$XGB)
saveRDS(regression_models$XGB, "XGB_model.rds")
# Evaluate all regression models on the test set
regression_evaluation_results <- lapply(names(regression_models), function(model_name) {
  predictions <- predict(regression_models[[model_name]], newdata = test_processed_reg)
  performance <- postResample(pred = predictions, obs = test_processed_reg$popularity)
  cat(paste0("\nPerformance of ", model_name, " (Regression) on Test Set:\n"))
  print(performance)
  
  # Return both performance metrics and predictions as a list
  return(list(performance = performance, predictions = predictions))
})
names(regression_evaluation_results) <- names(regression_models)

# Compile performance metrics for comparison
regression_results_df <- data.frame(
  Model = names(regression_models),
  RMSE = sapply(regression_evaluation_results, function(x) x$performance["RMSE"]),
  Rsquared = sapply(regression_evaluation_results, function(x) x$performance["Rsquared"]),
  MAE = sapply(regression_evaluation_results, function(x) x$performance["MAE"])
)

# Transpose the dataframe for the desired output
regression_results_df <- as.data.frame(t(regression_results_df[,-1]))
colnames(regression_results_df) <- names(regression_models)
rownames(regression_results_df) <- c("RMSE", "Rsquared", "MAE")


print("\nRegression Model Performance Comparison:")
print(regression_results_df)

# Determine the best performing regression model based on Rsquared
best_reg_model_name <- names(which.max(regression_results_df["Rsquared",]))
best_reg_model <- regression_models[[best_reg_model_name]]

cat(paste("\nBest performing regression model (based on Rsquared):", best_reg_model_name, "\n"))

# Get predictions from the best regression model
predictions_best_reg <- predict(best_reg_model, newdata = test_processed_reg)

# ----------------------------------------------------------------------------------------------------------------------
# Visualization of Regression Model Performance
# ----------------------------------------------------------------------------------------------------------------------

# Create a data frame for plotting performance metrics
performance_plot_data <- gather(regression_results_df, key = "Model", value = "Value") %>%
  mutate(Metric = rep(rownames(regression_results_df), times = length(regression_models)))

# Plotting RMSE with numeric values
rmse_plot <- ggplot(performance_plot_data %>% filter(Metric == "RMSE"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = -0.5), size = 10, col = "gold") + # Add numeric labels
  labs(title = "Comparison of RMSE for Regression Models", y = "RMSE") +
  theme_dark_custom() +
  theme(legend.position = "bottom")
print(rmse_plot)

# Plotting Rsquared with numeric values
rsquared_plot <- ggplot(performance_plot_data %>% filter(Metric == "Rsquared"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = 1), size = 10, col = "gold") + # Add numeric labels
  labs(title = "Comparison of Rsquared for Regression Models", y = "R-squared") +
  theme_dark_custom() +
  theme(legend.position = "bottom")
print(rsquared_plot)

# Plotting MAE with numeric values
mae_plot <- ggplot(performance_plot_data %>% filter(Metric == "MAE"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = 1), size = 10, col = "gold") + # Add numeric labels
  labs(title = "Comparison of MAE for Regression Models", y = "MAE") +
  theme_dark_custom() +
  theme(legend.position = "bottom")
print(mae_plot)

# Create a data frame for plotting predicted vs actual values
plot_data_reg <- data.frame(
  Model = rep(names(regression_models), each = nrow(test_processed_reg)),
  Predicted = unlist(lapply(regression_evaluation_results, function(x) x$predictions)),
  Actual = rep(test_processed_reg$popularity, times = length(regression_models))
)

# Scatter plot of predicted vs actual values for each model
regression_scatter_plot <- ggplot(plot_data_reg, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, color = "gold", linetype = "dashed") +
  geom_smooth(aes(group = Model), method = "lm", se = FALSE) +
  facet_wrap(~Model, scales = "free") +
  labs(title = "Predicted vs. Actual Popularity for Regression Models",
       x = "Actual Popularity",
       y = "Predicted Popularity") +
  theme_dark_custom() +
  theme(legend.position = "bottom")
print(regression_scatter_plot)

# ----------------------------------------------------------------------------------------------------------------------
# Feature Importance Plot (Best Regression Model)
# ----------------------------------------------------------------------------------------------------------------------

# Plot feature importance for the best regression model
vip_plot_reg <- vip(
  best_reg_model,
  main = paste("Feature Importance in", best_reg_model_name, "(Regression)")) +
  theme_dark_custom()

print(vip_plot_reg)


# =============================================================================
# 4. PREPARE DATA FOR CLASSIFICATION (FROM BEST REGRESSION OUTPUT)
# =============================================================================

cat("\n====================\n")
cat("Classification Based on Best Regression Model Predictions\n")
cat("====================\n")

# Define popularity thresholds
very_high_threshold <- 75
high_threshold <- 50
very_low_threshold <- 25

# Categorize the predicted popularity from the best regression model
predicted_popularity_level <- case_when(
  predictions_best_reg >= very_high_threshold ~ "very_high",
  predictions_best_reg < very_low_threshold ~ "very_low",
  predictions_best_reg >= high_threshold ~ "high",
  predictions_best_reg < high_threshold ~ "low"
)
predicted_popularity_level <- factor(predicted_popularity_level, levels = c("very_low", "low", "high", "very_high"))

# Create the actual popularity levels from the test set
actual_popularity_level <- case_when(
  test_data_reg$popularity >= very_high_threshold ~ "very_high",
  test_data_reg$popularity < very_low_threshold ~ "very_low",
  test_data_reg$popularity >= high_threshold ~ "high",
  test_data_reg$popularity < high_threshold ~ "low"
)
actual_popularity_level <- factor(actual_popularity_level, levels = c("very_low", "low", "high", "very_high"))

# Prepare data for classification model training
classification_data <- data.frame(
  popularity_level = actual_popularity_level,  # Use the actual levels from the test set
  test_processed_reg
)
# Remove the original 'popularity' column to avoid redundancy/potential issues
classification_data <- classification_data %>% select(-popularity)

# Split data for classification
set.seed(42)
train_index_class <- createDataPartition(classification_data$popularity_level, p = 0.8, list = FALSE)
train_data_class <- classification_data[train_index_class, ]
test_data_class <- classification_data[-train_index_class, ]

# =============================================================================
# 5. CLASSIFICATION MODEL TRAINING AND EVALUATION
# =============================================================================
cat("\n====================\n")
cat("Classification Model Training and Evaluation\n")
cat("====================\n")

# Custom summary function for multi-class ROC
multiClassSummary <- function (data, lev = NULL, model = NULL) {
  if (length(lev) > 2) {
    rocs <- pROC::multiclass.roc(data$obs, data[, lev])
    auc <- pROC::auc(rocs)
    names(auc) <- "AUC"
    accuracy <- mean(data$obs == data$pred)
    names(accuracy) <- "Accuracy"
    return(c(AUC = auc, Accuracy = accuracy))
  } else {
    return(defaultSummary(data, lev, model))
  }
}

# Define resampling strategy for classification
trainControl_roc <- trainControl(method = "cv",
                                 number = 5,
                                 allowParallel = TRUE,
                                 summaryFunction = multiClassSummary,
                                 classProbs = TRUE,
                                 savePredictions = TRUE)

# Initialize a list to store classification models
classification_models <- list()

# Train a classification model (Random Forest, for example)
cat("\nTraining Classification Model (Random Forest)...\n")
classification_models$R.F <- train(
  popularity_level ~ .,
  data = train_data_class,
  method = "ranger",
  trControl = trainControl_roc,
  tuneGrid = expand.grid(mtry = c(5, 7, 9),
                         min.node.size = c(1, 3, 5),
                         splitrule = "gini"),
  num.trees = 250,
  metric = "AUC"
)
cat("Random Forest Classification trained.\n")
print(classification_models$R.F)
print(classification_models$R.F$results)

# Train XGBoost
cat("\nTraining Classification Model (XGBoost)...\n")
classification_models$XGB <- train(
  popularity_level ~ .,
  data = train_data_class,
  method = "xgbTree",
  trControl = trainControl_roc,
  tuneGrid = expand.grid(nrounds = c(150, 250),
                         max_depth = c(3, 6),
                         eta = c(0.1, 0.5),
                         gamma = 0,
                         colsample_bytree = c(0.5, 0.75),
                         min_child_weight = c(1, 3),
                         subsample = 0.75),
  metric = "AUC",
  verbose = FALSE
)
cat("XGBoost Classification trained.\n")
print(classification_models$XGB)
print(classification_models$XGB$results)

# Modified plot_roc_curves function to return a ggplot object
plot_roc_curves_gg <- function(model, test_data, model_name) {
  pred_probs_class <- predict(model, newdata = test_data %>% select(-popularity_level), type = "prob")
  class_levels <- levels(test_data$popularity_level)
  roc_objects_class <- list()
  
  # Create ROC objects for each class
  for (i in seq_along(class_levels)) {
    current_class <- class_levels[[i]]
    # Ensure binary_response is a factor with levels 0 and 1 for pROC
    binary_response <- factor(ifelse(test_data$popularity_level == current_class, 1, 0), levels = c(0, 1))
    predictor <- pred_probs_class[, current_class]
    roc_objects_class[[current_class]] <- roc(response = binary_response, predictor = predictor)
  }
  
  roc_data_list <- lapply(names(roc_objects_class), function(class_name) {
    roc_obj <- roc_objects_class[[class_name]]
    data.frame(
      FPR = 1 - roc_obj$specificities,
      TPR = roc_obj$sensitivities,
      Class = class_name
      # REMOVED: AUC = auc(roc_obj) -- This was causing the row mismatch
    )
  })
  
  roc_data_df <- bind_rows(roc_data_list)
  
  # Calculate AUCs separately for annotation
  auc_values <- sapply(roc_objects_class, auc)
  auc_labels <- data.frame(
    Class = names(auc_values),
    AUC = round(auc_values, 3)
  )
  
  roc_plot <- ggplot(roc_data_df, aes(x = FPR, y = TPR, color = Class)) +
    geom_line(size = 1) +
    geom_abline(linetype = "dashed", color = "gold") +
    # Use auc_labels data frame for annotation
    # Adjust y-position based on number of classes to prevent overlap
    geom_text(data = auc_labels,
              aes(x = 0.7, y = 0.1 + (match(Class, class_levels) - 1) * 0.08,
                  label = paste0("AUC (", Class, ") = ", AUC)),
              color = "white", size = 4, hjust = 0) +
    labs(title = paste("One-vs-Rest ROC Curves (", model_name, ")", sep = ""),
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    theme_dark_custom() +
    scale_color_manual(values = c("very_low" = "#66c2a5", "low" = "#fc8d62", "high" = "#8da0cb", "very_high" = "#e78ac3")) +
    theme(legend.position = "bottom")
  
  return(roc_plot) 
}

# Generate ROC plots for both models
roc_rf_plot <- plot_roc_curves_gg(classification_models$R.F, test_data_class, "Random Forest")
roc_xgb_plot <- plot_roc_curves_gg(classification_models$XGB, test_data_class, "XGBoost")

# Arrange ROC curves side-by-side
grid.arrange(roc_rf_plot, roc_xgb_plot, ncol = 2,
             top = textGrob("One-vs-Rest ROC Curves for Classification Models",
                            gp = gpar(col = "white", fontsize = 20, fontface = "bold")))

# ==============================================================================
# Frequency Plots for Popularity Class Labels
# ==============================================================================

# Actual popularity class distribution
actual_class_plot <- ggplot(data.frame(Class = actual_popularity_level), aes(x = Class)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Frequency of Actual Popularity Classes", x = "Class", y = "Count") +
  theme_dark_custom()
print(actual_class_plot)

# Predicted popularity class distribution
predicted_class_plot <- ggplot(data.frame(Class = predicted_popularity_level), aes(x = Class)) +
  geom_bar(fill = "darkorange") +
  labs(title = "Frequency of Predicted Popularity Classes", x = "Class", y = "Count") +
  theme_dark_custom()
print(predicted_class_plot)

# =============================================================================
# 7. CONFUSION MATRIX AND CLASSIFICATION METRICS
# =============================================================================

cat("\n====================\n")
cat("Confusion Matrix and Classification Metrics\n")
cat("====================\n")

# Function to evaluate and print classification metrics
evaluate_classification_model <- function(model, test_data, model_name) {
  # Make predictions on the test set
  predictions_class <- predict(model, newdata = test_data %>% select(-popularity_level))
  
  # Create confusion matrix
  conf_matrix_class <- confusionMatrix(data = predictions_class, reference = test_data$popularity_level)
  
  # Print confusion matrix
  cat(paste("\nConfusion Matrix (", model_name, " on Test Set):\n", sep = ""))
  print(conf_matrix_class)
  
  # Extract overall metrics
  overall_metrics_class <- data.frame(conf_matrix_class$overall)
  cat("\nOverall Classification Metrics:\n")
  print(overall_metrics_class)
  
  # Extract class-specific metrics
  class_metrics_class <- data.frame(conf_matrix_class$byClass)
  cat("\nClass-Specific Classification Metrics:\n")
  print(class_metrics_class)
  
  return(list(confusion_matrix = conf_matrix_class, 
              overall_metrics = overall_metrics_class,
              class_metrics = class_metrics_class))
}

# ggplot confusion matrices
plot_confusion_matrix_gg <- function(conf_matrix_obj, model_name) {
  cm_df <- as.data.frame(conf_matrix_obj$table)
  colnames(cm_df) <- c("Prediction", "Reference", "Freq")
  
  ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", size = 5) +
    scale_fill_gradient(low = "skyblue", high = "blue") +
    labs(title = paste("Confusion Matrix -", model_name),
         x = "Predicted Class", y = "Actual Class") +
    theme_dark_custom()
}

# Evaluate Random Forest and XGBoost
rf_evaluation_results <- evaluate_classification_model(classification_models$R.F, test_data_class, "Random Forest")
xgb_evaluation_results <- evaluate_classification_model(classification_models$XGB, test_data_class, "XGBoost")

# ...
# Get confusion matrix plots
rf_cm_plot <- plot_confusion_matrix_gg(rf_evaluation_results$confusion_matrix, "Random Forest")
xgb_cm_plot <- plot_confusion_matrix_gg(xgb_evaluation_results$confusion_matrix, "XGBoost")

# Arrange confusion matrices side-by-side
grid.arrange(rf_cm_plot, xgb_cm_plot, ncol = 2,
             top = textGrob("Confusion Matrices for Classification Models",
                            gp = gpar(col = "white", fontsize = 20, fontface = "bold")))

# =============================================================================
# 8. PDF REPORT GENERATION (COMBINED WITH MULTIPLE REGRESSION MODELS)
# =============================================================================

# Stop PDF device
dev.off()
cat("All plots have been saved to j-k-combined_reg&classy_spotify_analysis_results_multiple_reg_revised.pdf
      in your working directory.\n")