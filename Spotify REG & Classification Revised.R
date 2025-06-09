#===============================#
## REGRESSION & CLASSIFICATION #
#===============================#


library(here)
setwd(here())
set.seed(42)
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
library(DT)
library(htmltools)
library(tibble)

# Start PDF device for all outputs
pdf("jk-combined_reg&classy_spotify_analysis_results_multiple_reg_r-vised.pdf", width = 15, height = 12)

cat("\n==============================================================================\n")
cat("Custom Dark Theme Function\n")
cat("==============================================================================\n")
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

cat("\n=============================================================================\n")
cat("1. DATA LOADING AND INITIAL EXPLORATION (SHARED)\n")
cat("=============================================================================\n")

spotify_charts <- read_csv("~/school docs/universal_top_spotify_songs.new.csv")

cat("\n--------------------------------------------------------------------------\n")
cat("Initial data exploration\n")
cat("--------------------------------------------------------------------------\n")

cat("\nSummary:\n")
print(summary(spotify_charts))
cat("\nSkim:\n")
print(skim(spotify_charts))

missing_values <- colSums(is.na(spotify_charts))
cat("\nMissing values per column:\n")
print(missing_values[missing_values > 0])

missing_heatmap_1 <- gg_miss_var(spotify_charts, show_pct = TRUE) +
  labs(title = "Missing Values per Column (raw data)") +
  theme_dark_custom()

print(missing_heatmap_1)

cat("\n=============================================================================\n")
cat("2. DATA CLEANING AND FEATURE ENGINEERING (SHARED)\n")
cat("=============================================================================\n")

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
  select(-duration_ms)

colnames(spotify_charts)[colnames(spotify_charts) == "duration_min"] <- "duration_min"

view(spotify_charts)

cat("\nCleaned and Engineered Data (First few rows):\n")
print(head(spotify_charts))

cat("\nPreparing dataset for regression modeling\n")
regression_data <- spotify_charts %>%
  select(-country, -other_charted_countries, -snapshot_date, -name, -artists,
         -album_name, -album_release_date, -spotify_id) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.numeric), ~if_else(is.na(.), median(., na.rm = TRUE), .))) %>%
  filter(popularity != 0)

cat("\nArranging columns\n")
regression_data <- regression_data %>%
  select(popularity, market_count, daily_rank, days_out, artist_count, daily_movement, weekly_movement,
         duration_min, is_explicit, mode, danceability, energy, loudness, speechiness,
         acousticness, instrumentalness, time_signature, liveness, valence, key, tempo)

cat("\nRegression Modeling Data (Summary):\n")

view(regression_data)

missing_regression <- gg_miss_var(regression_data, show_pct = TRUE) +
  labs(title = "Missing Values per Column (clean data)") +
  theme_dark_custom()

print(missing_regression)

cat("\n==============================================================================\n")
cat("DESCRIPTIVE & SUMMARY STATISTICS, SCATTER AND CORRELATION PLOTS\n")
cat("==============================================================================\n")

cat("\nDescriptive Statistics for All Numeric Features (Flextable with Dark Theme)\n")

numeric_cols_desc <- regression_data %>%
  select(where(is.numeric))

descriptive_stats <- numeric_cols_desc %>%
  summarise(across(everything(), list(
    Mean = ~round(mean(., na.rm = TRUE), 2),
    SE = ~round(sd(., na.rm = TRUE) / sqrt(n()), 2),
    Median = ~round(median(., na.rm = TRUE), 2),
    SD = ~round(sd(., na.rm = TRUE), 2),
    Min = ~round(min(., na.rm = TRUE), 2),
    Max = ~round(max(., na.rm = TRUE), 2),
    N = ~n()
  ))) %>%
  pivot_longer(cols = everything(),
               names_to = c("Attribute", ".value"),
               names_sep = "_(?=[^_]+$)"
  )

descriptive_stats$Attribute <- as.character(descriptive_stats$Attribute)

if (nrow(descriptive_stats) > 0) {
  table_flex <- flextable(descriptive_stats) %>%
    set_caption(caption = "Descriptive Statistics for All Numeric Features") %>%
    autofit() %>%
    fontsize(size = 10, part = "all") %>%
    theme_box() %>%
    bg(bg = "grey20", part = "header") %>%
    color(color = "white", part = "header") %>%
    bg(bg = "grey15", part = "body") %>%
    color(color = "grey85", part = "body") %>%
    border_outer(part = "all", border = fp_border(color = "grey50")) %>%
    border_inner_h(part = "all", border = fp_border(color = "grey40")) %>%
    border_inner_v(part = "all", border = fp_border(color = "grey40")) %>%
    set_formatter(
      Mean = function(x) sprintf("%.2f", x),
      SE = function(x) sprintf("%.2f", x),
      Median = function(x) sprintf("%.2f", x),
      SD = function(x) sprintf("%.2f", x),
      Min = function(x) sprintf("%.2f", x),
      Max = function(x) sprintf("%.2f", x)
    )
  print(table_flex)
} else {
  cat("No numeric columns found for descriptive statistics table.\n")
}

cat("\n------------------------------------------------------------------------------\n")
cat("Density Distributions for Numerical Features\n")
cat("------------------------------------------------------------------------------\n")

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

cat("\n------------------------------------------------------------------------------\n")
cat("Generate the scatter plot matrices\n")
cat("------------------------------------------------------------------------------\n")

print(is.null(regression_data))

plot1_dark <- ggpairs(
  regression_data,
  columns = 1:21,
  upper = list(continuous = wrap("cor", alpha = 0.7, color = "gold")),
  lower = list(continuous = wrap("points", alpha = 0.5, color = "skyblue")),
  diag = list(continuous = wrap("densityDiag", fill = "skyblue", alpha = 0.6))
) +
  theme_dark_custom() +
  labs(title = "Scatter Plot Matrix of Numerical Features")

print(plot1_dark)

cat("\n----------------------------------------------------\n")
cat("Correlation matrix\n")
cat("------------------------------------------------------\n")

correlation_matrix <- cor(regression_data, use = "complete.obs")

par(bg = "black", mar = c(0, 0, 2, 0))
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
par(bg = "transparent")

cat("\nExtract the correlations with 'popularity'\n")
popularity_correlations <- correlation_matrix["popularity", ]

print(popularity_correlations)

cat("\nPopularity distribution density plot\n")
popularity_dist_density_plot <- ggplot(data = regression_data, aes(x = popularity)) +
  geom_density(fill = "skyblue", alpha = 0.6, color = "white") +
  labs(title = "Popularity Distribution (After Cleaning)",
       x = "Popularity",
       y = "Density") +
  theme_dark_custom()
print(popularity_dist_density_plot)

cat("\n=============================================================================\n")
cat("3. MULTIPLE REGRESSION MODEL TRAINING AND EVALUATION\n")
cat("=============================================================================\n")

cat("\nMultiple Regression Model Training and Evaluation\n")

cat("\nSplit data for regression\n")
set.seed(42)
train_index_reg <- createDataPartition(regression_data$popularity, p = 0.8, list = FALSE)
train_data_reg <- regression_data[train_index_reg, ]
test_data_reg <- regression_data[-train_index_reg, ]

cat("\nCreate preprocessing recipe for regression\n")
preprocess_recipe_reg <- recipe(popularity ~ ., data = train_data_reg) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

prep_recipe_reg <- prep(preprocess_recipe_reg, training = train_data_reg)
train_processed_reg <- bake(prep_recipe_reg, new_data = train_data_reg)
test_processed_reg <- bake(prep_recipe_reg, new_data = test_data_reg)

cat("\nDefine resampling strategy\n")
ctrl_reg <- trainControl(method = "cv", number = 5, verboseIter = FALSE, savePredictions = "final")

regression_models <- list()

cat("\nTraining Random Forest (Regression)...\n")
regression_models$R.F <- train(
  popularity ~ .,
  data = train_processed_reg,
  method = "ranger",
  trControl = ctrl_reg,
  tuneGrid = expand.grid(mtry = c(5, 7, 9), splitrule = "variance", min.node.size = c(1, 3, 5)),
  importance = 'impurity',
  num.tree = 250
)
cat("Random Forest (Regression) trained.\n")
print(regression_models$R.F)
saveRDS(regression_models$R.F, "RF_model.rds")

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

cat("\nEvaluate all regression models on the test set\n")
regression_evaluation_results <- lapply(names(regression_models), function(model_name) {
  predictions <- predict(regression_models[[model_name]], newdata = test_processed_reg)
  performance <- postResample(pred = predictions, obs = test_processed_reg$popularity)
  cat(paste0("\nPerformance of ", model_name, " (Regression) on Test Set:\n"))
  print(performance)
  return(list(performance = performance, predictions = predictions))
})
names(regression_evaluation_results) <- names(regression_models)

cat("\nCompile performance metrics for comparison\n")
regression_results_df <- data.frame(
  Model = names(regression_models),
  RMSE = sapply(regression_evaluation_results, function(x) x$performance["RMSE"]),
  Rsquared = sapply(regression_evaluation_results, function(x) x$performance["Rsquared"]),
  MAE = sapply(regression_evaluation_results, function(x) x$performance["MAE"])
)

regression_results_df <- as.data.frame(t(regression_results_df[,-1]))
colnames(regression_results_df) <- names(regression_models)
rownames(regression_results_df) <- c("RMSE", "Rsquared", "MAE")

cat("\nRegression Model Performance Comparison:\n")
print(regression_results_df)

cat("\nDetermine the best performing regression model based on Rsquared\n")
best_reg_model_name <- names(which.max(regression_results_df["Rsquared",]))
best_reg_model <- regression_models[[best_reg_model_name]]

cat(paste("\nBest performing regression model (based on Rsquared):", best_reg_model_name, "\n"))

cat("\nGet predictions from the best regression model\n")
predictions_best_reg <- predict(best_reg_model, newdata = test_processed_reg)

cat("\n----------------------------------------------------------------------------------------------------------------------\n")
cat("Visualization of Regression Model Performance\n")
cat("----------------------------------------------------------------------------------------------------------------------\n")

cat("\nCreate a data frame for plotting performance metrics\n")
performance_plot_data <- gather(regression_results_df, key = "Model", value = "Value") %>%
  mutate(Metric = rep(rownames(regression_results_df), times = length(regression_models)))

cat("\nPlotting RMSE with numeric values\n")
rmse_plot <- ggplot(performance_plot_data %>% filter(Metric == "RMSE"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = -0.5), size = 10, col = "gold") +
  labs(title = "Comparison of RMSE for Regression Models", y = "RMSE") +
  theme_dark_custom() +
  theme(legend.position = "bottom")
print(rmse_plot)

cat("\nPlotting Rsquared with numeric values\n")
rsquared_plot <- ggplot(performance_plot_data %>% filter(Metric == "Rsquared"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = 1), size = 10, col = "gold") +
  labs(title = "Comparison of Rsquared for Regression Models", y = "R-squared") +
  theme_dark_custom() +
  theme(legend.position = "bottom")
print(rsquared_plot)

cat("\nPlotting MAE with numeric values\n")
mae_plot <- ggplot(performance_plot_data %>% filter(Metric == "MAE"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3), vjust = 1), size = 10, col = "gold") +
  labs(title = "Comparison of MAE for Regression Models", y = "MAE") +
  theme_dark_custom() +
  theme(legend.position = "bottom")
print(mae_plot)

cat("\nCreate a data frame for plotting predicted vs actual values\n")
plot_data_reg <- data.frame(
  Model = rep(names(regression_models), each = nrow(test_processed_reg)),
  Predicted = unlist(lapply(regression_evaluation_results, function(x) x$predictions)),
  Actual = rep(test_processed_reg$popularity, times = length(regression_models))
)

cat("\nScatter plot of predicted vs actual values for each model\n")
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

cat("\n----------------------------------------------------------------------------------------------------------------------\n")
cat("Feature Importance Plot (Best Regression Model)\n")
cat("----------------------------------------------------------------------------------------------------------------------\n")

cat("\nPlot feature importance for the best regression model\n")
vip_plot_reg <- vip(
  best_reg_model,
  main = paste("Feature Importance in", best_reg_model_name, "(Regression)")) +
  theme_dark_custom()

print(vip_plot_reg)

cat("\n=============================================================================\n")
cat("4. PREPARE DATA FOR CLASSIFICATION (FROM BEST REGRESSION OUTPUT)\n")
cat("=============================================================================\n")

cat("\nClassification Based on Best Regression Model Predictions\n")

cat("\nDefine popularity thresholds\n")
very_high_threshold <- 75
high_threshold <- 50
very_low_threshold <- 25

cat("\nCategorize the predicted popularity from the best regression model\n")
predicted_popularity_level <- case_when(
  predictions_best_reg >= very_high_threshold ~ "very_high",
  predictions_best_reg < very_low_threshold ~ "very_low",
  predictions_best_reg >= high_threshold ~ "high",
  predictions_best_reg < high_threshold ~ "low"
)
predicted_popularity_level <- factor(predicted_popularity_level, levels = c("very_low", "low", "high", "very_high"))

cat("\nCreate the actual popularity levels from the test set\n")
actual_popularity_level <- case_when(
  test_data_reg$popularity >= very_high_threshold ~ "very_high",
  test_data_reg$popularity < very_low_threshold ~ "very_low",
  test_data_reg$popularity >= high_threshold ~ "high",
  test_data_reg$popularity < high_threshold ~ "low"
)
actual_popularity_level <- factor(actual_popularity_level, levels = c("very_low", "low", "high", "very_high"))

cat("\nPrepare data for classification model training\n")
classification_data <- data.frame(
  popularity_level = actual_popularity_level,
  test_processed_reg
)
cat("\nRemove the original 'popularity' column to avoid redundancy/potential issues\n")
classification_data <- classification_data %>% select(-popularity)

cat("\nSplit data for classification\n")
set.seed(42)
train_index_class <- createDataPartition(classification_data$popularity_level, p = 0.8, list = FALSE)
train_data_class <- classification_data[train_index_class, ]
test_data_class <- classification_data[-train_index_class, ]

cat("\nDerive scores and predicted levels for the classification test set (for plotting)\n")
actual_popularity_for_classification_test_num <- test_data_reg[-train_index_class, ]$popularity
predicted_numerical_popularity_for_classification_test <- predictions_best_reg[-train_index_class]

predicted_popularity_level_for_plot <- case_when(
  predicted_numerical_popularity_for_classification_test >= very_high_threshold ~ "very_high",
  predicted_numerical_popularity_for_classification_test < very_low_threshold ~ "very_low",
  predicted_numerical_popularity_for_classification_test >= high_threshold ~ "high",
  predicted_numerical_popularity_for_classification_test < high_threshold ~ "low"
)
predicted_popularity_level_for_plot <- factor(predicted_popularity_level_for_plot,
                                              levels = c("very_low", "low", "high", "very_high"))

cat("\n=============================================================================\n")
cat("5. CLASSIFICATION MODEL TRAINING AND EVALUATION\n")
cat("=============================================================================\n")
cat("\nClassification Model Training and Evaluation\n")

cat("\nCustom summary function for multi-class ROC\n")
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

cat("\nDefine resampling strategy for classification\n")
trainControl_roc <- trainControl(method = "cv",
                                 number = 5,
                                 allowParallel = TRUE,
                                 summaryFunction = multiClassSummary,
                                 classProbs = TRUE,
                                 savePredictions = TRUE)

classification_models <- list()

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

cat("\nModified plot_roc_curves function to return a ggplot object\n")
plot_roc_curves_gg <- function(model, test_data, model_name) {
  pred_probs_class <- predict(model, newdata = test_data %>% select(-popularity_level), type = "prob")
  class_levels <- levels(test_data$popularity_level)
  roc_objects_class <- list()
  
  for (i in seq_along(class_levels)) {
    current_class <- class_levels[[i]]
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
    )
  })
  
  roc_data_df <- bind_rows(roc_data_list)
  
  auc_values <- sapply(roc_objects_class, auc)
  auc_labels <- data.frame(
    Class = names(auc_values),
    AUC = round(auc_values, 3)
  )
  
  roc_plot <- ggplot(roc_data_df, aes(x = FPR, y = TPR, color = Class)) +
    geom_line(size = 1) +
    geom_abline(linetype = "dashed", color = "gold") +
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

cat("\nGenerate ROC plots for both models\n")
roc_rf_plot <- plot_roc_curves_gg(classification_models$R.F, test_data_class, "Random Forest")
roc_xgb_plot <- plot_roc_curves_gg(classification_models$XGB, test_data_class, "XGBoost")

cat("\nArrange ROC curves side-by-side\n")
grid.arrange(roc_rf_plot, roc_xgb_plot, ncol = 2,
             top = textGrob("One-vs-Rest ROC Curves for Classification Models",
                            gp = gpar(col = "white", fontsize = 20, fontface = "bold")))

cat("\n==============================================================================\n")
cat("Frequency Plots for Popularity Class Labels\n")
cat("==============================================================================\n")

cat("\nActual popularity class distribution\n")
actual_class_plot <- ggplot(data.frame(Class = actual_popularity_level), aes(x = Class)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Frequency of Actual Popularity Classes", x = "Class", y = "Count") +
  theme_dark_custom()
print(actual_class_plot)

cat("\nPredicted popularity class distribution\n")
predicted_class_plot <- ggplot(data.frame(Class = predicted_popularity_level), aes(x = Class)) +
  geom_bar(fill = "darkorange") +
  labs(title = "Frequency of Predicted Popularity Classes", x = "Class", y = "Count") +
  theme_dark_custom()
print(predicted_class_plot)

cat("\n----------------------------------------------------------------------------------------------------------------------\n")
cat("Integrated Plot: Actual vs. Predicted Popularity Scores with Class Zones\n")
cat("----------------------------------------------------------------------------------------------------------------------\n")
plot_data_actual_vs_predicted_with_predicted_classes <- data.frame(
  Actual_Popularity_Score = actual_popularity_for_classification_test_num,
  Predicted_Popularity_Score = predicted_numerical_popularity_for_classification_test,
  Predicted_Popularity_Class = predicted_popularity_level_for_plot
)

quadrant_data <- expand.grid(
  actual_class_label = c("very_low", "low", "high", "very_high"),
  predicted_class_label = c("very_low", "low", "high", "very_high")
) %>%
  mutate(
    xmin = case_when(
      actual_class_label == "very_low" ~ 0,
      actual_class_label == "low" ~ very_low_threshold,
      actual_class_label == "high" ~ high_threshold,
      actual_class_label == "very_high" ~ very_high_threshold
    ),
    xmax = case_when(
      actual_class_label == "very_low" ~ very_low_threshold,
      actual_class_label == "low" ~ high_threshold,
      actual_class_label == "high" ~ very_high_threshold,
      actual_class_label == "very_high" ~ 100
    ),
    ymin = case_when(
      predicted_class_label == "very_low" ~ 0,
      predicted_class_label == "low" ~ very_low_threshold,
      predicted_class_label == "high" ~ high_threshold,
      predicted_class_label == "very_high" ~ very_high_threshold
    ),
    ymax = case_when(
      predicted_class_label == "very_low" ~ very_low_threshold,
      predicted_class_label == "low" ~ high_threshold,
      predicted_class_label == "high" ~ very_high_threshold,
      predicted_class_label == "very_high" ~ 100
    ),
    fill_color = "gray30"
  )

actual_vs_predicted_with_colored_zones_plot <- ggplot(plot_data_actual_vs_predicted_with_predicted_classes,
                                                      aes(x = Actual_Popularity_Score,
                                                          y = Predicted_Popularity_Score,
                                                          color = Predicted_Popularity_Class)) +
  
  geom_rect(data = quadrant_data, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = "gray30", alpha = 0.5, inherit.aes = FALSE) +
  
  geom_point(alpha = 0.7, size = 2.5) +
  geom_abline(intercept = 0, slope = 1, color = "gold", linetype = "dashed", size = 1) +
  
  geom_vline(xintercept = very_low_threshold, linetype = "dotted", color = "#FF7F00", size = 0.8) +
  geom_vline(xintercept = high_threshold, linetype = "dotted", color = "#377EB8", size = 0.8) +
  geom_vline(xintercept = very_high_threshold, linetype = "dotted", color = "#4DAF4A", size = 0.8) +
  
  geom_hline(yintercept = very_low_threshold, linetype = "dotted", color = "#FF7F00", size = 0.8) +
  geom_hline(yintercept = high_threshold, linetype = "dotted", color = "#377EB8", size = 0.8) +
  geom_hline(yintercept = very_high_threshold, linetype = "dotted", color = "#4DAF4A", size = 0.8) +
  
  annotate("text", x = 12.5, y = 98, label = "Actual:\nVery Low", color = "white", size = 3, hjust = 0.5, vjust = 1, fontface = "bold") +
  annotate("text", x = 37.5, y = 98, label = "Actual:\nLow", color = "white", size = 3, hjust = 0.5, vjust = 1, fontface = "bold") +
  annotate("text", x = 62.5, y = 98, label = "Actual:\nHigh", color = "white", size = 3, hjust = 0.5, vjust = 1, fontface = "bold") +
  annotate("text", x = 87.5, y = 98, label = "Actual:\nVery High", color = "white", size = 3, hjust = 0.5, vjust = 1, fontface = "bold") +
  
  annotate("text", x = 2, y = 12.5, label = "Predicted:\nVery Low", color = "white", size = 3, angle = 90, hjust = 0.5, vjust = 0, fontface = "bold") +
  annotate("text", x = 2, y = 37.5, label = "Predicted:\nLow", color = "white", size = 3, angle = 90, hjust = 0.5, vjust = 0, fontface = "bold") +
  annotate("text", x = 2, y = 62.5, label = "Predicted:\nHigh", color = "white", size = 3, angle = 90, hjust = 0.5, vjust = 0, fontface = "bold") +
  annotate("text", x = 2, y = 87.5, label = "Predicted:\nVery High", color = "white", size = 3, angle = 90, hjust = 0.5, vjust = 0, fontface = "bold") +
  
  labs(
    title = "Actual vs. Predicted Popularity Scores (Classification Test Set)",
    subtitle = "Points colored by Predicted Popularity Class with Class Boundaries and Zones",
    x = "Actual Popularity Score",
    y = "Predicted Popularity Score",
    color = "Predicted Class"
  ) +
  
  xlim(0, 100) +
  ylim(0, 100) +
  
  scale_color_manual(values = c("very_low" = "#e41a1c",
                                "low" = "#377eb8",
                                "high" = "#4daf4a",
                                "very_high" = "#984ea3")) +
  
  theme_dark_custom() +
  
  theme(legend.position = "bottom",
        plot.margin = unit(c(1, 1, 1, 1), "cm"))

print(actual_vs_predicted_with_colored_zones_plot)

cat("\n=============================================================================\n")
cat("7. CONFUSION MATRIX AND CLASSIFICATION METRICS\n")
cat("=============================================================================\n")

cat("\nConfusion Matrix and Classification Metrics\n")

cat("\nFunction to evaluate and print classification metrics\n")
evaluate_classification_model <- function(model, test_data, model_name) {
  predictions_class <- predict(model, newdata = test_data %>% select(-popularity_level))
  
  conf_matrix_class <- confusionMatrix(data = predictions_class, reference = test_data$popularity_level)
  
  cat(paste("\nConfusion Matrix (", model_name, " on Test Set):\n", sep = ""))
  print(conf_matrix_class)
  
  overall_metrics_class <- data.frame(conf_matrix_class$overall)
  cat("\nOverall Classification Metrics:\n")
  print(overall_metrics_class)
  
  class_metrics_class <- data.frame(conf_matrix_class$byClass)
  cat("\nClass-Specific Classification Metrics:\n")
  print(class_metrics_class)
  
  return(list(confusion_matrix = conf_matrix_class,
              overall_metrics = overall_metrics_class,
              class_metrics = class_metrics_class))
}

cat("\nggplot confusion matrices\n")
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

cat("\nEvaluate Random Forest and XGBoost\n")
rf_evaluation_results <- evaluate_classification_model(classification_models$R.F, test_data_class, "Random Forest")
xgb_evaluation_results <- evaluate_classification_model(classification_models$XGB, test_data_class, "XGBoost")

cat("\nGet confusion matrix plots\n")
rf_cm_plot <- plot_confusion_matrix_gg(rf_evaluation_results$confusion_matrix, "Random Forest")
xgb_cm_plot <- plot_confusion_matrix_gg(xgb_evaluation_results$confusion_matrix, "XGBoost")

cat("\nArrange confusion matrices side-by-side\n")
grid.arrange(rf_cm_plot, xgb_cm_plot, ncol = 2,
             top = textGrob("Confusion Matrices for Classification Models",
                            gp = gpar(col = "white", fontsize = 20, fontface = "bold")))

cat("\n==================================================================================\n")
cat("Average Model Accuracy Comparison\n")
cat("==================================================================================\n")

cat("\nExtract overall accuracy for Random Forest\n")
rf_accuracy <- rf_evaluation_results$overall_metrics["Accuracy", ]
cat(paste0("Random Forest Model Accuracy: ", round(rf_accuracy, 4), "\n"))

cat("\nExtract overall accuracy for XGBoost\n")
xgb_accuracy <- xgb_evaluation_results$overall_metrics["Accuracy", ]
cat(paste0("XGBoost Model Accuracy: ", round(xgb_accuracy, 4), "\n"))

cat("\nCompare and highlight the best model\n")
if (rf_accuracy > xgb_accuracy) {
  cat("\nRandom Forest shows higher overall accuracy.\n")
} else if (xgb_accuracy > rf_accuracy) {
  cat("\nXGBoost shows higher overall accuracy.\n")
} else {
  cat("\nBoth models have similar overall accuracy.\n")
}

cat("\nVisualize accuracy comparison with a bar plot\n")
accuracy_comparison_df <- data.frame(
  Model = c("Random Forest", "XGBoost"),
  Accuracy = c(rf_accuracy, xgb_accuracy)
)

accuracy_plot <- ggplot(accuracy_comparison_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5, color = "white") +
  labs(title = "Comparison of Average Model Accuracy",
       x = "Classification Model",
       y = "Accuracy") +
  theme_dark_custom() +
  scale_fill_manual(values = c("Random Forest" = "#66c2a5", "XGBoost" = "#fc8d62"))

print(accuracy_plot)

cat("\n===================================================================\n")
cat("\n8. PDF REPORT GENERATION (COMBINED WITH MULTIPLE REGRESSION MODELS)\n")
cat("=====================================================================\n")

# Stop PDF device
dev.off()
cat("All plots have been saved to jk-combined_reg&classy_spotify_analysis_results_multiple_reg_r-vised.pdf
      in your working directory.\n")