# Load necessary libraries
library(caTools)    # For data splitting
library(ggplot2)    # For visualization
library(MASS)       # for stepAIC
library(dplyr)      # For data manipulation
library(caret)      # For confusion matrix and cross-validation
library(class)      # For KNN
library(pROC)       # For ROC analysis
library(e1071)      # For tuning the KNN model


# Loading the data file
file_path <- "D:\\UMKC_Assignments\\Statistical Learning\\DataLab\\synthetic_wbc_datasets-1\\synthetic_data_42.csv"
data <- read.csv(file_path)

# Data Preprocessing

str(data)
data$diagnosis <- as.factor(data$diagnosis)

# Plot a few features
ggplot(data, aes(x = diagnosis, y = radius_mean, fill = diagnosis)) +
  geom_boxplot() + ggtitle("Radius Mean by Diagnosis")
ggplot(data, aes(x = diagnosis, y = texture_mean, fill = diagnosis)) +
  geom_boxplot() + ggtitle("Texture Mean by Diagnosis")


# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(data), 0.7 * nrow(data))  # 70% training data
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


# Task 1 Logistic Regression
# Fit the full logistic regression model with all predictors
full_model <- glm(diagnosis ~ radius_mean + texture_mean + perimeter_mean +
                    area_mean + smoothness_mean + compactness_mean + 
                    concavity_mean + concave.points_mean + symmetry_mean + 
                    fractal_dimension_mean, 
                  family = "binomial", data = train_data)

# Perform backward elimination using stepwise selection
backward_model <- step(full_model, direction = "backward")

# Summary of the final model after variable selection
summary(backward_model)

# Predict on the test set
predicted_probabilities <- predict(backward_model, newdata = test_data, type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, "M", "B")  # Threshold at 0.5

#print(prediced)
print(predicted_classes)

# Create a confusion matrix to evaluate the model
confusion_matrix <- table(test_data$diagnosis, predicted_classes)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Create a plot for the logistic regression model
ggplot(test_data, aes(x = radius_mean, y = predicted_probabilities)) +
  geom_point(aes(color = diagnosis)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE) +
  labs(title = "Logistic Regression Model", x = "Radius Mean", y = "Predicted Probability") +
  theme_minimal()

#  ------------------------------ Task 2 --------------------------------------
#  --------------------- LINEAR DISCRIMINATE ANALYSIS --------------------------

# Data preparation
train_data <- train_data %>%
  select(diagnosis, radius_mean, texture_mean, perimeter_mean, 
         area_mean, smoothness_mean, compactness_mean, 
         concavity_mean, concave.points_mean, 
         symmetry_mean, fractal_dimension_mean) %>%
  mutate(diagnosis = as.factor(diagnosis))  # Ensure diagnosis is a factor

# Train the LDA model
lda_model <- lda(diagnosis ~ ., data = train_data)

# Summary of the LDA model
print(lda_model)

# Predict on test data
lda_preds <- predict(lda_model, newdata = test_data)

# Confusion Matrix to evaluate LDA model performance
confusion_mat <- confusionMatrix(lda_preds$class, test_data$diagnosis)
print(confusion_mat)

# Plot the LDA results (if applicable)
plot(lda_model)

# If pairs plot is desired, you can visualize the predictions on original data
# Create a data frame of predictions and actual values
pred_results <- data.frame(Actual = test_data$diagnosis, Predicted = lda_preds$class)

# Visualization
ggplot(pred_results, aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "dodge") +
  labs(title = "Actual vs Predicted Classes", x = "Actual Diagnosis", y = "Count") +
  theme_minimal()


# ------------------------------ TASK 3 ---------------------------------------
# ------------------------------ K - NN ---------------------------------------

# Vectorized normalization function for numeric columns only
normalize <- function(df) {
  numeric_cols <- sapply(df, is.numeric)  # Identify numeric columns
  df[numeric_cols] <- lapply(df[numeric_cols], function(x) {
    (x - min(x)) / (max(x) - min(x))
  })
  return(df)
}

# Normalize and prepare training data
train_knn <- train_data %>%
  select(diagnosis, radius_mean, texture_mean, perimeter_mean, 
         area_mean, smoothness_mean, compactness_mean, 
         concavity_mean, concave.points_mean, 
         symmetry_mean, fractal_dimension_mean) %>%
  mutate(diagnosis = as.factor(diagnosis))  # Ensure diagnosis is a factor

# Normalize numeric columns except diagnosis
train_knn[, -1] <- normalize(train_knn[, -1])

# Normalize and prepare test data
test_knn <- test_data %>%
  select(diagnosis, radius_mean, texture_mean, perimeter_mean, 
         area_mean, smoothness_mean, compactness_mean, 
         concavity_mean, concave.points_mean, 
         symmetry_mean, fractal_dimension_mean) %>%
  mutate(diagnosis = as.factor(diagnosis))  # Ensure diagnosis is a factor

# Normalize numeric columns except diagnosis
test_knn[, -1] <- normalize(test_knn[, -1])

# Check the levels of the diagnosis factor
print(levels(train_knn$diagnosis))
print(levels(test_knn$diagnosis))

# Hyperparameter tuning for optimal k
set.seed(42)
tune_grid <- expand.grid(k = seq(1, 15, by = 2))  # Test odd values of k
knn_model <- train(diagnosis ~ ., data = train_knn, 
                   method = "knn", 
                   trControl = trainControl(method = "cv", number = 10),  # 10-fold cross-validation
                   tuneGrid = tune_grid)

# Make predictions on the test set
knn_pred <- predict(knn_model, newdata = test_knn)

# Confusion Matrix for KNN model performance
confusion_mat_knn <- confusionMatrix(knn_pred, test_knn$diagnosis)
print(confusion_mat_knn)

# Visualization for KNN results
pred_results_knn <- data.frame(Actual = test_knn$diagnosis, Predicted = knn_pred)
ggplot(pred_results_knn, aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "dodge") +
  labs(title = "Actual vs Predicted Classes (KNN)", x = "Actual Diagnosis", y = "Count") +
  theme_minimal()

# Plotting the KNN model performance across different k values
plot(knn_model)

# Plot the KNN model performance across different k values
plot(knn_model)

# -----------------------------------------------------------------------------

grid_radius <- seq(min(train_knn$radius_mean), max(train_knn$radius_mean), length.out = 100)
grid_texture <- seq(min(train_knn$texture_mean), max(train_knn$texture_mean), length.out = 100)

# Expand grid to cover all combinations of radius_mean and texture_mean
grid <- expand.grid(radius_mean = grid_radius, texture_mean = grid_texture)

# For simplicity, only use two predictors (radius_mean and texture_mean) for this plot
# Repeat for the remaining features (keeping them constant or averaged)
grid$perimeter_mean <- mean(train_knn$perimeter_mean)
grid$area_mean <- mean(train_knn$area_mean)
grid$smoothness_mean <- mean(train_knn$smoothness_mean)
grid$compactness_mean <- mean(train_knn$compactness_mean)
grid$concavity_mean <- mean(train_knn$concavity_mean)
grid$concave.points_mean <- mean(train_knn$concave.points_mean)
grid$symmetry_mean <- mean(train_knn$symmetry_mean)
grid$fractal_dimension_mean <- mean(train_knn$fractal_dimension_mean)

# Predict classes for each point in the grid
grid_predictions <- predict(knn_model, newdata = grid)

# Combine the grid and predictions for plotting
grid$diagnosis <- grid_predictions

# Plot the decision boundaries and clusters
ggplot() +
  geom_point(data = grid, aes(x = radius_mean, y = texture_mean, color = diagnosis), alpha = 0.2) +  # Decision boundaries
  geom_point(data = train_knn, aes(x = radius_mean, y = texture_mean, color = diagnosis), size = 3) +  # Training points
  labs(title = "K-NN Clusters and Decision Boundaries", x = "Radius Mean", y = "Texture Mean") +
  scale_color_manual(values = c("B" = "blue", "M" = "red")) +  # Customize colors for classes
  theme_minimal()

