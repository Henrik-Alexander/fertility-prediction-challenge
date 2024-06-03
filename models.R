### Prediction models ----------------------------

# Remove the id column
data <- df[, 2:ncol(df)]

# Select test data
n <- nrow(data)
selector <- sample(1:n, size = n * 0.8, replace = F)
train <- data[selector, ]
test <- data[!(1:n %in% selector), ]

# Make a logistic regression -----------------
mod_log <- glm(new_child ~ ., data = train, family = "binomial")
test$pred_log <- ifelse(predict(mod_log, test, type = "response") > 0.5, 1, 0)
mean(test$pred_log == test$new_child, na.rm = T)

# Make an elastic net regression -------------
cv_5 = trainControl(method = "cv", number = 5)
data$new_child <- factor(data$new_child)
mod_elnet = train(
  new_child ~ ., data = data,
  method = "glmnet",
  trControl =  trainControl(method = "cv", number = 5)
)
test$pred_elnet <- predict(mod_elnet, test, type = "raw")
mean(test$pred_elnet == test$new_child, na.rm = T)

# Random forest without cross-validation -------------
mod_rf1 <- randomForest(y = factor(train$new_child), 
                    x = train[, !(names(train) %in% "new_child")], 
                    ytest = factor(test$new_child), 
                    yxtest = test, importance = T)
test$pred_forest <- predict(mod_rf1, test)
mean(test$pred_forest == test$new_child)

# Random forest with cross-validation ------------------
mod_rrf = train(new_child ~ ., data = data, method = "RRFglobal", trControl =  trainControl(method = "cv", number = 5), importance = T)
test$pred_forest_cv <- predict(mod_rrf, test, type = "raw")
mean(test$pred_forest_cv == test$new_child)

# Save the best model
saveRDS(mod_rrf, file = "./model.rds")

## Support Vector Machines -----------------------------
svm_grid <- expand.grid("cost" = seq(0.01, 0.1, length.out = 3),
                          "weight" = 0,
                          "Loss" = seq(0.01, 0.1, length.out = 3))
mod_svm_cv = train(new_child ~ ., data = data, method = "svmLinearWeights2", trControl =  trainControl(method = "cv", number = 5),
                   tuneGrid = svm_grid)
test$pred_svm <- predict(mod_svm_cv, test, type = "raw")
mean(test$pred_forest_cv == test$new_child)

## Generalized boosted regression trees ----------------
mod_gbm = train(new_child ~ ., data = train, method = "gbm",
                  trControl =  trainControl(method = "cv", number = 5),
                  verbose = FALSE)
test$pred_gbm <- predict(mod_gbm, test, type = "raw")
mean(test$pred_gbm == test$new_child)

## XGboost -----------------------------------------
xgboost_params <- expand.grid("nrounds" = c(1, 10), 
                              "max_depth" = 1:5, 
                              "gamma" = 0, 
                              "subsample" = 1,
                              "eta" = 1:5, 
                             # "objective" = "binary:logistic", 
                              "colsample_bytree" = c(0.7, 1),
                              "min_child_weight" = 1)

mod_xgboost = train(new_child ~ ., data = data, method = "xgbTree",
                  trControl =  trainControl(method = "cv", number = 5), tuneGrid = xgboost_params)
test$pred_xgboost <- predict(mod_xgboost, test, type = "raw")
mean(test$pred_xgboost == test$new_child)

## Neural network -----------------------------------------
nn_parameters <- expand.grid("size" = 5:10,
                             "decay" = seq(from = 0.1, to = 0.5, by = 0.1),
                             "bag" = F)

mod_nn = train(y = data$new_child, x = subset(data, select = names(data)[names(data) != "new_child"]),
                    method = "avNNet", trControl =  trainControl(method = "cv", number = 5), tuneGrid = nn_parameters)
test$pred_nn <- predict(mod_nn, test, type = "raw")
mean(test$pred_n == test$new_child)
