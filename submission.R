# This is an example script to generate the outcome variable given the input dataset.
# 
# This script should be modified to prepare your own submission that predicts 
# the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.
# 
# The predict_outcomes function takes a data frame. The return value must
# be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
# should contain the nomem_encr column from the input data frame. The outcome
# column should contain the predicted outcome for each nomem_encr. The outcome
# should be 0 (no child) or 1 (having a child).
# 
# clean_df should be used to clean (preprocess) the data.
# 
# run.R can be used to test your submission.

# List your packages here. Don't forget to update packages.R!
library(tidyverse) # as an example, not used here
library(data.table)
library(randomForest)
library(caret)
library(glmnet)

# Load the data
path_data <- "U:/data/nld/liss/"
df <- fread(paste0(path_data, "training_data/PreFer_train_data.csv"))
out <- fread(paste0(path_data, "training_data/PreFer_train_outcome.csv"))
cb <- fread(paste0(path_data, "/codebooks/PreFer_codebook.csv"))

# Filter respondents that have the outcome
df <- df[outcome_available == 1, ]
df <- merge(df, out, by = "nomem_encr", all.x = T) 

clean_df <- function(df, background_df = NULL){
  # Preprocess the input dataframe to feed the model.
  ### If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command
  
  # Parameters:
  # df (dataframe): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
  # background (dataframe): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).
  
  
  # Returns:
  # data frame: The cleaned dataframe with only the necessary columns and processed variables.
  
  ## This script contains a bare minimum working example
  # Create new age variable
  df$age <- 2024 - df$birthyear_bg
  df$partnership_duration <- 2024 - df$cf20m028
  
  # Function to select the most recent income information
  select_most_recent <- function(data = df, varname = "^brutohh_f_"){
    if (!any(str_detect(names(data), varname))) stop("Could not find a variable!")
    # Function select last non-missing value
    select_last <- function (x) {
      x <- rev(x); res <- NA; count <- 1
      while(is.na(res) & count < length(x)) {
        res <- x[count]
        count <- count + 1
      }
      return(res)
    }
    selecter <-  names(data)[str_detect(names(data), varname)]
    tmp <- data[, ..selecter]
    res <- apply(tmp, 1, select_last)
    return(res)
  }
  
  # Clean the variables
  df$income <- select_most_recent(data = df, varname = "^brutohh_f")
  df$education <- select_most_recent(data = df, varname = "oplmet")
  
  # Selecting variables for modelling
  newvars <- c("partnership", "fertility_intentions", "cohabitation", "marriage", "nchild", "parent")
  oldvars <- c("cf20m024", "cf20m129", "cf20m025", "cf20m030", "cf14g036", "cf14g035")
  
  # Rename variables
  df <- setnames(df, old = oldvars, new = newvars)
  
  keepcols <- c('nomem_encr', # ID variable required for predictions,
                'age',  # Age of the respondent
                "new_child", # Childbirth
                "partnership_duration", # Duration of the ongoing partnership
                "education", # Income data
                newvars)        # All the other variables
  
  keepcols <- keepcols[keepcols != "fertility_intentions"]
  
  ## Keeping data with variables selected
  df <- df[, ..keepcols]
  
  # Make it to a classic data frame
  df <- as.data.frame(df)
  
  # Create partnership duration imputed
  df$partnership_duration[df$partnership == 1 & is.na(df$partnership_duration)] <- mean(df$partnership_duration, na.rm = T)
  
  # Additional cleaning
  df$partnership_duration[df$partnership == 2 | is.na(df$partnership)] <- 0
  df$nchild[is.na(df$parent)] <- 0
  df$parent[df$nchild == 0 & is.na(df$parent)] <- 0
  df$cohabitation[df$partnership == 2] <- 0
  df$marriage[df$partnership == 2] <- 0
  
  # Remove all the missing variables
  df <- na.omit(df)
  
  return(df)
}

# Predict the outcomes
df <- clean_df(df)

# Select test data
n <- nrow(df)
selector <- sample(1:n, size = n * 0.8, replace = F)
train <- df[selector, ]
test <- df[!(1:n %in% selector), ]

# Make a logistic regression
log1 <- glm(new_child ~ ., data = train, family = "binomial")
test$pred_log <- ifelse(predict(log1, test, type = "response") > 0.5, 1, 0)
mean(test$pred_log == test$new_child, na.rm = T)

# Make an elastic net regression
cv_5 = trainControl(method = "cv", number = 5)
train$new_child <- factor(train$new_child)
hit_elnet = train(
  new_child ~ ., data = train,
  method = "glmnet",
  trControl = cv_5
)
test$pred_elnet <- predict(hit_elnet, test)
mean(test$pred_elnet == test$new_child, na.rm = T)


# Random forest
rf1 <- randomForest(y = factor(train$new_child), x = train[, !(names(train) %in% "new_child")], ytest = factor(test$new_child), yxtest = test)
test$pred_forest <- predict(rf1, test)
mean(test$pred_forest == test$new_child)


predict_outcomes <- function(df, background_df = NULL, model_path = "./model.rds"){
  # Generate predictions using the saved model and the input dataframe.
  
  # The predict_outcomes function accepts a dataframe as an argument
  # and returns a new dataframe with two columns: nomem_encr and
  # prediction. The nomem_encr column in the new dataframe replicates the
  # corresponding column from the input dataframe The prediction
  # column contains predictions for each corresponding nomem_encr. Each
  # prediction is represented as a binary value: '0' indicates that the
  # individual did not have a child during 2021-2023, while '1' implies that
  # they did.
  
  # Parameters:
  # df (dataframe): The data dataframe for which predictions are to be made.
  # background_df (dataframe): The background data dataframe for which predictions are to be made.
  # model_path (str): The path to the saved model file (which is the output of training.R).
  
  # Returns:
  # dataframe: A dataframe containing the identifiers and their corresponding predictions.
  
  ## This script contains a bare minimum working example
  if( !("nomem_encr" %in% colnames(df)) ) {
    warning("The identifier variable 'nomem_encr' should be in the dataset")
  }
  
  # Load the model
  model <- readRDS(model_path)
  
  # Preprocess the fake / holdout data
  df <- clean_df(df, background_df)
  
  # Exclude the variable nomem_encr if this variable is NOT in your model
  vars_without_id <- colnames(df)[colnames(df) != "nomem_encr"]
  
  # Generate predictions from model
  predictions <- predict(model, 
                         subset(df, select = vars_without_id), 
                         type = "response") 
  
  # Create predictions that should be 0s and 1s rather than, e.g., probabilities
  predictions <- ifelse(predictions > 0.5, 1, 0)  
  
  # Output file should be data.frame with two columns, nomem_encr and predictions
  df_predict <- data.frame("nomem_encr" = df[ , "nomem_encr" ], "prediction" = predictions)
  # Force columnnames (overrides names that may be given by `predict`)
  names(df_predict) <- c("nomem_encr", "prediction") 
  
  # Return only dataset with predictions and identifier
  return( df_predict )
}

