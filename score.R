# predictions_df = data.frame with two columns; nomem_encr and predictions
# ground_truth_df = data.frame with two columns; nomem_encr and new_child

score <- function(predictions_df, ground_truth_df) {
  
  merged_df = merge(predictions_df, ground_truth_df, by = "nomem_encr",
                    all.y = TRUE) # right join
  
  # Calculate accuracy
  accuracy = sum( merged_df$prediction == merged_df$new_child) / length(merged_df$new_child) 
  
  # Calculate true positives, false positives, and false negatives
  true_positives = sum( merged_df$prediction == 1 & merged_df$new_child == 1 )
  
  false_positives = sum( merged_df$prediction == 1 & merged_df$new_child == 0 )
  
  false_negatives = sum( merged_df$prediction == 0 & merged_df$new_child == 1 )
  
  # Calculate precision, recall, and F1 score
  if( (true_positives + false_positives) == 0) {
    precision = 0
  } else {
    precision = true_positives / (true_positives + false_positives)
  }
  if( (true_positives + false_negatives) == 0) {
    recall = 0
  } else {
    recall = true_positives / (true_positives + false_negatives)
  }
  if( (precision + recall) == 0) {
    f1_score = 0
  } else {
    f1_score = 2 * (precision * recall) / (precision + recall)
  }
  
  metrics_df <- data.frame(
    'accuracy' = accuracy,
    'precision' = precision,
    'recall' = recall,
    'f1_score' = f1_score
  )
  
  return(metrics_df)
  
}