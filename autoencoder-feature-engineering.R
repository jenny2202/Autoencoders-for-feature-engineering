
# Feature Engineering can simultaneously target both: 
## (1) reduce the number of input variables that are needed to build the model and consequently reduce the computational time, 
## (2) improve the predictive quality of the model or ensure that it is negligible while the number of variables used decreases radically



#=================================
#  Stage 1: Data Pre-processing
#=================================

# Clear workspace: 
rm(list = ls())

# Load packages and data: 
library(tidyverse)
library(magrittr)
library(caret)
data("GermanCredit")

# Split data: 
set.seed(1)
id <- createDataPartition(y = GermanCredit$Class, p = 0.7, list = FALSE)
df_train <- GermanCredit[id, ] # For training
df_test <- GermanCredit[-id, ] # For testing


# Activate h2o package for deep learning: 
library(h2o)
h2o.init(nthreads = -1, max_mem_size = "16g") ##?

h2o.no_progress()

# Convert to h20 frame

train <- as.h2o(df_train)
test <- as.h2o(df_test)


# Indentity input and output

y <- "Class"
x <- setdiff(names(train), y) ## ???

# Train default random forest

pure_nn <- h2o.randomForest(x = x, y = y,
                            training_frame = train,
                            nfolds = 10,
                            stopping_rounds = 5,
                            stopping_metric = "AUC",
                            seed = 30)




# Collecting result from cross-validation:

results_df <- function(h2o_model) {
  h2o_model@model$cross_validation_metrics_summary %>% 
    as.data.frame() %>% 
    select(-mean, -sd) %>% 
    t() %>% 
    as.data.frame() %>% 
    mutate_all(as.character) %>% 
    mutate_all(as.numeric) %>% 
    select(Accuracy = accuracy, 
           AUC = auc, 
           Precision = precision, 
           Specificity = specificity, 
           Recall = recall, 
           Logloss = logloss) %>% 
    return()
}



# Function presents results by graph: 

visual_results <- function(df_results) {
  df_results %>% 
    gather(Metrics, Values) %>% 
    ggplot(aes(Metrics, Values, fill = Metrics, color = Metrics)) +
    geom_boxplot(alpha = 0.3, show.legend = FALSE) + 
    facet_wrap(~ Metrics, scales = "free") + 
    scale_y_continuous(labels = scales::percent) + 
    theme_minimal() + 
    labs(x = NULL, y = NULL, 
         title = "Model Performance Based on Cross Validation")
}


pure_nn %>% results_df() %>% summary()


pure_nn %>% results_df() %>% visual_results()


#=================================
#  Stage 2: Autoencoder as a Feature Engineering Technique
#=================================



# Buil a autoencoder: 

autoencoder <- h2o.deeplearning(x = x,
                                training_frame = train, 
                                autoencoder = TRUE, 
                                seed = 29, 
                                hidden = c(10, 20, 61), 
                                epochs = 30, 
                                activation = "Tanh")


#============================================================
#  Use Autoencoder as Feature Engineering Method (Version 1)
#============================================================

#Layer 1
train_autoen <- h2o.predict(autoencoder, train) %>% 
  as.data.frame() %>% 
  mutate(Class = df_train$Class) %>% 
  as.h2o()

test_autoen <- h2o.predict(autoencoder, test) %>% 
  as.data.frame() %>% 
  mutate(Class = df_test$Class) %>% 
  as.h2o()

nn_autoen_layers1 <- h2o.randomForest(x = setdiff(colnames(train_autoen), "Class"), 
                                      y = y, 
                                      training_frame = train_autoen,
                                      nfolds = 10, 
                                      stopping_rounds = 5, 
                                      stopping_metric = "AUC", 
                                      seed = 29)


#Layer2

train_features_l2 <- h2o.deepfeatures(autoencoder, train, layer = 2) %>%
  as.data.frame() %>%
  mutate(Class = df_train$Class) %>% 
  as.h2o()


test_features_l2 <- h2o.deepfeatures(autoencoder, test, layer = 2) %>%
  as.data.frame() %>%
  mutate(Class = df_test$Class) %>% 
  as.h2o()


nn_autoen_layers2 <- h2o.randomForest(x = setdiff(colnames(train_features_l2), "Class"), 
                                      y = y, 
                                      training_frame = train_features_l2,
                                      nfolds = 10, 
                                      stopping_rounds = 5, 
                                      stopping_metric = "AUC", 
                                      seed = 29)

#Layer3

train_features_l3 <- h2o.deepfeatures(autoencoder, train, layer = 3) %>%
  as.data.frame() %>%
  mutate(Class = df_train$Class) %>% 
  as.h2o()


test_features_l3 <- h2o.deepfeatures(autoencoder, test, layer = 3) %>%
  as.data.frame() %>%
  mutate(Class = df_test$Class) %>% 
  as.h2o()


nn_autoen_layers3 <- h2o.randomForest(x = setdiff(colnames(train_features_l3), "Class"), 
                                      y = y, 
                                      training_frame = train_features_l3,
                                      nfolds = 10, 
                                      stopping_rounds = 5, 
                                      stopping_metric = "AUC", 
                                      seed = 29)



#==========================
#  Compare between models
#==========================

do.call("bind_rows", 
        lapply(list(pure_nn,
                    nn_autoen_layers1,
                    nn_autoen_layers2,
                    nn_autoen_layers3), results_df)) -> df_compared


df_compared %<>% 
  mutate(Model = rep(c("Original", "Layer1", "20Var", "61Var"), each = 10, time = 1))


theme_set(theme_minimal())
df_compared %>% 
  gather(a, b, -Model) %>% 
  ggplot(aes(Model, b, fill = Model, color = Model)) + 
  geom_boxplot(alpha = 0.3) + 
  scale_y_continuous(labels = scales::percent) + 
  facet_wrap(~ a, scales = "free") + 
  labs(x = NULL, y = NULL, title = "Model Performance")
