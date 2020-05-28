

## from Using XGBoost with Tidymodels by Tychobra
## published in Rbloggers/ May 18, 2020


## library and data

# data

library(AmesHousing)

# data cleaning

library(janitor)

# data preparation

library(dplyr)

# tidymodels

library(rsample)
library(recipes)
library(parsnip)
library(tune)
library(dials)
library(workflows)
library(yardstick)


## speed up computation with parallel processing


library(doParallel)
all_cores <- parallel::detectCores(logical=FALSE)
registerDoParallel(cores=all_cores)


## loading data set


ames_data <- make_ames() %>%
  janitor::clean_names()           # however the stage of the column title, clean_names()
                                   # puts every word separately and conntects the title words with '_' and it is just small letter words
                                   # e.g. 'Lot_Frontage' gets 'lot_frontage' or 'MS_SubClass' gets 'ms_sub_class'

head(ames_data)
str(ames_data)


## EDA (Explanatory Data Analysis)

                             # extremely important in real life at this stage
                             # but skiped here for simplicity reasons



## Step 1: Initial Data Split

                             # split into training and testing data sets
                             # stratify by sale_price

ames_split <- rsample::initial_split(ames_data, prop=0.2, strata=sale_price)




## Step 2: Pre-Processing

                             # preprocessing alters the data to make the model more predictive
                             # and training process less compute intensive
                             # many models require careful and extensive variable processing
                             # to produce accurate predictions

                             # however, XGBoost is robust against highly skewed and/ or correlated data
                             # so required preprocessing with XGBoost is minimal

                             # recipe-package resulting in a 'recipe'

preprocessing_recipe <- 
  recipes::recipe(sale_price~., data=training(ames_split)) %>%
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  # combine low frequency factor levels
  recipes::step_other(all_nominal(), threshold=0.01) %>%
  # remove no variance predictors which provide no predictive information
  recipes::step_nzv(all_nominal()) %>%
  prep()

                             # all in variable neighborhoods with less than 1 % of total observations
                             # were defined low frequency factor levels
                             # and put togehter in 'other'
                             # see graph in tutorial



## Step 3: Splitting for Cross Validation

                             # applying the previously defined preprocessing recipe with bake()
                             # then we use cross-validation to randomly split the training data
                             # into further training and validation sets
                             # and we'll use these additional cross validation folds to tune our
                             # hyperparameters in a later step

ames_cv_folds <- 
  recipes::bake(
    preprocessing_recipe,
    new_data=training(ames_split)
  ) %>%
  rsample::vfold_cv(v=5)


                             # btw the quoting of 'package'::something
                             # not really necessary as we loaded the libraries
                             # but done here for better understanding which package where doing what...



## Step 4: XGBoost Model Specification

                             # using the parsnip-package
                             # to define the XGBoost model specification
                             # here we use
                                  # boost_tree() along with
                                  # tune()
                             # to define the hyperparameters
                             # to undergo tuning in a subsequent step

xgboost_model <- 
  parsnip::boost_tree(       # overview of usable models here: https://www.tidymodels.org/find/parsnip/ , rest is under ?boost_tree
    mode="regression",
    trees=1000,
    min_n=tune(),
    tree_depth=tune(),
    learn_rate=tune(),
    loss_reduction=tune()
  ) %>%
  set_engine("xgboost", objective = "reg:squarederror")


                             ## ATTENTION - I got something ##
                             # with recipe, prep and bake I handle the data <-- recipes-package
                             # with parsnip package I handle, i.e. I tune and specify, the model!!!
                             # whicht includes set_engine()


## Step 5: Grid Specification

                             # use the dials-package from tidymodels


xgboost_params <- dials::parameters(min_n(),         # simply what was defined with tune() above
                                    tree_depth(),
                                    learn_rate(),
                                    loss_reduction()
                                    )


                             # now the grid space is set up
                             # with dials::grid_*
                             # this function supports several methods to define the grid space
                             # here, dials::grid_max_entropy() is used
                             # which covers the hyperparameter space such that
                             # any portion of the space has an observed combination that is not too far from it

xgboost_grid <- 
  dials::grid_max_entropy(
    xgboost_params,
    size = 60
  )


knitr::kable(head(xgboost_grid),format="markdown")

            
                             # to tune the model, we perform grid search
                             # over the xgboost_grid's grid space
                             # to identify the hyperparameter values
                             # that have the lowest prediction error


## Step 6: Define the Workflow

xgboost_wf <- 
  workflows::workflow() %>%
  add_model(xgboost_model) %>%
  add_formula(sale_price ~ .)


## Step 7: Tune the Model

                             # HERE the tidymodels ecosystem COMES TOGETHER

xgboost_tuned <- tune::tune_grid(
  object = xgboost_wf,
  resamples = ames_cv_folds,
  grid = xgboost_grid,
  metrics = yardstick::metric_set(rmse,rsq,mae),
  control=tune::control_grid(verbose = TRUE)
)

                             # tune_grid() performs grid search over all the 
                             # 60 grid parameter combinations
                             # defined in xgboost_grid

                             # use 5 fold cross validation
                             # along with
                                  # rmse (Root Mean Squared Error)
                                  # rsq (R Squared)
                                  # mae (Mean Absolute Error)

                             # to measure prediction accuracy

                             # so the tidymodel tuning just fit
                             # 60 x 5 = 300 XGBoost models each with
                             # 1000 trees
                             # in search of th optimal hyperparameters


xgboost_tuned %>%
  tune::show_best(metric="rmse") %>%
  knitr::kable(format="markdown")



                             # now isolating the best performing hyperparameter values


xgboost_best_params <- xgboost_tuned %>%
  tune::select_best("rmse")

knitr::kable(xgboost_best_params,format="markdown")


                             # finalizing the XGBoost model
                             # to use the best tuning parameters

xgboost_model_final <- xgboost_model %>%
  finalize_model(xgboost_best_params)



## Step 8: Evaluate Performance on Test Data

                             # we use
                                  # rmse
                                  # rsq
                                  # mae
                             # from the yardstick package in our model evaluation


              # first, evaluating the metrics on the training data


train_processed <- bake(preprocessing_recipe,
                        new_data = training(ames_split))

train_prediction <- xgboost_model_final %>%
  # fit the model on all the training data 
  fit(formula = sale_price ~ .,
      data = train_processed) %>%
  # predict the sale prices for the training data
  predict(new_data = train_processed) %>%
  bind_cols(training(ames_split))

xgboost_score_train <- 
  train_prediction %>%
  yardstick::metrics(sale_price, .pred) %>%
  mutate(.estimate = format(round(.estimate,2),
                            big.mark = ","))

knitr::kable(xgboost_score_train, format="markdown")




              # and now for the test data

test_processed  <- bake(preprocessing_recipe, new_data=testing(ames_split))

test_prediction <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(formula= sale_price ~ ., data=train_processed) %>%
  # use the training model fit to predict the test data
  predict(new_data=test_processed) %>%
  bind_cols(testing(ames_split))

  # measure the accuracy of the model using 'yardstick'
xgboost_score <- 
  test_prediction %>%
  yardstick::metrics(sale_price, .pred) %>%
  mutate(.estimate = format(round(.estimate,2),
                            big.mark=","))

knitr::kable(xgboost_score,format="markdown")

                             # metrics on the test data
                             # are significantly worse than the training data metrics
                             # hence, we know that there is some overfitting going on in the model
                             

                             # to quickly check that there is not an obvious issue
                             # with the model's predictions,
                             # plotting the test data residuals


house_prediction_residual <- test_prediction %>%
  arrange(.pred) %>%
  mutate(residual_pct = (sale_price - .pred) / .pred) %>%
  dplyr::select(.pred,residual_pct)

ggplot(house_prediction_residual, aes(x=.pred,y=residual_pct)) +
  geom_point() +
  xlab("Predicted Sale Price") +
  ylab("Residual (%)") +
  scale_x_continuous(labels=scales::dollar_format()) +
  scale_y_continuous(labels=scales::percent)

                             # chart shows no obvious trends in the residuals
                             # which indicates
                             # that, at a very high level, our model is not systematically making inaccurate predictions
                             # for houses with certain predicted sale prices

                             # hence we would have to do more model validation here
                             # which is not further done in the tutorial

















































































































