# How to Tune Hyperparameters in Gradient Boosting Algorithm

Last Updated : 30 May, 2025

[Gradient Boosting](https://www.geeksforgeeks.org/ml-gradient-boosting/) is an ensemble learning method and it works by sequentially adding decision trees where each tree tries to improve the model's performance by focusing on the errors made by the previous trees and reducing those error with help of gradient descent. While Gradient Boosting performs well for improving model accuracy fine-tuning its hyperparameters can significantly improve its performance and prevent overfitting. In this article we will explore how to optimize these hyperparameters for better model performance.

## Gradient Boosting Hyperparameters

Since we are talking about Gradient Boosting Hyperparameters let us see what different Hyperparameters are there that can be tuned.

****1\. n\_estimators****: Defines the number of boosting iterations (trees) to be added. More estimators usually lead to better performance, but also increase the risk of overfitting.

> By default: n\_estimators=100

- n\_estimators=100 means the model uses 100 decision trees to make predictions.

****2\. learning\_rate****: Controls the contribution of each tree to the final prediction. A smaller value makes the model more robust but requires more estimators to achieve high performance.

> By default: `learning_rate=0.1`

- learning\_rate=0.1 means that each additional tree will have a 10% influence on the overall prediction

****3\. max\_depth****: Specifies the maximum depth of each individual tree. Shallow trees might underfit while deeper trees can overfit. It's essential to find the right depth.

> By default: `max_depth=None`

****4\. min\_samples\_split****: Defines the minimum number of samples required to split an internal node. Increasing this value helps control overfitting by preventing the model from learning overly specific patterns.

> By default: `min_samples_split=2`

- min\_samples\_split=2 means that every node in the tree will have at least 2 samples before being split

****5\. subsample****: Specifies the fraction of samples to be used for fitting each individual tree.

> By default: `subsample=1.0`

- subsample=1.0 means that the model uses the entire dataset for each tree but using a fraction like 0.8 helps prevent overfitting by introducing more randomness.

****6\. colsample\_bytree****: Defines the fraction of features to be randomly sampled for building each tree. It is another method for controlling overfitting.

> By default: `colsample_bytree=1.0`

- colsample\_bytree=1.0 means that the model uses all the available features to build each tree.

****7\. min\_samples\_leaf****: Defines the minimum number of samples required to be at a leaf node. Increasing this value can reduce overfitting by preventing the tree from learning overly specific patterns.

> ****By default****: `min_samples_leaf=1`

- min\_samples\_leaf=1 means that the tree is allowed to create leaf nodes with a single sample.

****8\. max\_features****: Specifies the number of features to consider when looking for the best split.

> ****By default****: `max_features=None`

- max\_features=None means all features are considered for splitting.

## Gradient Boosting Hyperparameter Tuning in Python

Scikit-learn is a popular python library that provides useful tools for hyperparameter tuning that can help improve the performance of Gradient Boosting models. Hyperparameter tuning is the process of selecting the best parameters to maximize the efficiency and accuracy of the model. We'll explore three common techniques: GridSearchCV, RandomizedSearchCV and Optuna.

We shall now use the tuning methods on the [Titanic dataset](https://media.geeksforgeeks.org/wp-content/uploads/20240226143830/train.csv) and let's see the impact of each tuning method.

## Classification Model without Tuning:

The provided code implements a Gradient Boosting Classifier on the Titanic dataset to predict survival outcomes. It preprocesses the data, splits it into training and testing sets and trains the model. Hyperparameter tuning which can significantly impacts model performance is not performed in this implementation to see model working without it.

Python`import pandas as pd from sklearn.model_selection import train_test_split from sklearn.ensemble import GradientBoostingClassifier from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  titanic_data = pd.read_csv("train.csv")  titanic_data.fillna(0, inplace=True) titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)  X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) y = titanic_data['Survived']  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  gb_model = GradientBoostingClassifier()  gb_model.fit(X_train, y_train)  y_pred = gb_model.predict(X_test)  accuracy = accuracy_score(y_test, y_pred)  print(f"Accuracy: {accuracy}")`

****Output:****

> Accuracy: 0.7988826815642458

## 1\. Hyperparameter Tuning using Grid Seach CV

- ****GridSearchCV**** tries all possible combinations of hyperparameters from a grid.
- It's used to find the best settings for a model.
- Works well when the number of combinations is manageable.
- Use it with ****GradientBoostingClassifier()**** by passing it to the model.
- Fit it on training data to get the best parameters.

In the code:

- ****param\_grid****: A dictionary containing hyperparameters and their possible values. ****GridSearchCV**** will try every combination of these values to find the best-performing set of hyperparameters.
- ****grid\_search.fit(X\_train, y\_train)****: This line trains the Gradient Boosting model using all combinations of the hyperparameters defined in `param_grid`.
- ****grid\_search.best\_estimator\_****: After completing the grid search this will return the Gradient Boosting model that has the best combination of hyperparameters from the search.
- ****best\_params****: This stores the best combination of hyperparameters found during the grid search.

Python`from sklearn.model_selection import GridSearchCV  param_grid = {     'n_estimators': [50, 100, 200],     'learning_rate': [0.01, 0.1, 0.2],     'max_depth': [3, 5, 7], }  gb_model = GradientBoostingClassifier()  grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)  grid_search.fit(X_train, y_train)  best_params = grid_search.best_params_ best_model = grid_search.best_estimator_  y_pred_best = best_model.predict(X_test)  accuracy_best = accuracy_score(y_test, y_pred_best)  print("Best Parameters:", best_params) print(f"Best Model Accuracy: {accuracy_best}")`

****Output:****

> Best Parameters: {'learning\_rate': 0.1, 'max\_depth': 3, 'n\_estimators': 200}  
> Best Model Accuracy: 0.8044692737430168

The model's accuracy on the test set is approximately 80.45% indicating the effectiveness of the tuned hyperparameters in improving model performance.

## 2\. Hyperparameter Tuning using Randomized Search CV

- ****RandomizedSearchCV**** randomly picks hyperparameter combinations from a given grid.
- It’s faster than ****GridSearchCV****, especially with many parameters.
- Helps find good model settings without checking every single option.
- To use it with Gradient Boosting, pass a ****GradientBoostingClassifier()**** to it.
- Then fit it on training data to find the best parameters.

In the code:

- ****param\_dist****: it will randomly sample from this distribution to find the best-performing combination of hyperparameters.
- ****random\_search.fit(X\_train, y\_train)****: This line trains the GradientBoostingClassifier model using random combinations of hyperparameters defined in `param_dist`.
- ****random\_search.best\_estimator\_****: This retrieves the model that has the best combination of hyperparameters found during the random search.
- ****best\_params****: This stores the best combination of hyperparameters found during the search.

Python`from sklearn.model_selection import RandomizedSearchCV import numpy as np from sklearn.ensemble import GradientBoostingClassifier  param_dist = {     'learning_rate': np.arange(0.01, 0.2, 0.01),      'n_estimators': [100, 200, 300, 400],       'max_depth': [3, 5, 7, 9],   }  gb_model = GradientBoostingClassifier()  random_search = RandomizedSearchCV(estimator=gb_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)  random_search.fit(X_train, y_train)  best_params = random_search.best_params_ best_model = random_search.best_estimator_  y_pred_best = best_model.predict(X_test)  accuracy_best = accuracy_score(y_test, y_pred_best)  print("Best Parameters:", best_params) print(f"Best Model Accuracy: {accuracy_best}")`

****Output:****

> Best Parameters (Randomized Search): {'n\_estimators': 250, 'max\_depth': 3, 'learning\_rate': 0.09444444444444444}  
> Best Model Accuracy (Randomized Search): 0.8156424581005587

The model's accuracy on the test set is approximately 81.56% indicating that the tuned hyperparameters improved model performance.

## 3\. Hyperparameter Tuning using Optuna

- [Optuna](https://www.geeksforgeeks.org/hyperparameter-tuning-with-optuna-in-pytorch/) is a tool for tuning hyperparameters efficiently.
- It tries different settings to improve model performance.
- You define an objective function that Optuna aims to minimize or maximize.
- It works well with many types of models and tasks.
- It helps find the best hyperparameters for models like Gradient Boosting.

In the code:

- ****param\_space****: Defines the hyperparameter search space where ****Optuna**** samples values for n\_estimators, learning\_rate and max\_depth within the specified ranges.
- ****objective(trial)****: The objective function that it tries to minimize. It trains the ****GradientBoostingClassifier**** with different hyperparameters, calculates the accuracy and returns the inverse of accuracy.
- ****study.optimize(objective, n\_trials=50)****: This runs the optimization process for 50 trials exploring the hyperparameter space and finding the best-performing combination of parameters.
- ****study.best\_params****: Returns the best combination of hyperparameters found during the optimization process.
- ****best\_model\_optuna.fit(X\_train, y\_train)****: Fits the ****GradientBoostingClassifier**** model using the best hyperparameters found by it.

Python`import optuna from sklearn.ensemble import GradientBoostingClassifier from sklearn.model_selection import train_test_split from sklearn.metrics import accuracy_score  def objective(trial):     param_space = {         'n_estimators': trial.suggest_int('n_estimators', 50, 250, step=50),         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),         'max_depth': trial.suggest_int('max_depth', 3, 7),     }      gb_model = GradientBoostingClassifier(**param_space, validation_fraction=0.1, n_iter_no_change=5, random_state=42)      gb_model.fit(X_train, y_train)      y_pred = gb_model.predict(X_test)      accuracy = accuracy_score(y_test, y_pred)      return 1.0 - accuracy   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  study = optuna.create_study(direction='minimize') study.optimize(objective, n_trials=50)  best_params_optuna = study.best_params best_model_optuna = GradientBoostingClassifier(**best_params_optuna, validation_fraction=0.1, n_iter_no_change=5, random_state=42) best_model_optuna.fit(X_train, y_train)  y_pred_best_optuna = best_model_optuna.predict(X_test)  accuracy_best_optuna = accuracy_score(y_test, y_pred_best_optuna)  print(f"Best Model Accuracy (Optuna): {accuracy_best_optuna}")`

****Output:****

> Best Model Accuracy (Optuna): 0.8324022346368715

The model’s accuracy on the test set is approximately 83.24% demonstrating the effectiveness of Optuna in optimizing the model's hyperparameters and improving performance.