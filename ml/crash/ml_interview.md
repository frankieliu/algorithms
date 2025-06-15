1.Big Picture: What are the typical stages involved in building an ML classifier?

Business Understanding: Understanding the problem (e.g. figuring out the target/label and metrics).
Problem Formulation: Figure out what should be used as the target/label.
Feature Construction: Look through the feature set available to solve the problem.
Exploratory Analysis: See how features are interacting with the label to better understand data distribution and figure out whether more transformations are needed.
Model Building: Start with a baseline model (simple and easy to explain, e.g. logistic regression) and add more complexity from there (e.g. gradient boosting tree method, feature selection, feature transformation, deep learning methodology depending on the data structure / tabular vs. unstructured). Could leverage certain techniques such as transform learning.
Validate Performance Offline: ROC/AUC score, recall / precision, F1 score (especially for imbalance set).
Implementation Strategy: A/B test. We need to determine which north star / long term metrics we want to improve on. We also need to determine sample size and duration of the A/B. We need to consider the power of test, figure out how to avoid bias. 
 

2. Data: What issues do you expect to encounter around data?

 

Understanding (pre-constructed) features (how were they constructed, what do they mean).
Missing values: (e.g. for linear regression, perceptron)
Imputation: e.g. mean imputation (numeric features), interpolation (numeric feature), mode imputation (categorical features), missing values as part of the feature.
Dropping rows with missing values (for tabular dataset, when there aren’t too many rows with missing values). Eventually we want to make sure we have enough data to train after dropping missing values.
Multicollinearity: (feature too linearly dependent to each other).
Applies to models that are linear in nature.
Introduces too much variance.
Less interpretability.
Treatment: 
Regularization (L1 or L2).L1 would drop down some of the features to 0, whereas L2 would not.
Drop some features using the VIF technique.
High-dimensionality / Sparsity:
Data that has too many columns but a lot of columns are not informative.
Treatment: different encoding methodology (e.g. ordinal or target encoding instead of one-hot-encoding), or using transformation techniques such as PCA to reduce the number of features. Trim down the list of categories (e.g. top k + others).
Q: Issues with PCA. A: We might have difficulty finding principal components. The output might be hard to interpret. 
5. Features might not be meaningful: (we might need feature selection).




3. Feature Selection Techniques:

Variance selection technique (reduce features that have a low variance).
Q: This could be problematic, why? A: Some variances might be lower by nature.
L1 norm: would help drop down some of the features to 0.
Random Forest for feature selection.
Q: How does RF do feature selection. A: Select a subsample of the data through bagging, select a random feature and figure out how much predictive power this feature has.
Shapley Values: way to help find the direction of the feature weights for a non-linear model.



4. Model Selection:

What classification models do you know and what model would you use in what circumstances?
Logistic Regression
KNN
Naive Bayes
Decision Tree
Gradient Boosting Tree
Random Forest
MLP / Deep Learning
Q: What is the difference between decision trees, gradient boosting trees, and random forest?

A: Random forest: using bagging to build a set of independent trees. GBT: goes through iterations, each trying to improve on the mistake of the previous. DT: single tree.

Q: What is the difference between bagging and boosting?

A: Bagging takes a random sample of the dataset, whereas boosting predict the errors of the previous terms in order to reduce the bias. One of the best models for tabular data.




5: Overfitting:

What is overfitting?
Occurs when the model performs well on the training dataset but does not generalize well on the testing dataset.

 

Treatments:

Add more data so that the model would generalize better.
Regularization (L1/L2).
Turn the hyper-parameter (e.g. learning rate for gradient descent based optimizations)
Early stopping (applies to models that are going through multiple iterations of optimization).
Proper train (64) /validation (16)/test split (20).
K-fold cross-validation.



What is/are the cause(s) of overfitting?
 

Too little data or data sparsity.
Imbalance dataset.
Model complexity.