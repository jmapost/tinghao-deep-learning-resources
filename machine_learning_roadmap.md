# Machine Learning Roadmap


1. Exploratory Data Analysis (EDA) -> pandas or matplotlib or seaborn

To get some insights for subsequent processing and modeling. Check list:

(a) Many search terms / products appeared several times.

(b) Text similarities are great features.

(c) Many products don’t have attributes features. Would this be a problem?

(d) Product ID seems to have strong predictive power. However the overlap of product ID between the training set and the testing set is not very high. Would this contribute to overfitting?


2. Feature Engineering / Feature Selection

(1) Distribution of features -> box plot -> box-cox transformation (normalize).

Note: Tree-based models don't depend on normalization, but neural networks do care.

(2) Features are linear related -> heat map plot -> Pearson correlation coefficient

(3) Outliers -> scatter plot -> remove outliers

(4) Classification -> scatter plot with colored labels

(5) Missing data -> mean, medium, delete, ...???

(6) Stack train & test -> Don't have to do feature transformation twice.

(6) Categorical variables -> stack train & test -> one-hot encoded

(7) Noise -> less regularized, more iterations or depth of trees or deeper networks

(8) Mixed features -> add, minus, multiply, divide by, ...???

(9) Count attributes. Find those frequent and easily exploited ones.




3. Modeling -> set seed

(1) Select Models

(a) Regression
- Linear Regression
- Lasso
- Polynomial Regression
- Gradient Boost Machine (GBM)
- XGBoost
- Neural Networks
- Ensembles

(b) Classification
- Decision Tree
- Random Forest
- Logistic Regression
- Support Vector Machines (SVMs)
- k-nearest neighbors
- Naive Bayes
- Gradient Boost Machine (GBM)
- XGBoost
- Neural Networks
- Ensembles

(c) Clustering
- k-means
- Gaussian Mixture Model (GMM)
- Mean-shift
- DBSCAN
- Agglomerative Clustering

(d) Dimension Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Isometric Feature Mapping (Isomap)



(2) Tune Parameters

(a) Grid Search

(b) Bayesian Optimization



(3) XGBoost Tunning Tips

(a) eta: Step size used in updating weights. Lower eta means slower training but better convergence.

(b) num_round: Total number of iterations.

(c) subsample: The ratio of training data used in each iteration. This is to combat overfitting.

(d) colsample_bytree: The ratio of features used in each iteration. This is like max_features in RandomForestClassifier.

(e) max_depth: The maximum depth of each tree. Unlike random forest, gradient boosting would eventually overfit if we do not limit its depth.

(f) early_stopping_rounds: If we don’t see an increase of validation score for a given number of iterations, the algorithm will stop early. This is to combat overfitting, too.

(h) Steps:

Step 1. Reserve a portion of training set as the validation set.

Step 2. Set eta to a relatively high value (e.g. 0.05 ~ 0.1), num_round to 300 ~ 500.

Step 3. Use grid search to find the best combination of other parameters.

Step 4. Gradually lower eta until we reach the optimum.

Step 5. Use the validation set as watch_list to re-train the model with the best parameters. Observe how score changes on validation set in each iteration. Find the optimal value for early_stopping_rounds.



(4) Cross Validation (CV)

(a) (Kaggle) Public leader board scores are not consistent with local CV scores due to noise or non ideal distribution. Local CV > public leader board.

(b) 5-fold CV is good enough.



(5) Ensemble Models

It reduces both bias and variance of the final model. Base models should be as unrelated as possibly. This is why we tend to include non-tree-based models in the ensemble even though they don’t perform as well. The math says that the greater the diversity, and less bias in the final ensemble. Also, performance of base models shouldn’t differ to much.

(a) Bagging: 

Use different random subsets of training data to train each base model. Then all the base models vote to generate the final predictions. This is how random forest works.

(b) Boosting: 

Train base models iteratively, modify the weights of training samples according to the last iteration. This is how gradient boosted trees work. (Actually it’s not the whole story. Apart from boosting, GBTs try to learn the residuals of earlier iterations.) It performs better than bagging but is more prone to overfitting.

(c) Blending: 

Use non-overlapping data to train different base models and take a weighted average of them to obtain the final predictions. This is easy to implement but uses less data.

(d) Stacking:

Take 5-fold stacking as an example. First we split the training data into 5 folds. Next we will do 5 iterations. In each iteration, train every base model on 4 folds and predict on the hold-out fold. You have to keep the predictions on the testing data as well. This way, in each iteration every base model will make predictions on 1 fold of the training data and all of the testing data. After 5 iterations we will obtain a matrix of shape #(samples in training data) X #(base models). This matrix is then fed to the stacker (it’s just another model) in the second level. After the stacker is fitted, use the predictions on testing data by base models (each base model is trained 5 times, therefore we have to take an average to obtain a matrix of the same shape) as the input for the stacker and obtain our final predictions.



(6) Pipline -> create a highly automated pipeline

(a) Automated grid search / bayesian optimization

(b) Automated ensemble selection.


[Reference](https://www.kdnuggets.com/2016/11/rank-ten-precent-first-kaggle-competition.html)
