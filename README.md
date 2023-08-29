# Artificial-Pancreas-Data-Analysis-for-Meal-Prediction

This project provides valuable insights into
the factors that impact blood glucose levels, such as the
impact of meals, the determination of meal times for
a person and the performance of an artificial pancreas
system with the help of Data mining techniques. The
results offer new strategies for improving meal tracking
and glycemic control for individuals with diabetes. The
project report summarizes the work and results of three
data mining projects, including data cleaning, feature
engineering, machine learning, and cluster validation.


An artificial pancreas system is a medical control
system that is used to manage blood glucose
levels in diabetic patients. It consists of a
continuous glucose monitor (CGM) and an insulin
pump that modulates insulin delivery based on
the CGM data. In this project, the data from
the Medtronic 670G system is considered. The
project report covers three data mining projects as
described below:

• Project 1: This project aims to extract and
compute a total of 18 metrics (refer: Requirements.txt) using the Insulin
and CGM sensor data from the Medtronic
670G, an Artificial Pancreas system, for both
manual and auto mode.

• Project 2: This project (refer: train.py, test.py) aims to develop a
machine learning model that can accurately
classify a time series data as meal or no
meal. It involves extracting features from data,
training the model, and evaluating it using
k-fold cross validation on the training data.
The output is a vector of 1s and 0s indicating
whether a sample represents a person who has
eaten or not eaten, respectively.

• Project 3: This project (refer: DecisionTree_Cluster.py) aims to use clustering
by K-means and DBSCAN based on
the amount of carbohydrates in each meal.
It involves comparing the clustering results
with a known ground truth by using cluster
validation techniques
