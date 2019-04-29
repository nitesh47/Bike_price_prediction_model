# Bike_price_prediction_model

INTRODUCTION: Bike sharing systems consist of a fleet of bikes available for rent for 
registered and casual users. Predicting the hourly bike demand will help in designing and
expanding bike sharing system as well as can be used to make important business decisions
like, maintaining adequate number of bikes at working stations, scheduling bike
maintenance task during low demand period and provide better customer support in terms
of cost and user experience.

DATA EXPLORATION: To understand the behavior of bike demand, we plotted Box plot and
Point plot under various parameters

Figure 1: Box plot for bike demand(cnt)
Figure 2: Box plot for bike demand for different seasons, which shows that bike demand is
higher in autumn and summer compared to winter and springer.
Figure 3: Point plot representing bike demand for holidays (i.e., weekend and holidays) and
working days, showing different behavior corresponding to various use cases (i.e., office
commute on workdays or casual riding on holidays)
Figure 4: Point plot representing bike demand by registered users on working days, which
shows high demand during office commute hours
Figure 5: Point plot representing bike demand by casual users on working days, which shows
linear increase in demand between 08:00 to 17:00 and very low demand during night time.
Figure 6 & 7: Point plot representing bike demand during weekends, showing similar
behavior by registered and casual users
Figure 8: Heat map plot representing correlation between attributes (i.e., day or weather)
and bike demand (i.e., registered or casual)

PREDICTIVE TASK: Our predictive task is to forecast bike rental demand of Bike sharing
program based on historical usage patterns in relation with weather, date and time.
Evaluation of model: We will be evaluating our model on the basis of mean absolute
deviation

Algorithm for Prediction: We evaluated performance of a few regression and classification
algorithms and found that extra tree regressor [1] performed better than other
algorrithms(i.e., BayesianRidgeRegression , AdaBoostClassifier, and RandomForestRegressor)
in terms of training time and prediction performance.

Prediction Approach: As observed earlier, casual user and registered user were showing
different behavior in terms of bike demand, so instead of training single regression model for
target class cnt, we trained an ensemble consisting individual regression models for target
classes like casual and registered and are saved on drive for future predictions.
To predict overall bike demand, predictions can be made for different type of users and
summation of prediction represents overall demand (i.e., cnt). Our evaluation results show
that this approach performs better with similar parameters (i.e., prediction model and
training attributes) compared to single regression model.

Performance: Mean Absolute Dev (ExtraTreesRegressor, 30% test): 23.78821212121212
Please find attached source code and performance results.

Task 1.2: To improve the code readability and maintenance, code is written in a structured
format maximizing use of functions and avoid hard coded values. Clean code practices are
used like meaningful variable names and comments to improve overall readability and
documentation of code. To keep future predictions accurate, regression model should be
updated frequently when new data is available.

Task 2:
Task 2.1: Current solution is suitable for small-medium data set using data storage and
machine learning libraries designed for above use case. while analyzing big data (i.e., several terabytes) we can face several challenges.
Challenges:

1) Data Storage: Usually single node machines are not designed for storing big data(i.e.,
In-memory and permanent storage) and since fast data storage are very expensive,
distributed data storage is a viable solution

2) Data Processing: current approach’s data preprocessing and training time is
dependent upon data, big data can increase overall time required for training
proportional to data size.

Task 2.2: To scale up the prediction model, we can adapt our application for big data or
hadoop environment.

1) Data Storage: For in memory data storage we can use dask instead of numpy, which
is designed for parallel data processing on a Hadoop cluster and instead of using csv
data file we can store data in column-based storage like apache hive or apache
parquet, which provides high compression and low latency.

2) Data Processing: To process big data, we need to process data parallelly on multiple
cluster nodes and use data preprocessing libraries designed to take advantage of
multi-threaded CPU architecture. We can use apache spark to process large amount
of data and build regression models using processed data.

Task 2.3: In addition to big data challenges, prediction models should also be updated with
the arrival of new instances or training data. Current machine learning approach are
designed for stationary data (i.e., correlation between attributes and target class does not
change over time), in most of real world use cases correlation between attributes and target
class can change over time (i.e., concept drift) resulting outdated prediction model.
To resolve mentioned limitation, prediction model should be able to detect change in data
(i.e., concept drift) and adapt to the change in the concept by forgetting older concepts and
learn new concepts.

Task 2.4:
I have hands-on experience with big data processing during my studies, which includes
processing data stream using Amazon AWS and Cloudera manager.
I have gained practical experience working with data streams during master thesis
implementation.
