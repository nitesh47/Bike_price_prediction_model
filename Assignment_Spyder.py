#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing Data Analysis and Prediction Model

# Author: Nitesh Pandey(niteshpandey9869@gmail.com)

# # Import packages
# The main packages for doing data analysis, visualizytion and machine learning(prediction) are numpy, pandas, scikit learn, pickle, matplotlib and seaborn


#%% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import pickle

from sklearn.ensemble import ExtraTreesRegressor
from datetime import timedelta


#%% Bike sharing analysis class
class BikeSharingAnalysis:
    
    def __init__(self, data_file):
        self.overall_DataFrame = self.read_data(data_file)
        
    # Method to initialize method parameters
    # Input: fig_dim_small (i.e., Plot Dimensions for small plots)
    # Input: fig_dim_big (i.e., Plot Dimensions for big plots)
    # Input: font_size_text (i.e., font size for legends)
    # Input: font_size_title (i.e., font size for title)
    # Input: fig_counter(i.e., counter for plots)
    # Input: border_width (i.e., margins for plot)
    # Input: train_data_window (i.e., batch size for regression model training)
    # Input: max_abs_diff_count (i.e., threshold value to identify bad performance and model retraining)
    # Input: incremental_model_update (i.e., boolean value to enable and disable incremental learning)
    # Input: algorithm (i.e., Prediction algorithm used)
    # Input: test_train_split_day(i.e., latest n number of days will be used as test set)
    # Input: parameter_n_estimators (i.e., Model training parameter)
    # Input: remove_columns(i.e., list of non relevant attributes)
    # Input: categorical_attributes(i.e., list of categorical attributes)
    # Input: attributes_plus_targetclass(i.e., list of attributes and target class)
    # Input: selected_attributes(i.e., list of required attributes for prediction)
    # Input: target_attribute_overall(i.e., Target class overall bike demand)
    # Input: target_attribute_primary(i.e., Primary target class bike demand casual)
    # Input: target_attribute_secondary(i.e., Secondary target class bike demand registered)
    def initialize_analysis_parameters ( self, fig_dim_small, fig_dim_big, font_size_text, font_size_title, fig_counter, border_width, algorithm, train_data_window, max_abs_diff_count, incremental_model_update, test_train_split_day, parameter_n_estimators, remove_columns, categorical_attributes, attributes_plus_targetclass, selected_attributes, target_attribute_overall, target_attribute_primary, target_attribute_secondary):
        self.fig_dim_small = fig_dim_small
        self.fig_dim_big = fig_dim_big
        self.font_size_text = font_size_text
        self.font_size_title = font_size_title
        self.fig_counter = fig_counter
        self.border_width = border_width
        self.train_data_window = train_data_window
        self.max_abs_diff_count = max_abs_diff_count
        self.incremental_model_update = incremental_model_update
        self.test_train_split_day=test_train_split_day
        self.parameter_n_estimators = parameter_n_estimators
        self.algorithm = algorithm
        self.remove_columns = remove_columns
        self.categorical_attributes = categorical_attributes
        self.attributes_plus_targetclass =  attributes_plus_targetclass
        self.selected_attributes = selected_attributes
        self.target_attribute_overall = target_attribute_overall
        self.target_attribute_primary = target_attribute_primary
        self.target_attribute_secondary = target_attribute_secondary
    
    # Method to read csv file
    # Input: data_file (i.e., file name )
    def read_data(self, data_file): 
        # # Read CSV file and manipulate the data
        # Read the data file named hour.csv using read_csv function in pandas
        data = pd.read_csv(data_file)        
        print('Data Loaded')
        return data


    # Method to calculate Mean absolute deviation
    # Input: predictions (i.e., Predicted demand )
    # Input: realizations (i.e., Actual demand )
    def Mean_absloute_dev(self, predictions, realizations):
        mad=np.mean(np.absolute(np.array(predictions)-np.array(realizations)))
        return mad

    # Method to calculate Root Mean Square Log Error
    # Input: predictions (i.e., Predicted demand )
    # Input: realizations (i.e., Actual demand )
    def RMSLE(self, predictions, realizations):
        predictions=predictions.clip(0)
        rmsle=np.sqrt(np.mean(np.array(np.log(predictions.astype(float) + 1) - np.log(realizations.astype(float) + 1))**2))
        return rmsle
    # Method to save plot under Plots folder and increment plot counter
    # Input: plt (i.e., plot object)
    def save_increment_plot(self, plt):
        plt.savefig("Plots/Figure_"+ str(self.fig_counter) +".png")
        self.fig_counter +=1
    
    # Method to plot correlation between attributes
    def print_correlation_plot(self):
        data_correlation = self.overall_DataFrame[self.attributes_plus_targetclass].corr()
        mask = np.array(data_correlation)
        mask[np.tril_indices_from(mask)] = False
        plt.subplots(figsize=self.fig_dim_small)
        plt.suptitle('Figure '+ str(self.fig_counter) +' : Correlation between attributes ', fontsize=self.font_size_title)
        sn.heatmap(data_correlation, mask=mask, vmax=1, square=True, annot=True)
        self.save_increment_plot(plt)

    # Method to plot predictions vs actual demand
    # Input: results_pred (i.e., Predicted demand)
    # Input: results_actual (i.e., Actual demand)
    # Input: target_attribute (i.e., Predicted / Target attribute name)
    def plot_prediction_and_actual_demand(self,results_pred,results_actual,target_attribute):
        plt.figure(figsize=self.fig_dim_small)
        plt.scatter(results_pred, results_actual, s=0.2)
        min_value= -self.border_width
        max_value= max(np.array(results_pred).max(),np.array(results_actual).max()) + self.border_width
        plt.xlim(min_value,max_value)
        plt.ylim(min_value,max_value)
        plt.plot([min_value,max_value],[min_value,max_value], color='r', linestyle='-', linewidth=2)
        plt.suptitle('Figure '+ str(self.fig_counter) +' : Predicted Vs Actual ( '+target_attribute+' ) : ' + self.algorithm, fontsize = self.font_size_title)
        plt.xlabel('Predicted Bike demand', fontsize=self.font_size_text)
        plt.ylabel('Real Bike demand', fontsize=self.font_size_text)
        self.save_increment_plot(plt)
        plt.show()
        return
    
    # Method to preprocess data
    def data_preprocess(self):
        # # Feature Engineering
        # 1) Remove less relevant attributes(i.e., not relevant for prediction or visualization) like instant and yr
        self.overall_DataFrame = self.overall_DataFrame.drop(self.remove_columns, axis=1)
        
        # # Feature Engineering
        # 2) From the date attribute, extract day of month and year and add these features to the dataframe as new attributes
        tempDataTime = pd.DatetimeIndex(self.overall_DataFrame['dteday'])
        self.overall_DataFrame['year']=tempDataTime.year
        self.overall_DataFrame['day']=tempDataTime.day
        
        # # Feature Engineering
        # 3) Change dataframe column names to more meaningful names
        self.overall_DataFrame.rename(columns={'weathersit':'weather',
                                 'mnth':'month',
                                 'hr':'hour',
                                 'hum': 'humidity',
                                 'cnt':'count',
                                 'dteday':'timestamp'},
                         inplace=True)
        
        self.overall_DataFrame['timestamp']=pd.DatetimeIndex(self.overall_DataFrame['timestamp'])
        
        if self.overall_DataFrame.isnull().values.any():
            print("Null Values Present")
        else:
            print("Null Values are not present")
        return
    
    # Method to generate visualization plots
    def data_visualization(self):
        fig, ax = plt.subplots(figsize=self.fig_dim_small)
        plot = sn.boxplot(data=self.overall_DataFrame[['count']], ax=ax)
        plot.set_xlabel("Bike Demand : count",fontsize=self.font_size_text)
        plot.axes.set_title("Figure "+ str(self.fig_counter) +" : Boxplot for bike demand",fontsize=self.font_size_title)
        self.save_increment_plot(plt)
        
        fig, ax = plt.subplots(figsize=self.fig_dim_big)
        plot=sn.boxplot(x=self.overall_DataFrame.loc[ : , 'season' ],y=self.overall_DataFrame.loc[ : , 'count'], ax=ax)
        plot.set_xlabel("Seasons",fontsize=self.font_size_text)
        plot.set_ylabel("Bike Demand : count",fontsize=self.font_size_text)
        plot.axes.set_title("Figure "+ str(self.fig_counter) +" : Boxplot for bike demand for diffrent seasons",fontsize=self.font_size_title)
        self.save_increment_plot(plt)
        
        fig, ax = plt.subplots(figsize=self.fig_dim_big)
        plot = sn.pointplot(data=self.overall_DataFrame.loc[ : ,['hour','count','holiday']],
                            x='hour', y='count',
                            hue='holiday',ax=ax, dodge=True)
        plot.set_xlabel("Hours",fontsize=self.font_size_text)
        plot.set_ylabel("Bike Demand : count",fontsize=self.font_size_text)
        plot.axes.set_title("Figure "+ str(self.fig_counter) +" : Use of bike sharing service on weekdays vs holidays",fontsize=self.font_size_title)
        self.save_increment_plot(plt)
        
        weekend_DataFrame=self.overall_DataFrame.loc[(self.overall_DataFrame['weekday'] == 0) | (self.overall_DataFrame['weekday'] == 6)]
        weekday_DataFrame=self.overall_DataFrame.loc[(self.overall_DataFrame['weekday'] != 0) & (self.overall_DataFrame['weekday'] != 6)]
        
        
        fig, ax = plt.subplots(figsize=self.fig_dim_big)
        plot = sn.pointplot(data=weekday_DataFrame.loc[ : ,['hour','registered','weekday']],
                            x='hour', y='registered',
                            hue='weekday',ax=ax, dodge=True)
        plot.set_xlabel("Hours",fontsize=self.font_size_text)
        plot.set_ylabel("Bike Demand : count",fontsize=self.font_size_text)
        plot.axes.set_title("Figure "+ str(self.fig_counter) +" : Use of the system by registered users on weekdays",fontsize=self.font_size_title)
        self.save_increment_plot(plt)
        
        fig, ax = plt.subplots(figsize=self.fig_dim_big)
        plot = sn.pointplot(data=weekday_DataFrame.loc[ : , ['hour','casual','weekday']],
                            x='hour', y='casual',
                            hue='weekday',ax=ax, dodge=True)
        plot.set_xlabel("Hours",fontsize=self.font_size_text)
        plot.set_ylabel("Bike Demand : count",fontsize=self.font_size_text)
        plot.axes.set_title("Figure "+ str(self.fig_counter) +" : Use of the system by casual users on weekdays",fontsize=self.font_size_title)
        self.save_increment_plot(plt)
        
        
        fig, ax = plt.subplots(figsize=self.fig_dim_big)
        plot = sn.pointplot(data=weekend_DataFrame.loc[ : , ['hour','registered','weekday']],
                            x='hour', y='registered',
                            hue='weekday',ax=ax, dodge=True)
        plot.set_xlabel("Hours",fontsize=self.font_size_text)
        plot.set_ylabel("Bike Demand : count",fontsize=self.font_size_text)
        plot.axes.set_title("Figure "+ str(self.fig_counter) +" : Use of the system by registered users on weekend",fontsize=self.font_size_title)
        self.save_increment_plot(plt)
        
        fig, ax = plt.subplots(figsize=self.fig_dim_big)
        plot = sn.pointplot(data=weekend_DataFrame.loc[ : ,['hour','casual','weekday']],
                            x='hour', y='casual',
                            hue='weekday',ax=ax, dodge=True)
        plot.set_xlabel("Hours",fontsize=self.font_size_text)
        plot.set_ylabel("Bike Demand : count",fontsize=self.font_size_text)
        plot.axes.set_title("Figure "+ str(self.fig_counter) +" : Use of the system by casual users on weekend",fontsize=self.font_size_title)
        self.save_increment_plot(plt)
        return
    
    # methon to change datatype of catagorical attributes
    def convert_categorical_attributes(self):
        for column in self.categorical_attributes:
            self.overall_DataFrame[column] = self.overall_DataFrame[column].astype('category')            
        return
    
    # method to build prediction model and generate predictions on test data
    def build_model_generate_predictions(self):
        # Subsetting overall data into training and testing data frames
        initial_test_train_split_date = self.overall_DataFrame['timestamp'].max() - timedelta(days = self.test_train_split_day)
        
        # Setting build_models = True for first build
        build_models = True
        
        # Initialize dataframe object to save perdicted and actual demand for evaluation pourpose
        self.results = pd.DataFrame(columns=['real_overall','real_primary', 'real_secondary', 'pred_primary', 'pred_secondary', 'pred_overall'])
        for i in range(self.test_train_split_day):
            # For each day, generate predictions, perform brief evaluation and retrain prediction models of performance is below user specified value(i.e., max_abs_diff_count)
            # Calculate timestamp values for test and train data spliting
            max_train_date = initial_test_train_split_date + timedelta(days = i)
            test_date = initial_test_train_split_date + timedelta(days = i)
            
            if build_models:
                # method for building primary and secondary models(i.e., for predicting registered and casual demand)
                model_primary, model_secondary = self.build_model(max_train_date)
                build_models = False
            
            test_dataset = self.overall_DataFrame[self.overall_DataFrame['timestamp']==test_date]
            
            test_dataframe_predictors = test_dataset.loc[ : ,self.selected_attributes]
            test_dataframe_targetclass_overall = test_dataset.loc[ : ,self.target_attribute_overall]
            test_dataframe_targetclass_primary = test_dataset.loc[ : ,self.target_attribute_primary]
            test_dataframe_targetclass_secondary = test_dataset.loc[ : ,self.target_attribute_secondary]
            
            # # Demand prediction using regression model
            # Perform Prediction using Extra Trees Regressor from sklearn ensumble. As observed earlier, bike demand is dependent on casual users and registered users, which are having diffrent behaviours depending upon week day. 
            # We have decided to use diffrent prediction models for casual and registered users. 
            # Generate predictions using built models
            model_primary_predictions = model_primary.predict(test_dataframe_predictors)
            model_secondary_predictions = model_secondary.predict(test_dataframe_predictors)
            
            # Save prediction models 
            filename = 'ExtraTreeRegression_Model_primary.sav'
            pickle.dump(model_primary, open(filename, 'wb'))
                
            filename = 'ExtraTreeRegression_Model_secondary.sav'
            pickle.dump(model_secondary, open(filename, 'wb'))
            
            # Insert prediction results for brief evaluation
            temp_results = pd.DataFrame({'real_overall':np.array(test_dataframe_targetclass_overall.loc[ : ,'count']),
                                         'real_primary':np.array(test_dataframe_targetclass_primary.loc[ : ,'casual']),
                                         'real_secondary':np.array(test_dataframe_targetclass_secondary.loc[ : ,'registered']), 
                                         'pred_primary':model_primary_predictions, 
                                         'pred_secondary':model_secondary_predictions})
            temp_results['pred_overall']=np.array(temp_results.loc[ : ,'pred_primary']) + np.array(temp_results.loc[ : ,'pred_secondary'])
            # Append prediction into self.results(i.e., overall predictions)
            self.results = self.results.append(temp_results, ignore_index = True)
            
            # Calculate absolute diffrences between predicted and actual damand
            temp_results['abs_diff_casual'] = temp_results.loc[ : ,'pred_primary'] - temp_results.loc[ : ,'real_primary']
            temp_results['abs_diff_casual'] = temp_results['abs_diff_casual'].abs()
            temp_results['abs_diff_registered'] = temp_results.loc[ : ,'pred_secondary'] - temp_results.loc[ : ,'real_secondary']
            temp_results['abs_diff_registered'] = temp_results['abs_diff_registered'].abs()
            temp_results['abs_diff_count'] = temp_results.loc[ : ,'pred_overall'] - temp_results.loc[ : ,'real_overall']
            temp_results['abs_diff_count'] = temp_results['abs_diff_count'].abs()
            
            # If incremental_model_update is enabled by user, perform brief evaluation and evaluate if model retraining is required or not
            if self.incremental_model_update:
                build_models = self.if_update_models_required(temp_results)
            
        return
    # Method to evaluate model performance, return boolean value representing if performance is above user expectations or not
    # If performance is low, models will be retrained
    def if_update_models_required(self, results):
        mean_abs_diff_count = results['abs_diff_count'].mean()
        
        if mean_abs_diff_count >= self.max_abs_diff_count :
            build_models=True
        else:
            build_models=False
        return build_models
        
    # Method to build prediction models
    # Input: max_train_date(i.e., timestamp value to distinguish between test and train data )
    def build_model(self,max_train_date):

        # Subsetting overall data into training and testing data frames
        # using train_data_window value, select instances for training in sliding window
        min_train_date = max_train_date - timedelta(days = self.train_data_window)
        
        train_dataset = self.overall_DataFrame[(self.overall_DataFrame['timestamp']<max_train_date) & (self.overall_DataFrame['timestamp']<min_train_date)]        
        train_dataframe_predictors = train_dataset.loc[ : ,self.selected_attributes]
        train_dataframe_targetclass_primary = train_dataset.loc[ : ,self.target_attribute_primary]
        train_dataframe_targetclass_secondary = train_dataset.loc[ : ,self.target_attribute_secondary]
        # # Demand prediction using regression model
        # Perform Prediction using Extra Trees Regressor from sklearn ensumble. As observed earlier, bike demand is dependent on casual users and registered users, which are having diffrent behaviours depending upon week day. 
     
        # We have decided to use diffrent prediction models for casual and registered users. 
          
        # Training regression models for target class casual
        model_primary = ExtraTreesRegressor(n_estimators=self.parameter_n_estimators)
        model_primary = model_primary.fit(train_dataframe_predictors, 
                                              train_dataframe_targetclass_primary[self.target_attribute_primary].values.ravel())
          
        # Training regression models for target class registered
        model_secondary = ExtraTreesRegressor(n_estimators=self.parameter_n_estimators)
        model_secondary = model_secondary.fit(train_dataframe_predictors, 
                                                  train_dataframe_targetclass_secondary[self.target_attribute_secondary].values.ravel())
        
        # Return trained models
        return model_primary, model_secondary;
        
    
    # Evaluate model performance and plot predicted and actual bike demand
    def evaluate_model_predictions(self):
        
        print('Target Attribute : ' + self.target_attribute_primary[0])
        print('Mean Absolute Dev : ' + self.algorithm , self.Mean_absloute_dev(self.results.loc[ : ,'pred_primary'],
                                                                   self.results.loc[ : ,'real_primary']))
        print('RMLSE : ' + self.algorithm , self.RMSLE(self.results.loc[ : ,'pred_primary'], 
                                             self.results.loc[ : ,'real_primary']))
        self.plot_prediction_and_actual_demand(self.results.loc[ : ,'pred_primary'], 
                                          self.results.loc[ : ,'real_primary'],
                                          self.target_attribute_primary[0])
        
        
        print('Target Attribute : ' + self.target_attribute_secondary[0])
        print('Mean Absolute Dev : ' + self.algorithm , self.Mean_absloute_dev(self.results.loc[ : ,'pred_secondary'],
                                                                   self.results.loc[ : ,'real_secondary']))
        print('RMLSE : ' + self.algorithm , self.RMSLE(self.results.loc[ : ,'pred_secondary'], 
                                             self.results.loc[ : ,'real_secondary']))
        self.plot_prediction_and_actual_demand(self.results.loc[ : ,'pred_secondary'], 
                                          self.results.loc[ : ,'real_secondary'],
                                          self.target_attribute_secondary[0])
        
        
        print('Target Attribute : ' + self.target_attribute_overall[0])
        print('Mean Absolute Dev : ' + self.algorithm , self.Mean_absloute_dev(self.results.loc[ : ,'pred_overall'],
                                                           self.results.loc[ : ,'real_overall']))
        print('RMLSE : ' + self.algorithm , self.RMSLE(self.results.loc[ : ,'pred_overall'], 
                                     self.results.loc[ : ,'real_overall']))
        self.plot_prediction_and_actual_demand(self.results.loc[ : ,'pred_overall'],
                                  self.results.loc[ : ,'real_overall'],
                                  self.target_attribute_overall[0])
        self.results['abs_diff_casual'] = self.results.loc[ : ,'pred_primary'] - self.results.loc[ : ,'real_primary']
        self.results['abs_diff_casual'] = self.results['abs_diff_casual'].abs().astype(float)
        self.results['abs_diff_registered'] = self.results.loc[ : ,'pred_secondary'] - self.results.loc[ : ,'real_secondary']
        self.results['abs_diff_registered'] = self.results['abs_diff_registered'].abs().astype(float)
        self.results['abs_diff_count'] = self.results.loc[ : ,'pred_overall'] - self.results.loc[ : ,'real_overall']
        self.results['abs_diff_count'] = self.results['abs_diff_count'].abs().astype(float)
        
        # Save prediction results for further analysis
        self.results.to_csv('results.csv', sep=',', encoding='utf-8')
        
        fig, ax = plt.subplots(figsize=self.fig_dim_big)
        plot = sn.boxplot(x="variable", y="value", data=pd.melt(self.results.loc[ : ,'abs_diff_casual':'abs_diff_count']))
        plot.set_xlabel("Predicted Values",fontsize=self.font_size_text)
        plot.set_ylabel("Absolute diffrence in bike demand",fontsize=self.font_size_text)
        plot.axes.set_title("Figure "+ str(self.fig_counter) +" : Absolute diffrence between pred and real",fontsize=self.font_size_title)
        self.save_increment_plot(plt)
        return
    
#%% Main Method

def main():
        # Parameters to control and easily modify application behaviour
        # Dimensions for small and big plots 
        fig_dim_small=[10,10]
        fig_dim_big=[20,10]
        
        # Data file name CSV
        data_file='data/hour.csv'
        
        # Plot text size
        font_size_text=10
        
        # Plot Title size
        font_size_title=20
        
        # Counter for plots
        fig_counter=1
        
        # Plot border or margin
        border_width=10
        
        # Boolean value to enable and disable incremental learning
        incremental_model_update = True
        
        # Threshold value to identify required retraining
        max_abs_diff_count = 50.0
        
        # Window size for training data
        train_data_window=365
        
        # Criteria to split overall data into test and train
        test_train_split_day=15
        
        # Model training parameter
        parameter_n_estimators=500
        
        # Prediction or regression model
        algorithm='ExtraTreesRegressor'
        
        # List of irrelevant attributes
        remove_columns=['instant','yr']
        
        # List of categorical attributes
        #categorical_attributes=['season','month','hour','holiday','weekday','workingday','weather','day','year']
        categorical_attributes=['season','month','hour','holiday','weekday','workingday','weather']
        
        # List of attributes and target classes
        attributes_plus_targetclass=['season','month','hour','year','holiday','workingday','weekday','weather','atemp','humidity','windspeed','casual', 'registered','count']
        
        # List of attributes required for bike demand prediction
        selected_attributes = ['season','month','hour','year','holiday','workingday','weekday','humidity','weather','atemp','windspeed']
        
        # Target Classes (i.e., To be predicted)
        target_attribute_overall=['count']
        target_attribute_primary=['casual']
        target_attribute_secondary=['registered']
        
        # Start Analysis
        analysis_object = BikeSharingAnalysis(data_file)
        
        # Initialize parameters
        analysis_object.initialize_analysis_parameters(fig_dim_small, fig_dim_big, font_size_text, font_size_title, fig_counter, border_width, algorithm, train_data_window, max_abs_diff_count, incremental_model_update, test_train_split_day, parameter_n_estimators, remove_columns, categorical_attributes, attributes_plus_targetclass, selected_attributes, target_attribute_overall, target_attribute_primary, target_attribute_secondary)
        
        # Preprocess data
        analysis_object.data_preprocess()
        
        # Visualize data and related target class 
        analysis_object.data_visualization()
        
        # Convert categorical attributes
        analysis_object.convert_categorical_attributes()
        
        # Plot correlation between attributes and target class
        analysis_object.print_correlation_plot()
        
        # Build prediction model and generate generate predictions
        analysis_object.build_model_generate_predictions()
        
        # Evaluate model performance and plot actual and predicted demand
        analysis_object.evaluate_model_predictions()
        
#%% Main Method call    
if __name__ == "__main__":
     main()        


