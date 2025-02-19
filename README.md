SITNovate


Team: TENSOR TITANS


Problem statement: DA01


SALES FORECASTING FOR RETAIL:


Predict future sales using time_series analysis (ARIMA, Prophet) on historical data. Identify trends and seasonal patterns for inventory planning.

 Data Collection  - 

 
Data collection is done 




That initially contained around 20 Million records.


DATA PREPROCESSING: 


Outliers detection and removal:


The dataset choose had (19_454_838, 13) records and after outlier detection we have (12_560_367, 13) records.



Heatmap and removal of missing values:




From this heatmap we can see that there are certain irrelevant columns, thus the irrelevant columns are removed. The columns removed are as follows: 

'promo_type_1', 'promo_bin_1', 
    'promo_type_2', 'promo_bin_2', 
    'promo_discount_2', 'promo_discount_type_2'

And now we will work on ARIMA, ML and Data visualisations.
Time: 5.00 pm


Round 2:

Data Preprocessing was done so shifted towards predictive analysis and data visualization.

Figures 1-4 represent visualised data

Started using Prophet


Created 2 datasets and forecasting predictions for them one is for big data and one is small data

Time 8:00 pm

Round 3:

PowerBI is done

Tried using ARIMA,Prophet and other techniques but did not work well

NeuralProphet was picked for reliability

r^2 value is 0.267 have to improve it

Analysis(powerbi) on the big data will be done

Time : 11:00pm

Round 4:


Power BI for both the dataset is done.


Worked on the 10k Dataset using ARIMA 


Started working on simpler base line models like Lightgbm and Random Forest Regressor


R^2 value of Neural Prophet had improved


If we remove the outliers the target values become constant as the outliers removed represented the peak and through values of the sales. So we have decided to keep the outliers.


Time: 1:53 am


Round 5:

Made the R^2 and other parameters as optimal,MAE,MSE were also good. Done using GridSearchCV and HyperParameter Tuning

Graph is also coming nicely. Need the graph of all models

Worked on baseline models like xgb

ARIMA still needs work

Worked on documentation of the entire project

Trying and making it scalable and simple at the same time







