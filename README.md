# Module-14-Challenge - Machine Learning Trading Bot
As one of the top financial advisory firms in the world, your firm wants to develop a cutting edge machine learning trading bot to maintain the edge over her competition.

The Starter Code was provided along with the historical data.

---

## User Story
Your firm, one of the top five financial advisory firms in the world, constantly competes with the other major firms to manage and automatically trade assets in a highly dynamic environment. In recent years, your firm has heavily profited by using computer algorithms that can buy and sell faster than human traders.

The speed of these transactions gave your firm a competitive advantage early on. But, people still need to specifically program these systems, which limits their ability to adapt to new data. You’re thus tasked to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, you’ll enhance the existing trading signals with machine learning algorithms that can adapt to new data.

---

## Acceptance Criteria  
The application must meet the following acceptance criteria:  

* Establish a Baseline Performance
* Tune the Baseline Trading Algorithm
* Evaluate a New Machine Learning Classifier
* Create an Evaluation Report

---

## The Application  

Created a ML library **ml_lib** so I could play with different variables for tuning purposes and avoid duplicating the code. I have commented out sections of the Starter Code that was provided and coded the application using these library methods

The **ml_lib** contains `create_train_test_dataframes, scale_features_df, model_fit_predict`, and `get_Strategy_and_Actual_returns` methods.

The application uses these methods as appropriate.

### Establish a Baseline Performance

#### Prepare the Data
* Import the OHLCV dataset into a Pandas DataFrame.
* Generate trading signals using short- and long-window SMA values.
* Split the data into training and testing datasets.
* Scale the data using StandardScaler

#### Use SVC Classifier Model 
* Create a SVC classifier model
* Fit the training data 
* Predict based upon the testing data
* Review the classification report 
* Create a predictions DataFrame with “Predicted” values, “Actual Returns”, and “Strategy Returns” columns
* Plot the 'Actual Returns' and 'Strategy Returns'
* Save the plot [Baseline plot](baseline.png) for comparison with the variations of this model and other models.  

The classification report gave the following baseline results:  
        
                    precision    recall  f1-score   support

               -1.0     0.43      0.04      0.07      1804
                1.0     0.56      0.96      0.71      2288
    
        accuracy                            0.55      4092  
    
       macro avg        0.49      0.50      0.39      4092
    weighted avg        0.50      0.55      0.43      4092
 
 Accuracy of 55% indiactes that there is a slightly better chance than simply flipping a coin. The model  
 is good at predicting 1s at 96% but very poor at prediciting -1s at just 4%!  The [Baseline plot](baseline.png) shows  
 the predicted cumulative returns against the actual returns with higher returns than actuals predicted consistently since mid 2018.

### Tune the Baseline Trading Algorithm
#### Tune the training algorithm by adjusting the size of the training dataset 
* Run the notebook with 1, 2, 3, 6, 10 month training data and save the cumulative return plots
* Print the respective classification reports

1 (ONE) MONTH 

              precision    recall  f1-score   support

        -1.0       0.37      0.03      0.06      1828
         1.0       0.56      0.96      0.70      2324

    accuracy                           0.55      4152
    macro avg       0.47      0.49      0.38     4152
    weighted avg    0.48      0.55      0.42     4152


2 (TWO) MONTH

              precision    recall  f1-score   support

        -1.0       0.39      0.04      0.06      1825
         1.0       0.56      0.96      0.70      2318

    accuracy                           0.55      4143
    macro avg      0.47      0.50      0.38      4143
    weighted avg   0.48      0.55      0.42      4143

3 (THREE) MONTH - BASELINE

              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
    macro avg      0.49      0.50      0.39      4092
    weighted avg   0.50      0.55      0.43      4092

6 (SIX) MONTH

              precision    recall  f1-score   support

        -1.0       0.44      0.02      0.04      1732
         1.0       0.56      0.98      0.71      2211

    accuracy                           0.56      3943
    macro avg      0.50      0.50      0.38      3943
    weighted avg   0.51      0.56      0.42      3943

10 (TEN) MONTH

              precision    recall  f1-score   support

        -1.0       0.43      0.03      0.05      1549
         1.0       0.57      0.97      0.72      2017

    accuracy                           0.56      3566
    macro avg      0.50      0.50      0.38      3566
    weighted avg   0.51      0.56      0.43      3566


* Review the [cumulative returns plots](ML_Models_1.pdf) against the Baseline Algorithm `(3 month training data, Short_window=4, Long_window=100)`

If you look at the  [cumulative returns plots](ML_Models_1.pdf) for 1, 2, 3, 6, 10 month training data, it appears that by increasing the training window the accuracy does improve very slightly from 55 to 56%. If you reduce from the baseline level of 3 month training to 1 or 2 months, the results are similar. Also, the recall values for positve signal is very high at around 98% while the recall values are terrible for predicting -1 regardless of the training period. The cumulative return plots seem to indicate fairly good predictability when compared with the actual returns at 2 months and 10 months training intervals.

#### Tune the trading algorithm by adjusting the SMA input features 

* Beginning at `short_window=2, long_window=50` to `short_window=10, long_window=100`, created 9 sets of training data.
* Run the notebook for each of the training data sets and save the cumulative return plots
* Print the respective classification reports

SHORT WINDOW = 2, LONG_WINDOW = 50, Month Offset = 3
              precision    recall  f1-score   support

        -1.0       0.39      0.07      0.12      1826
         1.0       0.56      0.92      0.69      2321

    accuracy                           0.54      4147
    macro avg      0.48      0.49      0.40      4147
    weighted avg   0.49      0.54      0.44      4147

SHORT WINDOW = 2, LONG_WINDOW = 100, Month Offset = 3

              precision    recall  f1-score   support

        -1.0       0.41      0.01      0.03      1804
         1.0       0.56      0.99      0.71      2288

    accuracy                           0.56      4092
    macro avg      0.49      0.50      0.37      4092
    weighted avg   0.49      0.56      0.41      4092

SHORT WINDOW = 3, LONG_WINDOW = 100, Month Offset = 3
              precision    recall  f1-score   support

        -1.0       0.43      0.03      0.05      1549
         1.0       0.57      0.97      0.72      2017

    accuracy                           0.56      3566
    macro avg      0.50      0.50      0.38      3566
    weighted avg   0.51      0.56      0.43      3566


SHORT WINDOW = 4, LONG_WINDOW = 100, Month Offset = 3

              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
    macro avg      0.49      0.50      0.39      4092
    weighted avg   0.50      0.55      0.43      4092

SHORT WINDOW = 10, LONG_WINDOW = 120, Month Offset = 3

              precision    recall  f1-score   support

        -1.0       0.46      0.02      0.04      1793
         1.0       0.56      0.98      0.71      2284

    accuracy                           0.56      4077
    macro avg      0.51      0.50      0.38      4077
    weighted avg   0.51      0.56      0.42      4077



SHORT WINDOW = 10, LONG_WINDOW = 150, Month Offset = 3
              precision    recall  f1-score   support

        -1.0       0.58      0.01      0.02      1791
         1.0       0.56      1.00      0.72      2278

    accuracy                           0.56      4069
    macro avg      0.57      0.50      0.37      4069
    weighted avg   0.57      0.56      0.41      4069

SHORT WINDOW = 10, LONG_WINDOW = 175, Month Offset = 3
              precision    recall  f1-score   support

        -1.0       0.46      0.04      0.07      1745
         1.0       0.56      0.96      0.71      2232

    accuracy                           0.56      3977
    macro avg      0.51      0.50      0.39      3977
    weighted avg   0.52      0.56      0.43      3977

SHORT WINDOW = 10, LONG_WINDOW = 200, Month Offset = 3
              precision    recall  f1-score   support

        -1.0       0.45      0.11      0.17      1740
         1.0       0.56      0.89      0.69      2227

    accuracy                           0.55      3967
    macro avg      0.50      0.50      0.43      3967
    weighted avg   0.51      0.55      0.46      3967

* Review the [cumulative returns plots](ML_Models_2.pdf) against the Baseline Algorithm `(3 month training data, Short_window=4, Long_window=100)`
  
Looking at the [cumulative returns plots](ML_Models_2.pdf) for various short and long SMA values, it appears that the best performance is at `short_window=10, long_window=100`. Below this and above this the performance seems to deteriorate rapidly as you move away from this set, be it short_window or long_window tuning alone or together. The accurtacy seems to hover just around 50%, with the highest at 56%. The models are extremely good at predicting 1s at high 90s% but fail at detecting -1s at below 12%, with most numbers in single digits!

#### Choose the set of parameters that best improved the trading algorithm returns
The SMA window values of `short_window=10 and long_window=100` seemed to work the best in terms of cumulative returns with the training window offset at `3 months`. 

### Evaluate a New Machine Learning Classifier   

* Import `DecisionTreeRegressor` Classifier
* Use the orginal baseline data 
* Fit the model
* Make prediction based upon the test data
* Print Classification Report
* Plot the Actual and Strategy cumulative returns

Repeat the above steps for `RandomForestClassifier` and `AdaBoostClassifier`
Classification Reports:
`DecisionTreeRegressor`
              precision    recall  f1-score   support

        -1.0       0.44      0.67      0.53      1793
         1.0       0.56      0.34      0.42      2284

    accuracy                           0.48      4077
    macro avg      0.50      0.50      0.48      4077
    weighted avg   0.51      0.48      0.47      4077

`RandomForestClassifier`
              precision    recall  f1-score   support

        -1.0       0.44      0.94      0.60      1793
         1.0       0.54      0.05      0.10      2284

    accuracy                           0.44      4077
    macro avg      0.49      0.50      0.35      4077
    weighted avg   0.49      0.44      0.32      4077

`AdaBoostClassifier`
              precision    recall  f1-score   support

        -1.0       0.44      0.94      0.60      1793
         1.0       0.57      0.06      0.10      2284

    accuracy                           0.45      4077
    macro avg      0.50      0.50      0.35      4077
    weighted avg   0.51      0.45      0.32      4077

Ran 3 different classifiers. following are the [results](ML_Models_3.pdf). 

`DecisionTreeRegressor` seemed to be better in predicting -1s (67%) and not so well +1s (34%) when compared with the SVC model (1s at around 98% and -1 in single digits!) . Also, the accuracy was 48% against SVC at around 55-56%. The cumulative returns plots were very poor. 

`AdaBoostClassifier` fared similar to `RandomForestClassifier` performed very poorly against the SVC model. Both did well in predicting -1s in 90%s and fared terribly at predicitng 1s at 5/6%. The accuracy was also in mid 40%s.

None of the new models provided a better alternative. But, it appears that if you train the `AdaBoostClassifier` or `RandomForestClassifier` models, there is a possibility we can get a better model. But only after extensive training and testing.

### Evaluation Report

The SVC model performed the best overall, with the predicted cumulative returns more in line with the actual returns. The alternate models performed very poorly against this metric when compared with the SVC model, as is evident from the various plots.

With the SVC model, at SMA parameters of `short_window=10 and long_window=100` it seemed to work the best in terms of cumulative returns with the training window offset at `3 months`. The biggest problem seemed to be with predicting -1s which is not resolved with any of the number of training parameters. 

None of the new models provided a better alternative. But, it appears that if you train the `DecisionTreeRegressor` model, there is a possibility (only after extensive training and testing) that we can get a better model since this is the only case where both 1 and -1s predictions had more balanced numbers at 34 and 67% respectively compared to complete polarized results in SVC model..

---

## Technologies
The application is developed using:  
* Language: Python 3.7,   
* Packages/Libraries: Pandas; StandardScaler from sklearn.preprocessing; AdaBoostClassifier, RandomForestClassifier from sklearn.ensemble;  DecisionTreeRegressor from 'tree' at sklearn.
* Development Environment: VS Code and Terminal, Anaconda 2.1.1 with conda 4.11.0, Jupyterlab 3.2.9
* OS: Mac OS 12.1

---
## Installation Guide
Following are the instructions to install the application from its Github respository.  

### Clone the application code from Github as follows:
copy the URL link of the application from its Github repository      
open the Terminal window and clone as follows:  

   1. %cd to_your_preferred_directory_where_you want_to_store_this_application  
    
   2. %git clone URL_link_that_was_copied_in_step_1_above   
    
   3. %ls     
        Module-14-Challenge    
        
   4. %cd Module-14-Challenge     

At this point you will have the the entire application files in the current directory as follows:
    * ML_models_1.pdf                          Image files
    * ML_models_2.pdf                          Image files
    * ML_models_3.pdf                          Image files
    * README.md                               (this file you are reading)
    * Starter_Code
        - machine_learning_trading_bot.ipynb  (modified starter code)
        - ml_lib.py                           (libraray created following DRY code)
    * Resources
    * Resources                      (Folder with the data required) 
        - emerging_markets_ohlcv.csv  
       
---

## Usage
The following details the instructions on how to run the application.  

### Setup the environment to run the application 

Setup the environment using conda as follows:

    5. %conda create dev -python=3.7 anaconda  
    
    6. %conda activate dev  
    
    7. %jupyter lab  

---

#### Run the Application
THIS ASSUMES FAMILIARITY WITH JUPYTER LAB. If not, then [click here for information on Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/).  

After step 7 above, this will take you to the jupyter lab window, where you can open the application notebook **machine_learning_trading_bot.ipynb** and run the application. Note that the application is in **Starter Code** folder. 

**NOTE**:
>Your shell prompt will look something like __(dev) ashokpandey@Ashoks-MBP dir%__ ,  with:  
    - '(dev)' indicating the activated 'dev' environment,   
    - ' ashokpandey@Ashoks-MBP ' will be different for you as per your environment, and   
    - 'dir' directory is where the application is located.  
    - '%' sign is the shell prompt - it may be a dollar sign in your implementation 

---

## Contributors
Ashok Pandey - ashok.pragati@gmail.com   
www.linkedin.com/in/ashok-pandey-a7201237

---

## License
The source code is the property of the developer. The users can copy and use the code freely but the developer is not responsible for any liability arising out of the code and its derivatives.

---