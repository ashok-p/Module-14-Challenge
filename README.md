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
* Save the plot in a PNG format to use it as a baseline for comparison with the variations of this model and other models.

### Tune the Baseline Trading Algorithm
#### Tune the training algorithm by adjusting the size of the training dataset 
* Run the notebook with 1, 2, 3, 6, 10 month training data and save the cumulative return plots
* Print the respective classification reports
* Review the plots against the Baseline Algorithm `(3 month training data, Short_window=4, Long_window=100)`

If you look at the  [cumulative returns plots](ML_Models_1.pdf) for 1, 2, 3, 6, 10 month training data, it appears that by increasing the training window the accuracy does improve very slightly from 55 to 56%. If you reduce from the baseline level of 3 month training to 1 or 2 months, the results are similar. Also, the recall values for positve signal is very high at around 98% while the recall values are terrible for predicting -1 regardless of the training period. The cumulative return plots seem to indicate fairly good predictability when compared with the actual returns at 2 months and 10 months training intervals.

#### Tune the trading algorithm by adjusting the SMA input features 

* Beginning at `short_window=2, long_window=50` to `short_window=10, long_window=100`, created 9 sets of training data.
* Run the notebook for each of the training data sets and save the cumulative return plots
* Print the respective classification reports
* Review the plots against the Baseline Algorithm `(3 month training data, Short_window=4, Long_window=100)`
  
Looking at the [cumulative returns plots](ML_Models_2.pdf) for various short and long SMA values, it appears that the best performance is at `short_window=10, long_window=100`. Below this and above this the performance seems to deteriorate rapidly as you move away from this set, be it short_window or long_window tuning alone or together. 


#### Choose the set of parameters that best improved the trading algorithm returns
The SMA window values of `short_window=10 and long_window=100` seemed to work the best in terms of cumulative returns. 

### Evaluate a New Machine Learning Classifier   

* Import `DecisionTreeRegressor` Classifier
* Use the orginal baseline data 
* Fit the model
* Make prediction based upon the test data
* Print Classification Report
* Plot the Actual and Strategy cumulative returns

Repeat the above steps for `RandomForestClassifier` and `AdaBoostClassifier`

Ran 3 different classifiers. following are the [results](ML_Models_3.pdf). 

`DecisionTreeRegressor` performed very poorly against the SVC model.

`RandomForestClassifier` seemed to be better in predicting -1s (34%) and not so well +1s (66%) when compared with the SVC model (1s at around 98% and -1 in single digits!) . Also, the accuracy was 52% against SVC at around 55-56%. The cumulative returns plots were very poor. 

`AdaBoostClassifier` fared similar to `RandomForestClassifier`. 

None of the new models provided a better alternative. But, it appears that if you train the `AdaBoostClassifier` or `RandomForestClassifier` models, there is a possibility we can get a better model. But only after extensive training and testing.

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