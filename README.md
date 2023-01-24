# ACM Research coding challenge (Spring 2023)

![image](https://user-images.githubusercontent.com/72369124/211179527-0ee60624-2794-4e13-bf7f-f88b5c950e44.png)

## Project Overview

This project is an evaluation of machine learning models trained on the star type data set. I haven't had much practice working with tabular data so this was a welcome challenge. I've created a fairly flexible solution to predict the numerical and categorical features of the dataset.

## Methodology

The machine learning framework I chose to use for this project was **XGBoost** due to its excellent performance with classification and regression.

A standard first step is to perform EDA on the dataset.
The most significant insights I gained were that the dataset is quite small and that the dataset is imbalanced for the Star color and Spectral Class features. This was something to keep in mind, as it could cause overfitting to the majority category in each feature.

Since the Star type feature has the least imbalance, I decided to use it as my initial target value to test the model, then ran similar tests on the other categorical features. I evaluated the models using the standard **SKLearn** metrics, with the highest focus on accuracy, since it's the most intuitive for classification. This can all be found in *star-classification.ipynb*. Changing *target_value* will change what the model targets for prediction.

As an additional challenge, I also attempted to use XGBoost regression to predict star temperature values and the other numerical features. I evaluated these using R2 scores, since they are a fairly common and useful metric for regression. This can all be found in *star-regression.ipynb*.

## Results

Overall the results were excellent. The following were my highest prediction statistics for each category:

Star color

                     precision    recall  f1-score   support

               Blue       0.89      1.00      0.94        16
         Blue White       0.00      0.00      0.00         1
         Blue white       1.00      0.00      0.00         2
         Blue-white       1.00      0.80      0.89         5
                Red       1.00      1.00      1.00        21
              White       0.00      1.00      0.00         0
    Yellowish White       0.00      1.00      0.00         0
              white       1.00      0.00      0.00         1
       yellow-white       1.00      1.00      1.00         2

           accuracy                           0.90        48
          macro avg       0.65      0.64      0.43        48
       weighted avg       0.94      0.90      0.89        48

Spectral Class

                  precision    recall  f1-score   support

               A       0.75      1.00      0.86         3
               B       1.00      0.90      0.95        10
               F       1.00      0.50      0.67         2
               K       1.00      1.00      1.00         1
               M       1.00      1.00      1.00        26
               O       0.86      1.00      0.92         6

        accuracy                           0.96        48
       macro avg       0.93      0.90      0.90        48
    weighted avg       0.97      0.96      0.96        48

Star Type

                  precision    recall  f1-score   support

               0       1.00      1.00      1.00         7
               1       1.00      1.00      1.00         8
               2       1.00      1.00      1.00        10
               3       1.00      1.00      1.00         6
               4       1.00      1.00      1.00        13
               5       1.00      1.00      1.00         4

        accuracy                           1.00        48
       macro avg       1.00      1.00      1.00        48
    weighted avg       1.00      1.00      1.00        48

Overall the accuracy is quite high, but there is evidence of overfitting. The high support scores for M/B spectral class and Red/Blue colors indicate these values were weighted much more heavily than the others. The small size of the dataset is also likely a contributing factor. On the other hand, the Star Type feature prediction is perfect. This may be due to the fact that it's very balanced and the dataset is small enough that it was able to satisfy all the test variables. I expect the perfect accuracy wouldn't hold up with a larger test set.

For the regression model, my R2 score for temperature predictions was around 0.80 on average, which is fairly high. Here are the average scores for the other numerical features:

    Temperature: 0.80
    Luminosity: 0.53
    Radius: 0.95
    Absolute Magnitude: 0.98

The scores largely correspond to the range of values the model needs to consider for each feature, with a larger range corresponding to a lower accuracy. This is consistent with the temperature results and makes sense considering the small size of the dataset. Additionally, I didn't report with a very high level of precision since the scores tended to fluctuate significantly for some features. The scores likely would have been higher with a more comprehensive dataset.

## Sources

The star dataset itself is sourced from kaggle from user Deepraj Baidya:
https://www.kaggle.com/datasets/deepu1109/star-dataset

XGBoost Documentation:
https://xgboost.readthedocs.io/en/stable/

SKLearn Documentation:
https://scikit-learn.org/0.21/documentation.html

Credit to user Jefferson Santos on StackOverflow for the LabelEncoder solution used in the star classification portion of this project:
https://stackoverflow.com/a/72132612