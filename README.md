# AI-Driven Fruit Quality Assessment and Optimization System
**Author:** Meina Liu  
**Date:** Fall Quarter 2023

## Background on the Apple Problem
I know our favorite part of 109 lecture is receiving a little prize for our questions. Whether it be chocolate, oranges, apples, and the list goes on! But, we are college students, and with that comes a forgetfulness to eat many of these fun treats as they often go days on our desk or in our fridge, waiting to be eaten on the day the dining hall fails us. But, how do we know if that food is still good?

In my project, I aim to predict the expiration status of these treats, specifically apples, based on two features: the number of days since bought/awarded and the storage location. These predictions are made using a Naive Bayes classifier.

### Maximum Likelihood Estimation (MLE)
From lecture, we know that MLE is a method for estimating the parameters of a statistical model. In the context of our apple expiration problem, MLE helps us determine the probability distribution for the freshness of apples given observed features. The key formula for MLE that we will be utilizing is:
L(θ) = ∏(i=1 to n) f(X_i|θ)

Corresponding to this formula, the following Python code snippet showcases the estimation of prior and likelihood probabilities:

```python
# Calculate prior probabilities
priors = labels.value_counts(normalize=True).to_dict()

# Calculate likelihood probabilities with Laplace smoothing
feature_probs = {}
for feature in features.columns:
    feature_probs[feature] = {}
    for value in features[feature].unique():
        prob_given_label = {}
        for label in labels.unique():
            count = features[(features[feature] == value) & (labels == label)].shape[0]
            prob = (count + 1) / (labels.value_counts()[label] + features[feature].nunique())
            prob_given_label[label] = prob
        feature_probs[feature][value] = prob_given_label
```
   The Naive Bayes theorem allows us to make predictions using the calculated probabilities. It is expressed as: P(Y|X) = (P(X|Y)P(Y)) / P(X)
## Test Data Formatting

Now, I believe it is important to discuss how I created the data sets for the training data and the testing data. The training data set was designed to reflect realistic scenarios of apple storage and expiration. Features were encoded to represent days after purchase and storage conditions, with labels indicating whether the apple was expired. 

This is because the date bought would keep track of when the apple was purchased or awarded to an individual throughout the year. The location stored was important as supermarkets state that there is a difference in shelf life when kept in the fridge or kept on the counter tops (or in our case the desk). Finally, the number of days after it was bought is included as that determines if it has passed its shelf-life as provided by general supermarket stores. If so, the status would be marked as expired. If not, it would be marked as not expired. 

In doing so, this allows us to categorize the apples depending on how long it's been to train the algorithm on whether after a certain amount of time and depending on its location, it is expired or not. To do so, I created this dataset by hand by utilizing a random number generator ranging from 0-365 and an online dice to determine if it was kept in the fridge or the counter.

## The Algorithm and Data Modification

Now, to take this a step further, I feel as if there is a general assumption within the mass population that you need to have good data in order for there to be effective machine learning algorithms, but I wanted to challenge this notion. This led me to explore how making purposeful errors in the training dataset can alter the accuracy of the same algorithm applied to data sets. 

As such, I had two data sets I evaluated: the first data-set with 100% accuracy in its status as compared to the second with purposeful errors.

### First Dataset Values

The training data set was designed to reflect realistic and accurate scenarios of apple storage and expiration. This data set indicated that apples stored in the fridge and kept for ≤ 56 days were not expired, and any longer, they would be classified as expired. Similarly, apples stored on the counter and kept for ≤ 7 days were not expired, and any longer, they would be classified as expired. 

This was determined by [Healthline: How Long Do Apples Last?](https://www.healthline.com/nutrition/how-long-do-apples-last#shelf-life).

### Second Dataset with Errors

To simulate the impact of incorrect data on model performance, a controlled number of errors were introduced into the training data set. Inaccuracies were added in increments starting from 10% inaccuracies to 25%, and finally 30%. This experiment was designed to measure the robustness of the Naive Bayes classifier for apple expiration dates against data imperfections.

## Statistical Significance of the Results

We observed a change in the accuracy of our Naive Bayes classifier after introducing errors into the training data. To evaluate whether this change in accuracy is due to random chance or is statistically significant, we calculated p-values using a two-proportion z-test.

### Calculating P-Values

I included the Python function I used below to calculate the p-values:

```python
from scipy.stats import norm

# Function to calculate the z-statistic and p-value for the change in accuracy
def calculate_p_value(accuracy_before, accuracy_after, n_before, n_after):
    p1 = accuracy_before
    p2 = accuracy_after
    p_pool = (p1 * n_before + p2 * n_after) / (n_before + n_after)

    z_stat = (p1 - p2) / (p_pool * (1 - p_pool) * (1 / n_before + 1 / n_after))**0.5
    p_value = norm.sf(abs(z_stat)) * 2  # two-tailed test
    return z_stat, p_value

