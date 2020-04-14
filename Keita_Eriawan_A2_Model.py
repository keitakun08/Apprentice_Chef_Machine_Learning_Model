# timeit

# Student Name : Keita Eriawan
# Cohort       : MSBA2 - Haight

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np

# Scikit-learn packages #
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import roc_auc_score
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import make_scorer
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.tree            import DecisionTreeClassifier


################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# Changing the FOLLOWED_RECOMMENDATIONS_PCT column into a percentage format #

apprentice_chef['FOLLOWED_RECOMMENDATIONS_PCT'] = apprentice_chef['FOLLOWED_RECOMMENDATIONS_PCT'] / 100

# Number of meal prepared that follows the recommendation #

apprentice_chef['TOTAL_MEAL_FOLLOWED_RECOMMENDATIONS'] = apprentice_chef['TOTAL_MEALS_ORDERED'] * apprentice_chef['FOLLOWED_RECOMMENDATIONS_PCT']

# Round the value into 2 decimal point #

apprentice_chef['TOTAL_MEAL_FOLLOWED_RECOMMENDATIONS'] = apprentice_chef['TOTAL_MEAL_FOLLOWED_RECOMMENDATIONS'].round(0)

# Number of meal prepared that does not follows the recommendation #

apprentice_chef['TOTAL_MEAL_NOT_FOLLOWED_RECOMMENDATIONS'] = apprentice_chef['TOTAL_MEALS_ORDERED'] - apprentice_chef['TOTAL_MEAL_FOLLOWED_RECOMMENDATIONS']

# Average price for one meal per month #

apprentice_chef['AVERAGE_MEAL_PRICE_PER_MONTH'] = apprentice_chef['REVENUE'] / apprentice_chef['TOTAL_MEALS_ORDERED']

# Rounding of to 2 decimal points #

apprentice_chef['AVERAGE_MEAL_PRICE_PER_MONTH'] = apprentice_chef['AVERAGE_MEAL_PRICE_PER_MONTH'].round(2)

# Average total meal ordered per month #

apprentice_chef['AVERAGE_MEAL_ORDERED_PER_MONTH'] = apprentice_chef['TOTAL_MEALS_ORDERED'] / 12

# Rounding off to 2 decimal points #

apprentice_chef['AVERAGE_MEAL_ORDERED_PER_MONTH'] = apprentice_chef['AVERAGE_MEAL_ORDERED_PER_MONTH'].round(2)

# Total number of cancellation #

apprentice_chef['TOTAL_CANCELLATION'] = apprentice_chef['CANCELLATIONS_BEFORE_NOON'] + apprentice_chef['CANCELLATIONS_AFTER_NOON']

# Total number of login #

apprentice_chef['TOTAL_LOGIN'] = apprentice_chef['MOBILE_LOGINS'] + apprentice_chef['PC_LOGINS']

# Adding the number of early deliveries and late deliveries to create total number of special deliveries #
apprentice_chef['TOTAL_SPECIAL_DELIVERIES'] = apprentice_chef['EARLY_DELIVERIES'] + apprentice_chef['LATE_DELIVERIES']

# Creating the total number of normal deliveries #
apprentice_chef['TOTAL_NORMAL_DELIVERIES'] = apprentice_chef['TOTAL_MEALS_ORDERED'] - apprentice_chef['TOTAL_SPECIAL_DELIVERIES']

# Creating average customer engagement per site visit #

apprentice_chef['AVG_CUSTOMER_ENGAGEMENT'] = apprentice_chef['PRODUCT_CATEGORIES_VIEWED'] / apprentice_chef['AVG_CLICKS_PER_VISIT']

# Rounding it to 2 decimal places #
apprentice_chef['AVG_CUSTOMER_ENGAGEMENT'] = apprentice_chef['AVG_CUSTOMER_ENGAGEMENT'].round(2)

### Splitting email names: name and domain ###

# Placeholder list #
placeholder_lst = []

# Creating a for loop to separate the domain and email name #
for index, col in apprentice_chef.iterrows():
    split_email = apprentice_chef.loc[index, 'EMAIL'].split(sep = '@')
    placeholder_lst.append(split_email)

# Creating a new dataframe for the separate email name and domain #
email_df = pd.DataFrame(placeholder_lst)

### Concatenating the new email domain with the  original dataset ###

# Renaming the column #
email_df.columns = ['email_name', 'email_domain']

# Concatenating email_domain with apprentice_chef df #
apprentice_chef = pd.concat([apprentice_chef, email_df.loc[ : , 'email_domain']],
                           axis=1)

# Checking the counts on each domain #
apprentice_chef.loc[ : , 'email_domain'].value_counts()

### Categorising the email type by domain ###

# Creating a list of domain types #

personal_email_domains = ['@gmail.com', '@yahoo.com', '@protonmail.com']
junk_email_domains = ['@me.com','@aol.com', '@hotmail.com', '@live.com', '@msn.com',
                    '@passport.com']

# placeholder list #
placeholder_lst = []

# Creating a loop for categorising email domain #
for domain in apprentice_chef['email_domain']:
    if '@' + domain in personal_email_domains:
        placeholder_lst.append('personal')
    elif '@' + domain in junk_email_domains:
        placeholder_lst.append('junk')
    else:
        placeholder_lst.append('professional')

# Concatenating with original DataFrame #
apprentice_chef['DOMAIN_GROUP'] = pd.Series(placeholder_lst)

## Creating a dummy variable for the email domain ###

#one hot encoding categorical variables #
one_hot_domain_group = pd.get_dummies(apprentice_chef['DOMAIN_GROUP'])

# Joining codings together #
apprentice_chef = apprentice_chef.join([one_hot_domain_group])

# saving new columns #
new_columns = apprentice_chef.columns

#one hot encoding categorical variables #
one_hot_median_meal_rating = pd.get_dummies(apprentice_chef['MEDIAN_MEAL_RATING'])

# Joining codings together #
apprentice_chef = apprentice_chef.join([one_hot_median_meal_rating])

# saving new columns #
new_columns = apprentice_chef.columns

# Renaming the column names #
apprentice_chef.rename(columns={1:'VERY_LOW_MEAL_RATING',
                                2:'LOW_MEAL_RATING',
                                3:'NEUTRAL_MEAL_RATING',
                                4:'HIGH_MEAL_RATING',
                                5:'VERY_HIGH_MEAL_RATING'},
                                inplace = True)

# Creating the outliers threshold #

REVENUE_hi                      = 3000
TOTAL_MEALS_ORDERED_hi          = 150
UNIQUE_MEALS_PURCH_hi           = 7
CONTACTS_W_CUSTOMER_SERVICE_lo  = 4
CONTACTS_W_CUSTOMER_SERVICE_hi  = 9
AVG_TIME_PER_SITE_VISIT_hi      = 180
CANCELLATIONS_BEFORE_NOON_hi    = 1
CANCELLATIONS_AFTER_NOON_hi     = 2
MOBILE_LOGINS_lo                = 5
MOBILE_LOGINS_hi                = 6
PC_LOGINS_lo                    = 1
PC_LOGINS_hi                    = 2
WEEKLY_PLAN_hi                  = 15
EARLY_DELIVERIES_hi             = 2
LATE_DELIVERIES_hi              = 5
AVG_PREP_VID_TIME_hi            = 200
LARGEST_ORDER_SIZE_lo           = 2
LARGEST_ORDER_SIZE_hi           = 6
MASTER_CLASSES_ATTENDED_hi      = 2
AVG_CLICKS_PER_VISIT_lo         = 10
TOTAL_PHOTOS_VIEWED_hi          = 500
TOTAL_PHOTOS_VIEWED_lo          = 0

### Creating the outlier columns  ###

# REVENUE #

apprentice_chef['OUT_REVENUE'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_REVENUE'][apprentice_chef['REVENUE']
                                                                 > REVENUE_hi]

apprentice_chef['OUT_REVENUE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# TOTAL_MEALS_ORDERED #

apprentice_chef['OUT_TOTAL_MEALS_ORDERED'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_TOTAL_MEALS_ORDERED'][apprentice_chef['TOTAL_MEALS_ORDERED']
                                                                 > TOTAL_MEALS_ORDERED_hi]

apprentice_chef['OUT_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# UNIQUE_MEALS_PURCH #

apprentice_chef['OUT_UNIQUE_MEALS_PURCH'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_UNIQUE_MEALS_PURCH'][apprentice_chef['UNIQUE_MEALS_PURCH']
                                                                 > UNIQUE_MEALS_PURCH_hi ]

apprentice_chef['OUT_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE #

apprentice_chef['OUT_CONTACTS_W_CUSTOMER_SERVICE'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE']
                                                                         > CONTACTS_W_CUSTOMER_SERVICE_hi]
condition_lo = apprentice_chef.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][apprentice_chef['CONTACTS_W_CUSTOMER_SERVICE']
                                                                         < CONTACTS_W_CUSTOMER_SERVICE_lo]

apprentice_chef['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

apprentice_chef['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# AVG_TIME_PER_SITE_VISIT #

apprentice_chef['OUT_AVG_TIME_PER_SITE_VISIT'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_AVG_TIME_PER_SITE_VISIT'][apprentice_chef['AVG_TIME_PER_SITE_VISIT']
                                                                         > AVG_TIME_PER_SITE_VISIT_hi]

apprentice_chef['OUT_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# CANCELLATIONS_BEFORE_NOON #

apprentice_chef['OUT_CANCELLATIONS_BEFORE_NOON'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][apprentice_chef['CANCELLATIONS_BEFORE_NOON']
                                                                         > CANCELLATIONS_BEFORE_NOON_hi]

apprentice_chef['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# CANCELLATIONS_AFTER_NOON #

apprentice_chef['OUT_CANCELLATIONS_AFTER_NOON'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_CANCELLATIONS_AFTER_NOON'][apprentice_chef['CANCELLATIONS_AFTER_NOON']
                                                                         > CANCELLATIONS_AFTER_NOON_hi]

apprentice_chef['OUT_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)


# EARLY_DELIVERIES #

apprentice_chef['OUT_EARLY_DELIVERIES'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_EARLY_DELIVERIES'][apprentice_chef['EARLY_DELIVERIES']
                                                                 > EARLY_DELIVERIES_hi]

apprentice_chef['OUT_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)
# LATE_DELIVERIES #

apprentice_chef['OUT_LATE_DELIVERIES'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_LATE_DELIVERIES'][apprentice_chef['LATE_DELIVERIES']
                                                                 > LATE_DELIVERIES_hi]

apprentice_chef['OUT_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# AVG_PREP_VID_TIME #

apprentice_chef['OUT_AVG_PREP_VID_TIME'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_AVG_PREP_VID_TIME'][apprentice_chef['AVG_PREP_VID_TIME']
                                                                 > AVG_PREP_VID_TIME_hi]

apprentice_chef['OUT_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# LARGEST_ORDER_SIZE #

apprentice_chef['OUT_LARGEST_ORDER_SIZE'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_LARGEST_ORDER_SIZE'][apprentice_chef['LARGEST_ORDER_SIZE']
                                                                         > LARGEST_ORDER_SIZE_hi]
condition_lo = apprentice_chef.loc[0:,'OUT_LARGEST_ORDER_SIZE'][apprentice_chef['LARGEST_ORDER_SIZE']
                                                                         < LARGEST_ORDER_SIZE_lo]

apprentice_chef['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

apprentice_chef['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# MASTER_CLASSES_ATTENDED #

apprentice_chef['OUT_MASTER_CLASSES_ATTENDED'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_MASTER_CLASSES_ATTENDED'][apprentice_chef['MASTER_CLASSES_ATTENDED']
                                                                 > MASTER_CLASSES_ATTENDED_hi]

apprentice_chef['OUT_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)


# AVG_CLICKS_PER_VISIT #

apprentice_chef['OUT_AVG_CLICKS_PER_VISIT'] = 0

# Creating the condition for outliers #
condition_lo = apprentice_chef.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][apprentice_chef['AVG_CLICKS_PER_VISIT']
                                                                         < AVG_CLICKS_PER_VISIT_lo]

apprentice_chef['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# TOTAL_PHOTOS_VIEWED #

apprentice_chef['OUT_TOTAL_PHOTOS_VIEWED'] = 0

# Creating the condition for outliers #
condition_hi = apprentice_chef.loc[0:,'OUT_TOTAL_PHOTOS_VIEWED'][apprentice_chef['TOTAL_PHOTOS_VIEWED']
                                                                         > TOTAL_PHOTOS_VIEWED_hi]
condition_lo = apprentice_chef.loc[0:,'OUT_TOTAL_PHOTOS_VIEWED'][apprentice_chef['TOTAL_PHOTOS_VIEWED']
                                                                         < TOTAL_PHOTOS_VIEWED_lo]

apprentice_chef['OUT_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

apprentice_chef['OUT_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# Dropping unnecessary variables #

apprentice_chef = apprentice_chef.drop('EMAIL', axis = 1)
apprentice_chef = apprentice_chef.drop('FIRST_NAME', axis = 1)
apprentice_chef = apprentice_chef.drop('FAMILY_NAME', axis = 1)
apprentice_chef = apprentice_chef.drop('email_domain', axis = 1)
apprentice_chef = apprentice_chef.drop('DOMAIN_GROUP', axis = 1)
################################################################################
# Train/Test Split
################################################################################

# train/test split with the significant full model #
apprentice_chef_data   =  apprentice_chef.loc[ : , candidate_dict['logit_sig']]
apprentice_chef_target =  apprentice_chef.loc[ : , 'CROSS_SELL_SUCCESS']

# Creating the default SEED #
SEED = 222

# Train_test_split the data #
X_train, X_test, y_train, y_test = train_test_split(
            apprentice_chef_data,
            apprentice_chef_target,
            test_size    = 0.25,
            random_state = SEED,
            stratify     = apprentice_chef_target)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model

# Instantiating the GradientBoostingClassifier #
gbm_tuned = GradientBoostingClassifier(loss = 'exponential',
                                       criterion ='friedman_mse',
                                       learning_rate = 0.1,
                                       max_depth     = 1,
                                       n_estimators  = 100,
                                       random_state  = SEED)


# Fitting the dataset #
gbm_tuned_fit = gbm_tuned.fit(X_train, y_train)


# Predicting based on the testing set #
gbm_tuned_pred = gbm_tuned_fit.predict(X_test)


# Printing the results #
print('Training ACCURACY:', gbm_tuned_fit.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', gbm_tuned_fit.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = gbm_tuned_pred).round(4))

################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = roc_auc_score(y_true  = y_test,
                           y_score = gbm_tuned_pred).round(4)
