from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
    
class EmbarkedImputer(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        return None
    def fit(self, X):
        return self  # nothing else to do
    def transform(self, X):
        # deep copy the df
        df = X.copy()
        
        # Clean up fares.
        value_to_input = df.loc[(df['Fare'] < 85) & (df['Fare'] > 75)  & (df['Pclass'] == 1)]['Embarked'].mode()

        value_to_input = value_to_input[0]

        df.loc[(df['Embarked'].isnull()),['Embarked']] = value_to_input

        return(df)
    
    
class GeneralImputer(BaseEstimator, TransformerMixin):
    def __init__(self, col_impute, col_group, impute_method = 'median'): # no *args or **kargs
        self.col_impute = col_impute
        self.col_group = col_group
        self.impute_method = impute_method
        return None
    def fit(self, X):
        return self  # nothing else to do
#     def transform(self, X, col_impute, col_group, impute_method = 'median'):
    def transform(self, X):
        # deep copy the df because of transform
        df = X.copy()

        # Create a groupby object: by_sex_class
        grouped = df.groupby(self.col_group)

        # function to impute median
        def imputer_median(series):
            return series.fillna(series.median())
        # function to impute average
        def imputer_average(series):
            return series.fillna(series.mean())

        if self.impute_method == 'median':
            # impute median
#             print(type(grouped[col_impute].transform(imputer_median)))
            df[self.col_impute] = grouped[self.col_impute].transform(imputer_median)
            return(df)
        elif self.impute_method == 'average':
            # impute average
#             print(type(grouped[col_impute].transform(imputer_average)))
            df[self.col_impute] = grouped[self.col_impute].transform(imputer_average)
            return(df)
        else:
            return np.nan

        
class TitleCreator(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        return None
    def fit(self, X):
        return self  # nothing else to do
    def transform(self, X):
        # deep copy the df because of transform
        df = X.copy()

        df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev','Sir','Jonkheer','Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        df['Title'] = df['Title'].fillna(np.nan) 

        return(df)
    
#######
# GeneralImputer Example

# dd = titfun.create_sample_df(nrow=22)
# dd
# col_impute=['one'] 
# col_group=['three','four']

# print(dd)

# general_imputer_2 = transformer_classes.GeneralImputer()
# dd2 = general_imputer_2.transform(X=dd ,col_impute=['one'], col_group=['three','four'], method='average')

# print(dd2)