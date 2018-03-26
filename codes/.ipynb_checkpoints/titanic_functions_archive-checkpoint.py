import pandas as pd
import numpy as np

###############################################
### HELPER FUNCTIONS FOR DATAFRAME HANDLING ###

# return list of indices which are True 
def get_true_indices(tf_list):
    return([i for i, x in enumerate(tf_list) if x == True])

# return elements of list based on true/false list of indices 
def select_list_elements(input_list, tf_list):
    # tbd: different length input_list and tf_list
    indices = tuple(get_true_indices(tf_list))
    return([input_list[i] for i in indices])

# return elements of the 1st list which are not in the 2nd list
def compare_list_elements(list_a, list_b, exclude = True):
    tf_indices = [x in list_b for x in list_a]
    
    if exclude:
        tf_indices = [not i for i in tf_indices]

    return(select_list_elements(list_a, tf_indices))

# return names of columns of the 1st data frame which are not in the 2nd data frame
def compare_df_names(df_A, df_B, exclude = True):
    a_nm = df_A.columns.tolist()
    b_nm = df_B.columns.tolist()

    return(compare_list_elements(a_nm, b_nm, exclude))

#Method for finding substrings
# taken from https://www.kaggle.com/rcasellas/ensemble-stacking-with-et-script
# perhaps rewrite to a nicer form
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan

###############################################

def fare_imputer(df):
    # Clean up fares.
    for passenger in df[(df['Fare'].isnull())].index:
        df.loc[passenger, 'Fare'] = np.average(df[(df['Fare'].notnull())]['Fare'])
        
    return(df)
        
###############################################        
        
def embarked_imputer(df):
    # Clean up fares.
    value_to_input = df.loc[(df['Fare'] < 85) & (df['Fare'] > 75)  & (df['Pclass'] == 1)]['Embarked'].mode()
  
    value_to_input = value_to_input[0]
    
    df.loc[(df['Embarked'].isnull()),['Embarked']] = value_to_input
             
    return(df)
        
###############################################   
def df_cleaner(df):
    # taken from https://www.kaggle.com/thebrocean/benchmarking-random-forests
    # rewritten to better dataframe handling
    """
    Clean up a few variables in the training/test sets.
    """

    # Clean up ages - input median value within Sex-Pclass groups
    df = impute_value(df, ['Age'], ['Sex', 'Pclass'] , method = 'median')
    

    
    # Manually convert values to numeric columns for clarity.

    # Change the sex to a binary column.
#     df.loc[df['Sex'] == 'male', 'Sex_code']    = 0
#     df.loc[df['Sex'] == 'female', 'Sex_code']  = 1
#     df.loc[df['Sex'].isnull(), 'Sex_code']     = 2
    
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2
    df.loc[df['Embarked'].isnull(), 'Embarked'] = 3
    
    # cabin mapping
    #Map and Create Deck feature
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck'] = df['Cabin'].astype(str).map(lambda x: substrings_in_string(x, cabin_list))
    
#     df.loc[df['Deck'] == 'A', 'Deck_code'] = 1
#     df.loc[df['Deck'] == 'B', 'Deck_code'] = 2
#     df.loc[df['Deck'] == 'C', 'Deck_code'] = 3
#     df.loc[df['Deck'] == 'D', 'Deck_code'] = 4
#     df.loc[df['Deck'] == 'E', 'Deck_code'] = 5
#     df.loc[df['Deck'] == 'F', 'Deck_code'] = 6
#     df.loc[df['Deck'] == 'G', 'Deck_code'] = 7
#     df.loc[df['Deck'] == 'T', 'Deck_code'] = 8
#     df["Deck"] = df["Deck"].fillna(0)

    df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

    return df

###############################################  

# imputs missing data by selected method applied to groups by selected columns
def impute_value(df, col_impute, col_group, method = 'median'):
    # tbd: use also different functions
    
    # deep copy the df because of transform
    df_2 = df.copy()
    
    # Create a groupby object: by_sex_class
    grouped = df_2.groupby(col_group)

    # Write a function that imputes median
    def imputer_median(series):
        return series.fillna(series.median())

#     print('-'*10)
#     print(df_2)

    if method == 'median':
        # impute median
        df_2[col_impute] = grouped[col_impute].transform(imputer_median)
        
#         print('-'*10)
#         print(df)
        
        return(df_2)
    else:  
        return np.nan
    
###############################################  
    
# function to create sample df
def create_sample_df(nrow=15):
    df = pd.DataFrame(np.random.randint(9,size=(nrow,2)), columns=['one', 'two'])
    df_rows = df.shape[0]
    items_xyz = ['xx', 'yy', 'zz']
    items_ab = ['aa', 'bb']
    df['three'] = np.repeat(items_xyz, (df_rows // len(items_xyz)) + 1)[0:df_rows]
    df['four'] = (items_ab*((df_rows // len(items_ab)) + 1))[0:df_rows]
    df.loc[[0,1,8],'one'] = np.nan
    return(df)

############################################### 

# define title column
# taken from https://www.kaggle.com/startupsci/titanic-data-science-solutions
def add_title_column(df):
#     title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0) 
    
    return(df)