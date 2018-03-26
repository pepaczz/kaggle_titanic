import pandas as pd
import numpy as np

#########################################################
### HELPER FUNCTIONS FOR DATAFRAME AND LISTS HANDLING ###

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
    
# imputs missing data by selected method applied to groups by selected columns
# this function is deprecated, see the GeneralImputer in submission_RandomForestClassifier_20180307a.py file
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