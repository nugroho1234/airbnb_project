
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

#importing machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor



def clean_price(df, col):
    '''
    Input : dataframe to be cleaned (df), and column to be cleaned(col) indicating price
    Output : cleaned dataframe with col dtype = float (extract only the number)
    '''
    df[col] = df[col].str.replace('$', '')
    df[col] = df[col].str.replace(',', '')
    df[col] = df[col].astype('float')
    return df



def clean_percentage(df, col):
    '''
    Input : dataframe to be cleaned (df), and column to be cleaned(col) indicating percentage
    Output : cleaned dataframe with col dtype = float (eliminate '%')
    '''
    df[col] = df[col].str.replace('%', '')
    df[col] = df[col].astype('float')
    return df



def clean_dataframe(df, drop_cols):
    '''
    Input = listing dataframe
    Output = Semi cleaned dataframe with price and percentage values possess float dtype, thumbnail_available column,
    verification_method column, and imputed values for city, market and cleaning_fee column

    '''
    #drop columns if the columns exist in dataframe
    for col in drop_cols:
        try:
            df.drop(col, axis = 1, inplace = True)
        except:
            continue

    #cleaning thumbnail_available
    df['thumbnail_available'] = np.where(df['thumbnail_url'].isnull(),'yes','no')
    try:
        df.drop('thumbnail_url', axis = 1, inplace = True)
    except:
        df = df

    #cleaning host_since, get host_year column
    df[['host_since_year', 'month', 'date']] = df['host_since'].str.split('-', expand = True)
    df.drop(['month', 'date', 'host_since'], axis = 1, inplace = True)
    df['host_since_year'].fillna(0, inplace = True)
    df['host_year'] = 2018 - df.host_since_year.astype('int')
    df.drop(['host_since_year'], axis = 1, inplace = True)

    #cleaning price-related columns
    price_col = ['weekly_price', 'monthly_price', 'cleaning_fee', 'extra_people', 'price']
    for col in price_col:
        df = clean_price(df, col)

    #filling NaN values in weekly_price and monthly_price
    df['weekly_price'].fillna(0, inplace = True)
    df['price_per_week'] = np.where(df['weekly_price'] == 0, df['price']*7, df['weekly_price'])
    df.drop('weekly_price', axis = 1, inplace = True)
    df['monthly_price'].fillna(0, inplace = True)
    df['price_per_month'] = np.where(df['monthly_price'] == 0, df['price']*30, df['monthly_price'])
    df.drop('monthly_price', axis = 1, inplace = True)

    #creating verification_method column
    df['verification_method'] = df['host_verifications'].str.count(',') + 1
    df.drop(['host_verifications'], axis = 1, inplace = True)

    #cleaning percentage-related coumns
    percentage_col = ['host_response_rate', 'host_acceptance_rate']
    for col in percentage_col:
        df = clean_percentage(df, col)

    #filling NaN values for city, market, and cleaning_fee
    df['city'].fillna(df['city'].mode()[0], inplace=True)
    df['market'].fillna(df['market'].mode()[0], inplace=True)
    df['cleaning_fee'].fillna(0, inplace=True)
    return df



def predict_zip(df, master_df):
    '''
    Input ---- a dataframe with neighbourhood_cleansed and zipcode columns in the dataframe
    Output --- a dataframe with zipcode already predicted
    '''

    #creating a new dataframe, get dummies on neighbourhood_cleansed, drop that column
    df_zip = df[['neighbourhood_cleansed', 'zipcode']].copy()
    df_zip = pd.concat([df_zip, pd.get_dummies(df_zip['neighbourhood_cleansed'])], axis = 1)
    df_zip.drop('neighbourhood_cleansed', axis = 1, inplace = True)

    #divide the dataframe into a dataframe containing NaN values for zipcode and other that doesn't
    df_zip_pred = df_zip[df_zip.isnull().any(axis = 1)].copy()
    df_zip.dropna(inplace = True)

    #create training and testing set using df_zip
    print("Creating training and testing dataset...")
    X = df_zip.drop('zipcode', axis = 1)
    y = df_zip['zipcode']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    print("DONE!")

    #using decision tree classifier to predict zipcode
    print("Creating decision tree classifier model...")
    tree_mod = DecisionTreeClassifier()
    tree_mod.fit(X_train, y_train)
    y_pred = tree_mod.predict(X_test)
    print("DONE!")
    print("F1 score for this model is: ",f1_score(y_test, y_pred, average = 'micro'))
    print("Predicting Zip Code...")

    #using the model to predict the NaN values in zipcode
    df_zip_pred.drop('zipcode', axis = 1, inplace = True)
    df_zip_pred['zipcode_fix'] = tree_mod.predict(df_zip_pred)
    print("DONE!")

    #concat the predicted zipcode to original dataframe,
    #rename the column into zipcode_fix, drop zipcode from original dataframe
    zipcode_fix = pd.concat([df_zip['zipcode'], df_zip_pred['zipcode_fix']])
    df = pd.concat([master_df, zipcode_fix], axis = 1)
    df.rename(columns = {0: 'zipcode_fix'}, inplace = True)
    df.drop('zipcode', axis = 1, inplace = True)
    return df



def rf_type(X_train, X_test, y_train, y_test):
    '''
    Input: This function inputs training and testing sets from X and y variables
    Output: This function prints f1 score for training and testing phase, and returns best model to be used to predict..
    '''

    clf = RandomForestClassifier(random_state=42)

    parameters = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [5, 10, 15],
                  'max_depth': [20, 50, 100],
                  'min_samples_leaf':[2,4],
                  'min_samples_split': [20,30,40]}

    #create scorer using f1 score
    scorer = make_scorer(f1_score, average = 'micro')

    #conducting gridsearch
    grid_obj = GridSearchCV(clf, parameters, scoring = scorer)
    grid_fit = grid_obj.fit(X_train, y_train)
    #obtaining best model, fit it to training set
    best_clf = grid_fit.best_estimator_
    best_clf.fit(X_train, y_train)

    # Make predictions using the new model.
    best_train_predictions = best_clf.predict(X_train)
    best_test_predictions = best_clf.predict(X_test)

    # Calculate the f1_score of the new model.
    print('The training F1 Score is', f1_score(best_train_predictions, y_train, average = 'micro'))
    print('The testing F1 Score is', f1_score(best_test_predictions, y_test, average = 'micro'))
    return best_clf



def predict_property_type(df, master_df, cols_to_clean):
    '''
    Input: a copy of master_df
    Output: a dataframe with single column containing predicted values of property type
    '''
    df = master_df.copy()
    #dropping unneeded variables
    df.drop(cols_to_clean, axis = 1, inplace = True)
    #obtaining columns that will be used to onehotencode
    cat_vars = df.select_dtypes(include = ['object']).columns
    cols_to_predict = ['property_type', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating']
    cols_to_dummy = [x for x in cat_vars if x not in cols_to_predict]
    #creating dummy variable for categorical variables
    for col in cols_to_dummy:
        df = pd.concat([df.drop(col, axis = 1), pd.get_dummies(df[col], prefix = col, prefix_sep = '_', drop_first = True)], axis = 1)

    #create 2 dataframes, 1 is free of NaN values, the other one is loaded with NaN values
    df_to_pred = df[df.isnull().any(axis = 1)].copy()
    df.dropna(inplace = True)

    #creating training and testing sets
    X = df.drop('property_type', axis = 1)
    y = df['property_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    #using rf_type to get the best model
    clf = rf_type(X_train, X_test, y_train, y_test)
    importances = clf.feature_importances_
    std = np.std([t.feature_importances_ for t in clf.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]
    print("feature rank:")
    for i in range (X_train.shape[1]):
        print("{}. feature {} {}".format(i+1, indices[i], importances[indices[i]]))

    #print(list(zip(clf.coef_, clf.features)))
    #predicting NaN values of property_type
    df_to_pred_copy = df_to_pred.copy()
    df_to_pred.dropna(subset = ['property_type'], inplace = True)
    property_type_pred = df_to_pred_copy[df_to_pred_copy['property_type'].isnull()].copy()
    property_type_data = df_to_pred_copy[pd.notnull(df_to_pred_copy['property_type'])].copy()
    num_vars = property_type_pred.select_dtypes(include = ['float64', 'int64', 'int32']).columns

    for var in num_vars:
        property_type_pred[var].fillna(property_type_pred[var].mean(), inplace = True)

    #obtaining a single column dataframe containing property type
    property_type_pred.drop(['property_type'], axis = 1, inplace = True)
    property_type_pred['property_type'] = clf.predict(property_type_pred)
    property_type_fix = pd.concat([property_type_pred['property_type'], property_type_data['property_type']])
    property_type_cc = pd.concat([df['property_type'], property_type_fix])
    return property_type_cc



def predict_rf(df, col, master_df, cols_to_clean):

    #creating copy of master_df, dropping unneeded variables
    df = master_df.copy()
    df.drop(cols_to_clean, axis = 1, inplace = True)

    #create 2 dataframes, 1 with NaN values and 1 without NaN values
    cat_vars = df.select_dtypes(include = ['object']).columns
    for var in cat_vars:
        df = pd.concat([df.drop(var, axis = 1), pd.get_dummies(df[var], prefix = var, prefix_sep = '_', drop_first = True)], axis = 1)
    df_to_pred = df[df.isnull().any(axis = 1)].copy()
    df_clean = df.copy()
    df_clean.dropna(inplace = True)

    #create training and testing sets
    X = df_clean.drop(col, axis = 1)
    y = df_clean[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    #using random forest regresor
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    #print the MSE for model evalation
    print("MSE for {} prediction = {}".format(col, mean_squared_error(y_test, y_pred)))
    df_to_pred_copy = df_to_pred.copy()
    #creating 2 dataframes: 1 with NaN values and 1 without NaN values
    col_pred = df_to_pred_copy[df_to_pred_copy[col].isnull()].copy()
    col_data = df_to_pred_copy[pd.notnull(df_to_pred_copy[col])].copy()
    col_pred.drop(col, axis = 1, inplace = True)

    num_vars = col_pred.select_dtypes(include = ['float64', 'int64', 'int32']).columns

    for var in num_vars:
        col_pred[var].fillna(col_pred[var].mean(), inplace = True)
    #predicting the NaN values
    col_pred[col] = rf.predict(col_pred)
    col_fix = pd.concat([col_pred[col], col_data[col]])
    col_cc = pd.concat([df_clean[col], col_fix])

    return col_cc



def plot_features(master_df, y_col):
    '''
    Input: the master dataframe and a column which represents target variable
    Output: this function returns nothing, but it plots the top 3 feature importances of the random forest regressor
    '''

    #create a copy of the master dataframe and drop unneeded columns
    df = master_df.copy()
    cols_to_drop = ['id', 'host_id','picture_url', 'neighbourhood_cleansed', 'calendar_updated', 'amenities']
    df.drop(cols_to_drop, axis = 1, inplace = True)

    #one hot encode the categorical variable
    cat_vars = df.select_dtypes(include = ['object']).columns
    for var in cat_vars:
        df = pd.concat([df.drop(var, axis = 1), pd.get_dummies(df[var], prefix = var, prefix_sep = '_', drop_first = True)], axis = 1)

    #create a dataframe with target variables in it
    df_pred = df[['number_of_reviews', 'reviews_per_month', 'review_scores_rating']].copy()

    #createing X and y variables depending on which y_col we choose
    if y_col == 'review_scores_rating':
        X = df.drop([y_col], axis = 1).copy()
        y = df_pred[y_col]
    elif y_col == 'number_of_reviews' or y_col == 'reviews_per_month':
        X = df.drop(['number_of_reviews', 'reviews_per_month'], axis = 1).copy()
        y = df_pred[y_col]

    #conducting random forest regression
    rf_reg = RandomForestRegressor(random_state = 42)
    rf_reg.fit(X, y)

    #mapping feature importance
    importances = rf_reg.feature_importances_
    std = np.std([t.feature_importances_ for t in rf_reg.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]
    feat_imp = pd.DataFrame({'importance':rf_reg.feature_importances_})
    feat_imp['feature'] = X.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    #plotting feature importance
    feat_imp = feat_imp.iloc[:3]
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title='Feature Importances')
    plt.xlabel('Feature Importance Score')
    plt.show()



def wifi_available(amenities_str):
    '''
    INPUT
        amenities_str - a string of one of the values from the amenities column

    OUTPUT
        return 1 if "Wireless Internet" in amenities_str
        return 0 otherwise

    '''
    if "Wireless Internet" in amenities_str:
        return 1
    else:
        return 0



def heating_available(amenities_str):
    '''
    INPUT
        formal_ed_str - a string of one of the values from the Formal Education column

    OUTPUT
        return 1 if "Heating" in amenities_str
        return 0 otherwise

    '''
    if "Heating" in amenities_str:
        return 1
    else:
        return 0



def kitchen_available(amenities_str):
    '''
    INPUT
        formal_ed_str - a string of one of the values from the Formal Education column

    OUTPUT
        return 1 if "Kitchen" in amenities_str
        return 0 otherwise

    '''
    if "Kitchen" in amenities_str:
        return 1
    else:
        return 0
