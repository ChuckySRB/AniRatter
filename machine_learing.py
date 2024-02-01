from datetime import datetime

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def format_data_prediction(data):
    # Flatten the data into a list of dictionaries
    flattened_data = []
    for entry in data:
        for media_entry in entry.get('entries', []):
            if media_entry['score'] == 0 and media_entry['media']['meanScore'] is not None:
                flattened_data.append({
                    'title': media_entry['media']['title']['romaji'],
                    'type': media_entry['media']['type'],
                    'format': media_entry['media']['format'],
                    'seasonYear': media_entry['media']['startDate']['year'],
                    'source': media_entry['media']['source'],
                    'meanScore': media_entry['media']['meanScore'],
                })
    df = pd.DataFrame(flattened_data)
    return df

def format_data_plan(data):
    # Flatten the data into a list of dictionaries
    flattened_data = []
    for entry in data:
        for media_entry in entry.get('entries', []):
            if media_entry['score'] == 0 and media_entry['media']['meanScore'] is not None:
                flattened_data.append({
                    #'title': media_entry['media']['title']['english'],
                    'type': media_entry['media']['type'],
                    'format': media_entry['media']['format'],
                    'seasonYear': int(media_entry['media']['startDate']['year']),
                    'source': media_entry['media']['source'],
                    'genres': media_entry['media']['genres'],
                    'meanScore': int(media_entry['media']['meanScore']),
                    #'averageScore': media_entry['media']['averageScore'],
                    #'popularity': media_entry['media']['popularity'],
                    #'favourites': media_entry['media']['favourites'],
                    'tags': [tag['name'] for tag in media_entry['media']['tags'] if tag['rank'] >= 40],
                    'score': int(media_entry['score'])
                })

    # Create a DataFrame from the flattened data
    df = pd.DataFrame(flattened_data)

    # Handling Categorical Variables (One-hot encoding)
    df = pd.get_dummies(df, columns=['type', 'format', 'source'])

    # Assuming 'genres' column contains lists of genres
    df_genres = pd.get_dummies(df['genres'].apply(pd.Series), prefix='genre')
    df_genres = df_genres.loc[:, ~df_genres.columns.duplicated()]

    # Assuming 'tags' column contains lists of tags
    df_tags = pd.get_dummies(df['tags'].apply(pd.Series), prefix='tag')
    df_tags = df_tags.loc[:, ~df_tags.columns.duplicated()]

    # Concatenate the new columns to the DataFrame
    df = pd.concat([df, df_genres, df_tags], axis=1)

    # Drop the original 'genres' and 'tags' columns
    df.drop(['genres', 'tags'], axis=1, inplace=True)

    # Handling Missing Values (Fill missing numerical values with mean)
    df.fillna(df.mean(), inplace=True)

    # Normalize all numerical features
    numerical_features = ['meanScore']
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])


    return df
def format_data(data):
    # Flatten the data into a list of dictionaries
    flattened_data = []
    for entry in data:
        for media_entry in entry.get('entries', []):
            if media_entry['score'] > 0 and media_entry['media']['meanScore'] is not None:
                flattened_data.append({
                    #'title': media_entry['media']['title']['english'],
                    'type': media_entry['media']['type'],
                    'format': media_entry['media']['format'],
                    'seasonYear': int(media_entry['media']['startDate']['year']),
                    'source': media_entry['media']['source'],
                    'genres': media_entry['media']['genres'],
                    'meanScore': int(media_entry['media']['meanScore']),
                    #'averageScore': media_entry['media']['averageScore'],
                    #'popularity': media_entry['media']['popularity'],
                    #'favourites': media_entry['media']['favourites'],
                    'tags': [tag['name'] for tag in media_entry['media']['tags'] if tag['rank'] >= 40],
                    'score': int(media_entry['score'])
                })

    # Create a DataFrame from the flattened data
    df = pd.DataFrame(flattened_data)

    # Handling Categorical Variables (One-hot encoding)
    df = pd.get_dummies(df, columns=['type', 'format', 'source'])

    # Assuming 'genres' column contains lists of genres
    df_genres = pd.get_dummies(df['genres'].apply(pd.Series), prefix='genre')
    df_genres = df_genres.loc[:, ~df_genres.columns.duplicated()]

    # Assuming 'tags' column contains lists of tags
    df_tags = pd.get_dummies(df['tags'].apply(pd.Series), prefix='tag')
    df_tags = df_tags.loc[:, ~df_tags.columns.duplicated()]

    # Concatenate the new columns to the DataFrame
    df = pd.concat([df, df_genres, df_tags], axis=1)

    # Drop the original 'genres' and 'tags' columns
    df.drop(['genres', 'tags'], axis=1, inplace=True)

    # Handling Missing Values (Fill missing numerical values with mean)
    df.fillna(df.mean(), inplace=True)

    # # Normalize all numerical features
    numerical_features = ['meanScore']
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])


    return df

def split_data(data, split_ratio):
    # Split the data into features (X) and labels (y)
    X = data.drop('score', axis=1)
    y = data['score']
    # Check for duplicate columns in X
    duplicate_columns = X.columns[X.columns.duplicated()]
    if duplicate_columns.any():
        print(f"Warning: Duplicate columns found in X: {duplicate_columns}")
    else:
        print(f"Warning: NO DOUBLE in X:")
    return train_test_split(X, y, test_size=1-split_ratio, random_state=42)

def evaluate_model(X, y):
    model = RandomForestRegressor()
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
    mae_scores = -scores  # Convert negative scores to positive MAE values
    return mae_scores.mean(), mae_scores.std()


def model_results(model, test_data, original_data, X_train):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder = "predictions/"

    # Make predictions on the test data
    # Ensure X_test has the same columns as X_train without duplicates
    X_test = test_data.drop('score', axis=1).T.drop_duplicates().T.reindex(columns=X_train.columns, fill_value=False)
    predictions = model.predict(X_test)
    print(predictions)
    # Create a DataFrame with original data and predictions
    result_df = pd.DataFrame({
        'Anime Name': original_data['title'],
        'Predicted_Score': predictions,
        'Mean_Score': original_data['meanScore'],
        'Type': original_data['type'],
        'Year': original_data['seasonYear'],
        'Source': original_data['source'],
    })

    # Sort Predicted Score in descending order
    result_df = result_df.sort_values(by='Predicted_Score', ascending=False)

    # Save predictions to CSV
    file_name = f"predictions_{timestamp}.csv"
    result_df.to_csv(folder + file_name, index=False)

    print(f"Predictions saved to {folder}{file_name}")