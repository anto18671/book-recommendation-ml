import json
import pandas as pd
import math

# File paths
json_books = r'C:\Users\Anthony\Desktop\book-recommendation-ml\dataset\processedData\processedBookCategory.json'
ratings_json = r'C:\Users\Anthony\Desktop\book-recommendation-ml\dataset\processedData\Ratings.json'   
users_json = r'C:\Users\Anthony\Desktop\book-recommendation-ml\dataset\processedData\Users.json'     

# Read JSON data
with open(json_books, 'r') as file:
    books_data = json.load(file)
books_df = pd.DataFrame(books_data)

with open(ratings_json, 'r') as file:       
    ratings_data = json.load(file)
ratings_df = pd.DataFrame(ratings_data)

with open(users_json, 'r') as file:         
    users_data = json.load(file)
users_df = pd.DataFrame(users_data)

# Print initial counts
print(f"Initial number of ratings: {len(ratings_df)}")
print(f"Number of unique users: {users_df['User-ID'].nunique()}")
print(f"Number of unique books: {books_df['ISBN'].nunique()}")

# Process the data
books_df = books_df.applymap(lambda s: s.lower() if type(s) == str else s)
users_df = users_df.applymap(lambda s: s.lower() if type(s) == str else s)

# Merge ratings with users and capture the number of ratings after the merge
merged_with_users = pd.merge(ratings_df, users_df, on='User-ID', how='inner')

# Merge the result with books data
merged_final = pd.merge(merged_with_users, books_df, on='ISBN', how='inner')

# Further processing steps
average_rating = merged_final[merged_final['Book-Rating'] != 0]['Book-Rating'].mean()
merged_final.loc[merged_final['Book-Rating'] == 0, 'Book-Rating'] = math.ceil(average_rating)
cols_to_drop = ['Location', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
merged_final.drop(columns=cols_to_drop, inplace=True)
average_age = merged_final[(merged_final['Age'] >= 12) & (merged_final['Age'] <= 99)]['Age'].mean()
merged_final['Age'].fillna(value=math.floor(average_age), inplace=True)
merged_final.dropna(inplace=True)

# Shuffle the dataframe
merged_final = merged_final.sample(frac=1).reset_index(drop=True)

# File path for the processed dataset
processed_data_output = r'C:\Users\Anthony\Desktop\book-recommendation-ml\dataset\ProcessedData.json'

# Save the dataset
merged_final.to_json(processed_data_output, orient='records', lines=True)

# Print confirmation message
print(f"Processed data has been saved to {processed_data_output}.")
print(f"Total number of transactions in the processed data: {merged_final.shape[0]}")
