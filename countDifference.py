import pandas as pd

# File paths
json_books_path = r"C:\Users\Anthony\Desktop\book-recommendation-ml\dataset\bookCategory.json"
csv_books_path = r"C:\Users\Anthony\Desktop\book-recommendation-ml\dataset\initialData\Books.csv"

# Load JSON books data
with open(json_books_path, 'r') as file:
    json_books_data = pd.read_json(file)
    
# Load CSV books data
csv_books_df = pd.read_csv(csv_books_path, converters={'ISBN': str}, low_memory=False)

# Find unique ISBNs in both datasets
unique_isbn_json = set(json_books_data['ISBN'].unique())
unique_isbn_csv = set(csv_books_df['ISBN'].unique())

# Find books that are in CSV but not in JSON
books_in_csv_not_json = unique_isbn_csv - unique_isbn_json
books_to_add = csv_books_df[csv_books_df['ISBN'].isin(books_in_csv_not_json)]

# Add the 'Category' attribute with an empty string value
books_to_add['Category'] = ''

# Append the missing books to the original JSON data
updated_books_data = pd.concat([json_books_data, books_to_add], ignore_index=True)

# Save the updated data back to bookCategory.json
updated_books_data.to_json(json_books_path, orient='records')

print(f"Added {len(books_in_csv_not_json)} books with 'Category' set to an empty string to {json_books_path}.")
