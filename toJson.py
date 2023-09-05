import pandas as pd
import os

# Define the paths for CSV files and the output directory
ratings_csv_path = r"C:\Users\Anthony\Desktop\book-recommendation-ml\dataset\initialData\Ratings.csv"
users_csv_path = r"C:\Users\Anthony\Desktop\book-recommendation-ml\dataset\initialData\Users.csv"
output_dir = r"C:\Users\Anthony\Desktop\book-recommendation-ml\dataset"

# Read CSV files
ratings_df = pd.read_csv(ratings_csv_path)
users_df = pd.read_csv(users_csv_path)

# Convert dataframes to JSON
ratings_json = ratings_df.to_json(orient="records")
users_json = users_df.to_json(orient="records")

# Save JSON files
ratings_json_path = os.path.join(output_dir, "Ratings.json")
users_json_path = os.path.join(output_dir, "Users.json")

with open(ratings_json_path, "w") as ratings_json_file:
    ratings_json_file.write(ratings_json)

with open(users_json_path, "w") as users_json_file:
    users_json_file.write(users_json)

print("Conversion to JSON complete!")
