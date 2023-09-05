import json
import string
from collections import Counter

def tokenize(text):
    # Replace punctuation with a space and make lowercase
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.lower().translate(translator)

    # Tokenize by splitting and filter out short words
    tokens = [word for word in text.split() if len(word) > 2]
    
    return tokens

def process_category(tokens, popular_words_set):
    # Keep tokens that are among the 2000 most popular words
    tokens = [token for token in tokens if token in popular_words_set]

    # Count the frequency of each term in the filtered list
    counter = Counter(tokens)

    # Get the 6 most common terms from the filtered list
    most_common = [item[0] for item in counter.most_common(6)]

    # If there are fewer than 6 terms, fill the remaining with 'NOTHING'
    while len(most_common) < 6:
        most_common.append('NOTHING')
        
    return most_common

def main():
    input_file = "C:\\Users\\Anthony\\Desktop\\book-recommendation-ml\\dataset\\bookCategory.json"
    output_file = "C:\\Users\\Anthony\\Desktop\\book-recommendation-ml\\dataset\\processedBookCategory.json"

    # Load the data
    with open(input_file, 'r') as f:
        data = json.load(f)

    global_counter = Counter()
    processed_data = []

    # First pass: Tokenize all categories to count global popularity
    for entry in data:
        tokens = tokenize(entry['Category'])
        global_counter.update(tokens)

    # Identify the 2000 most popular words
    popular_words = [item[0] for item in global_counter.most_common(2000)]
    popular_words_set = set(popular_words)

    # Second pass: Process the categories considering only the popular words
    for entry in data:
        tokens = tokenize(entry['Category'])
        processed_category = process_category(tokens, popular_words_set)

        # Create a new entry with processed category
        new_entry = entry.copy()
        new_entry['Category'] = processed_category

        processed_data.append(new_entry)

    # Create a mapping for 'NOTHING' and popular words to integers
    word_to_int = {'NOTHING': 0}
    for idx, word in enumerate(popular_words, start=1):
        word_to_int[word] = idx

    # Replace words in processed_data with their respective integers
    for entry in processed_data:
        entry['Category'] = [word_to_int[word] for word in entry['Category']]

    # Save the processed data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

    # Print the number of popular words (should be 2000 or fewer)
    print(f"Number of popular words: {len(popular_words_set)}")

if __name__ == "__main__":
    main()
