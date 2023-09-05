# book-recommendation-ml

Overview
--------

This project aims to build a robust book recommendation system using machine learning. The system leverages user ratings and book metadata to provide accurate and personalized book recommendations.

Repository Structure
--------------------

- `train.py`: The heart of the machine learning model. This script trains a neural network to predict book ratings based on various features.
- `cleanMerge.py`: A data preprocessing script that cleans, merges, and processes the initial datasets.
- `countDifference.py`: A utility script to identify discrepancies between JSON and CSV datasets.
- `genreScrapping.py`: A web scraping script that fetches genre or category information for books from the OpenLibrary API.
- `toJson.py`: A conversion script that transforms CSV datasets into JSON format for easier processing.
- `tokenize-script.py`: A script dedicated to tokenizing and processing the 'Category' attribute of books.

Detailed Script Descriptions
----------------------------

### genreScrapping.py:

This script is a testament to the importance of data in machine learning. It's designed to enrich the dataset by fetching genre or category information for books from the OpenLibrary API. 

- **Asynchronous Programming**: Utilizes the power of asynchronous programming to handle multiple requests efficiently.
- **Data Fetching**: Sends requests to the OpenLibrary API based on book titles to retrieve genre information.
- **Data Merging**: Identifies books present in the CSV dataset but missing from the JSON dataset and appends them.
- **Output**: Saves the enriched dataset back to a JSON file.

### train.py:

This script is where the magic happens. It trains a machine learning model to predict book ratings, ensuring that the recommendations are as accurate as possible.

- **Data Handling**: Loads processed data from a JSON file, splits it into training and validation sets, and preprocesses it.
- **Neural Network**: Constructs a deep neural network with multiple layers, batch normalization, ReLU activation, and dropout.
- **Training**: Uses the Adam optimizer and mean squared error loss for training. The training progress is visualized using plots of loss and mean absolute error over epochs.

Prerequisites
-------------

- Python 3.x
- Libraries: pandas, numpy, tensorflow, sklearn, aiohttp, asyncio
- A passion for books and machine learning!

Installation & Setup
--------------------

1. Clone the repository: `git clone https://github.com/anto18671/book-recommendation-ml.git`
2. Navigate to the project directory: `cd book-recommendation-ml`
3. Install the required libraries: `pip install -r requirements.txt` (Note: You might need to create this file with the required libraries listed)

Usage
-----

1. Preprocess the dataset: `python cleanMerge.py`
2. Enrich the dataset with genre information: `python genreScrapping.py`
3. Convert datasets to JSON format: `python toJson.py`
4. Tokenize and process the 'Category' attribute: `python tokenize-script.py`
5. Train the model: `python train.py`

Contributing
------------

We welcome contributions! If you have an idea or improvement:

1. Fork the repository.
2. Create a new branch with a descriptive name.
3. Make your changes.
4. Submit a pull request.

For major changes, please open an issue first to discuss what you'd like to change.

License
-------

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code, but please acknowledge the source.

Acknowledgements
----------------

A big shoutout to the OpenLibrary API for providing genre data and Kaggle for the initial data.

