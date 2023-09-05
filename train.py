# === IMPORTS ===
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU

# === CONSTANTS ===
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 0.0008
MAX_CATEGORY_LENGTH = 4
DROPOUT_RATE = 0.3

TRAINING_DATA_PATH = "dataset/ProcessedData.json"
MODEL_SAVE_PATH = "model/book_rating_predictor.h5"
VALIDATION_SPLIT = 0.2

# Filter future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# === DATA LOADING AND PREPROCESSING FUNCTIONS ===

def load_data():
    data_df = pd.read_json(TRAINING_DATA_PATH, lines=True)
    train_df = data_df.sample(frac=1-VALIDATION_SPLIT, random_state=42)
    validation_df = data_df.drop(train_df.index)
    features_train, book_ratings_train, features_val, book_ratings_val, feature_columns = preprocess_data(train_df, validation_df)
    return features_train, book_ratings_train, features_val, book_ratings_val

def save_label_encoders(label_encoders, path="label_encoders.pkl"):
    with open(path, "wb") as f:
        pickle.dump(label_encoders, f)

def load_label_encoders(path="label_encoders.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_features_and_target(df, feature_columns, target_column):
    features = df[feature_columns]
    
    # Convert the 'Category' list of integers into multiple columns
    def pad_category(category_list):
        arr = np.zeros(MAX_CATEGORY_LENGTH)
        for i, category in enumerate(category_list[:MAX_CATEGORY_LENGTH]):
            arr[i] = category
        return arr

    categories = np.stack(features['Category'].apply(pad_category).to_numpy())
    category_df = pd.DataFrame(categories, columns=[f"Category_{i}" for i in range(MAX_CATEGORY_LENGTH)], index=df.index)
    
    features = pd.concat([features, category_df], axis=1)
    return features, df[target_column]

def handle_unseen_labels(train_series, val_series):
    # Get unique values from train series
    train_unique_values = set(train_series.unique())
    
    # Replace values in validation set that are not in training set with a placeholder
    val_series = val_series.apply(lambda x: x if x in train_unique_values else "-unseen-")
    
    return train_series, val_series

def preprocess_data(data_df, validation_df):
    # Extract features and target
    feature_columns = ['User-ID', 'Book-Title', 'Age', 'Book-Author', 'Category']
    target_column = 'Book-Rating'
    
    features_train, book_ratings_train = extract_features_and_target(data_df, feature_columns, target_column)
    features_val, book_ratings_val = extract_features_and_target(validation_df, feature_columns, target_column)

    # Encode categorical columns
    label_encoders = {}
    for col in ['User-ID', 'Book-Title', 'Book-Author']:
        le = LabelEncoder()
        
        # Handle unseen labels in the validation set
        features_train[col], features_val[col] = handle_unseen_labels(features_train[col], features_val[col])
        
        # Add 'UNKNOWN' to the training set and fit the encoder
        le.fit(pd.concat([features_train[col].astype(str), pd.Series("UNKNOWN")]))
        
        # Transform the training data
        features_train[col] = le.transform(features_train[col].astype(str))
        
        # Replace "-unseen-" with "UNKNOWN" for validation set and transform
        features_val[col] = features_val[col].replace("-unseen-", "UNKNOWN")
        features_val[col] = le.transform(features_val[col].astype(str))
        
        label_encoders[col] = le
    
    # Remove the 'Category' list column, as it has been expanded to constant-length columns.
    features_train.drop(columns='Category', inplace=True)
    features_val.drop(columns='Category', inplace=True)

    return features_train, book_ratings_train, features_val, book_ratings_val, feature_columns

def preprocess_new_data(new_data_df, label_encoders, feature_columns):
    # Extract features
    features, _ = extract_features_and_target(new_data_df, feature_columns, target_column=None)
    
    # Handle and encode categorical columns
    for col in ['User-ID', 'Book-Title', 'Book-Author']:
        # Replace unseen values with 'UNKNOWN' (which was added when the encoder was originally fitted)
        features[col] = features[col].apply(lambda x: x if x in label_encoders[col].classes_ else "UNKNOWN")
        
        # Transform the data
        features[col] = label_encoders[col].transform(features[col].astype(str))
    
    # Remove the 'Category' list column
    features.drop(columns='Category', inplace=True)

    return features

# === MODEL BUILDING AND TRAINING FUNCTIONS ===

def build_model(input_shape, layers=[768, 384, 192]):
    model = Sequential()
    
    # Input layer
    model.add(Dense(layers[0], input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(DROPOUT_RATE))
    
    # Hidden layers
    for units in layers[1:]:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(DROPOUT_RATE))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    model.add(keras.layers.Lambda(lambda x: x * 9 + 1))
    
    return model

def compile_and_train(model, features_train, book_ratings_train, features_val, book_ratings_val):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                  loss='mean_squared_error', metrics=['mae'])
    history = model.fit(features_train, book_ratings_train, 
                        validation_data=(features_val, book_ratings_val), 
                        epochs=EPOCHS, batch_size=BATCH_SIZE)
    return history

# === PLOTTING UTILITY ===

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# === MAIN FUNCTION ===

def main():
    features_train, book_ratings_train, features_val, book_ratings_val = load_data()
    model = build_model((features_train.shape[1],))
    history = compile_and_train(model, features_train, book_ratings_train, features_val, book_ratings_val)
    plot_training_history(history)
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()