import aiohttp
import asyncio
import csv
import json

SEARCH_URL = 'https://openlibrary.org/search.json'
GET_URL = 'http://openlibrary.org/api/get'
CHUNK_SIZE = 250

INPUT_PATH = "C:\\Users\\Anthony\\Desktop\\book-recommendation-ml\\dataset\\Books2.csv"
OUTPUT_PATH = "C:\\Users\\Anthony\\Desktop\\book-recommendation-ml\\dataset\\bookCategory2.json"

# Increase this to the number of concurrent requests you want to handle
MAX_CONCURRENT_REQUESTS = 400  # adjust based on your testing and server's rate limits

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def return_empty_list():
    return []

async def fetch_with_semaphore(sem, session, url, params=None):
    async with sem:
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    print(f"Error {response.status} fetching {url}")
                    return None  # or however you want to handle this
                content_type = response.headers.get('Content-Type')
                if 'application/json' not in content_type:
                    print(f"Unexpected content type {content_type} for {url}")
                    return None  # or however you want to handle this
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    print(f"Failed to decode JSON from {url}")
                    return None
        except aiohttp.ClientError as e:
            print(f"Error fetching {url}. Error: {e}")
            return None

async def fetch_key_by_title(session, title):
    params = {'title': title, 'limit': 1}
    data = await fetch_with_semaphore(semaphore, session, SEARCH_URL, params)
    if data and 'docs' in data and data['numFound'] > 0 and 'key' in data['docs'][0]:
        return data['docs'][0]['key']
    return None


async def fetch_subject_by_key(session, key):
    params = {'key': key}
    data = await fetch_with_semaphore(semaphore, session, GET_URL, params)
    
    if data is None:
        return []
    
    if 'result' in data and 'subjects' in data['result']:
        return data['result']['subjects']
    return []

async def process_books_chunk(session, books):
    titles = [book['Book-Title'] for book in books]
    keys = await asyncio.gather(*(fetch_key_by_title(session, title) for title in titles))

    # Filter out books with None keys
    valid_books = [book for book, key in zip(books, keys) if key]
    valid_keys = [key for key in keys if key]

    # Fetch subjects for valid keys
    subjects_list = await asyncio.gather(*(fetch_subject_by_key(session, key) for key in valid_keys))

    for book, subjects in zip(valid_books, subjects_list):
        book["Category"] = ";".join(subjects)

    return valid_books

async def main():
    all_results = []

    async with aiohttp.ClientSession() as session:
        with open(INPUT_PATH, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            books = [
                {
                    'ISBN': row['ISBN'],
                    'Book-Title': row['Book-Title'],
                    'Book-Author': row['Book-Author'],
                    'Year-Of-Publication': row['Year-Of-Publication'],
                    'Publisher': row['Publisher'],
                    'Image-URL-S': row['Image-URL-S'],
                    'Image-URL-M': row['Image-URL-M'],
                    'Image-URL-L': row['Image-URL-L']
                }
                for row in reader
            ]
            total_books = len(books)

            for i in range(0, total_books, CHUNK_SIZE):
                chunk = books[i:i+CHUNK_SIZE]
                chunk_results = await process_books_chunk(session, chunk)
                all_results.extend(chunk_results)
                print(f"Processed {i + len(chunk)}/{total_books} books.")

    try:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4)
    except Exception as e:
        print(f"Error while writing to JSON: {e}")

asyncio.run(main())
