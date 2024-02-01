import requests
from anilist_queries import *

url = 'https://graphql.anilist.co'

def fetch_anilist_data(user, status):
    data = []
    message = "Anime Successfully Fetched"

    # Fetch anime data
    anime_response = requests.post(url, json={'query': animelist_query(username=user, status=status, type='ANIME')})
    if anime_response.status_code == 200:
        data += anime_response.json().get('data', {}).get('MediaListCollection', {}).get('lists', [])
    else:
        message = f"Error fetching anime data: {anime_response.status_code}"

    # Fetch manga data
    manga_response = requests.post(url, json={'query': animelist_query(username=user, status=status, type='MANGA')})
    if manga_response.status_code == 200:
        data += manga_response.json().get('data', {}).get('MediaListCollection', {}).get('lists', [])
    else:
        message = f"Error fetching manga data: {manga_response.status_code}"

    return data, message

def get_anilist_rated(user):
    return fetch_anilist_data(user, '_not:PLANNING')

def get_anilist_planning(user):
    return fetch_anilist_data(user, ':PLANNING')
