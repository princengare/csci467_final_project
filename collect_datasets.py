import os, csv, requests
from bs4 import BeautifulSoup
import lyricsgenius
import tweepy
import glob

os.makedirs("data", exist_ok=True)

BASE_URL = "https://www.poetryfoundation.org"
SEARCH_URL = f"{BASE_URL}/search?query=love&page="

def get_poem_links(pages=5):
    links = []
    for i in range(1, pages+1):
        r = requests.get(SEARCH_URL + str(i))
        soup = BeautifulSoup(r.content, "html.parser")
        for a in soup.select("a.c-hdgSans"):
            href = a.get("href")
            if href and "/poems/" in href:
                links.append(BASE_URL + href)
    return list(set(links))

def scrape_poem(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    title = soup.find("h1").text.strip() if soup.find("h1") else ""
    text_tag = soup.find("div", {"class": "o-poem"})
    text = text_tag.get_text(separator="\n").strip() if text_tag else ""
    return title, text

print("Scraping Poetry Foundation...")
poems = []
for link in get_poem_links(10):
    try:
        title, text = scrape_poem(link)
        poems.append({"title": title, "text": text, "label": "romantic"})
    except:
        continue

with open("data/poetry_foundation.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["title", "text", "label"])
    writer.writeheader()
    writer.writerows(poems)

print("Scraping Genius Lyrics...")
genius = lyricsgenius.Genius("0rvZMokYYjWeeX1p0lKjnwmHKbuYYeRSVryvcjwdx6lEJhq83xTfEL-Mv7RZcFzF", timeout=15)
artists = ["Adele", "Ed Sheeran", "Taylor Swift"]
songs = []

for artist in artists:
    try:
        results = genius.search_artist(artist, max_songs=10, sort="popularity")
        for song in results.songs:
            if "love" in song.title.lower() or "love" in song.lyrics.lower():
                songs.append({
                    "artist": artist,
                    "title": song.title,
                    "text": song.lyrics,
                    "label": "romantic"
                })
    except Exception as e:
        print(f"Skipping {artist} due to error: {e}")

with open("data/genius_lyrics.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["artist", "title", "text", "label"])
    writer.writeheader()
    writer.writerows(songs)

print("Scraping Twitter...")
client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAAFFW1AEAAAAAUe%2FW67L1be8J2SEBALwAqV8b8hU%3Dz68VQJk7ISA9yu3RGL9w45SUJYALJIjxQKLrJnc83eBlePDBMH")
query = "#LoveYou -is:retweet lang:en"
tweets = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["text"])

tweet_data = []
for tweet in tweets.data:
    tweet_data.append({"text": tweet.text, "label": "romantic"})

with open("data/twitter_data.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(tweet_data)

def extract_love_passages(file_path, label="romantic"):
    with open(file_path, encoding='utf-8') as f:
        text = f.read()
    passages = text.split("\n\n")
    love_passages = [p for p in passages if "love" in p.lower()]
    return [{"text": p.strip(), "label": label} for p in love_passages[:50]]

print("Parsing Multilingual Literature...")
all_texts = []
for file in glob.glob("gutenberg/*.txt"):
    all_texts.extend(extract_love_passages(file))

with open("data/multilingual_lit.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(all_texts)

print("Parsing OPUS Dialogue...")
opus_file = "opus/opus_love.tsv"
pairs = []
with open(opus_file, encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if "love" in line.lower():
            source = line.strip().split('\t')[0]
            pairs.append({"text": source, "label": "romantic"})

with open("data/opus_dialogue.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(pairs)

print("\n All datasets collected and saved in /data")


