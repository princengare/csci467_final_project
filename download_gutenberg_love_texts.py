import requests
import os
import time

os.makedirs("gutenberg", exist_ok=True)

book_ids = [
    # English romance and letters
    1342, 174, 31100, 345, 25305, 15865, 14533, 20469, 25051, 16389,
    12557, 16035, 15339, 28520, 29281, 35348, 43236, 58528, 52006, 17332,
    32554, 22851, 21136, 18716, 20108, 19778, 23008, 26135, 25263, 28458,
    13501, 13225, 16993, 12065, 29756, 19995, 22911, 31831, 41604, 29262,
    15558, 32343, 17463, 28106, 29085, 10455, 35520, 22818, 18032, 18581,
    # French romance/love letters
    20047, 39751, 18985, 18584, 17342, 20393, 28077, 24365, 15330, 45644,
    41917, 31865, 25036, 31361, 20032, 20781, 16974, 25195, 27977, 18956,
    23228, 26434, 19328, 33534, 25496, 18445, 21256, 15325, 23545, 26358,
    18601, 22351, 26754, 15770, 32956, 25467, 26217, 27891, 33720, 24042,
    24574, 30050, 28483, 32648, 24096, 27571, 25138, 23901, 31277, 27280
]

book_ids = list(dict.fromkeys(book_ids))[:350]

for i, book_id in enumerate(book_ids):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        print(f"[{i+1}/{len(book_ids)}] Downloading {book_id}-0.txt...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(f"gutenberg/love_{book_id}.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        time.sleep(0.5)
    except Exception as e:
        print(f"[!] Failed to download book {book_id}: {e}")

print("\n Successfully downloaded all files into /gutenberg")