import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from pymongo import MongoClient, errors


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "data-center"
TABLE_NAME = "Articles"


def crawl_articles(base_url):
    article_urls = []
    try:
        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "articles" in href:
                full_url = urljoin(base_url, href)
                article_urls.append(full_url)

        print(f"Found {len(article_urls)} article(s) on {base_url}.")
        return article_urls

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {base_url}: {e}")
        return []


def save_articles(articles, base_url):
    client = MongoClient(MONGO_URI)

    docs = [
        {
            "link": url,
            "baseUrl": base_url,
            "status": "not processed yet",
        }
        for url in articles
    ]

    try:
        result = client[DB_NAME][TABLE_NAME].insert_many(docs, ordered=False)
        print(f"Inserted {len(result.inserted_ids)} new articles.")

    except errors.BulkWriteError as bwe:
        inserted = bwe.details.get("nInserted", 0)
        print(
            f"Inserted {inserted} new articles; {len(docs) - inserted} were duplicates."
        )


if __name__ == "__main__":
    base_url = "https://www.nature.com"

    articles = crawl_articles(base_url)
    print(f"Discovered {len(articles)} unique article URLs.")

    save_articles(articles, base_url)
