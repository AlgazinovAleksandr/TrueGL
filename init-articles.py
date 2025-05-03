import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin
from pymongo import MongoClient, errors

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "truegl-database"
COLLECTION = "Articles"


def fetch_and_save_articles(
    url_template: str,
    topic: str,
    max_articles: int,
    source_name: str,
    crawl_delay: tuple = (1.0, 3.0),
):
    """
    Fetch up to `max_articles` from paginated `url_template`, and as soon as
    each article is parsed, insert it into MongoDB with fields:
      - link, source, content, status='processed'
    """
    # 1) connect & ping once
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    coll = client[DB_NAME][COLLECTION]
    print(f"Connected to {DB_NAME}.{COLLECTION}")

    seen = set()
    count = 0
    page = 1

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    while count < max_articles:
        page_url = url_template.format(page=page)
        try:
            resp = requests.get(page_url, headers=headers)
            resp.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch page {page_url}: {e}")
            break

        soup = BeautifulSoup(resp.content, "html.parser")
        links = {
            urljoin("https://www.mentalfloss.com", a["href"])
            for a in soup.find_all("a", href=True)
            if a["href"].startswith("/article/")
            or a["href"].startswith("https://www.mentalfloss.com/article/")
        }

        if not links:
            print(f"No more articles on page {page}; stopping.")
            break

        for link in links:
            if count >= max_articles:
                break
            if link in seen:
                continue
            seen.add(link)

            # fetch article
            try:
                art = requests.get(link, headers=headers)
                art.raise_for_status()
                art_soup = BeautifulSoup(art.content, "html.parser")
                block = art_soup.find("div", class_="article-content") or art_soup.find(
                    "main"
                )
                paragraphs = (
                    [p.get_text(strip=True) for p in block.find_all("p")]
                    if block
                    else []
                )
                content = " ".join(paragraphs).strip()
                if not content:
                    raise ValueError("Empty content")
            except Exception as e:
                print(f"  • Skipping {link} (fetch/parsing error: {e})")
                time.sleep(random.uniform(*crawl_delay))
                continue

            doc = {
                "link": link,
                "source": source_name,
                "content": content,
                "status": "processed",
            }

            # insert (skip duplicates)
            try:
                coll.insert_one(doc)
                count += 1
                print(f"[{count}/{max_articles}] inserted: {link}")
            except errors.DuplicateKeyError:
                print(f"[{count}/{max_articles}] duplicate, skipped: {link}")
            except Exception as e:
                print(f"[{count}/{max_articles}] insert failed: {e}")

            time.sleep(random.uniform(*crawl_delay))

        page += 1

    print(f"Done! Total saved: {count}/{max_articles}")


if __name__ == "__main__":
    fetch_and_save_articles(
        url_template="https://www.mentalfloss.com/section/animals?page={page}",
        topic="animals",
        max_articles=5000,
        source_name="Mental Floss",
    )
