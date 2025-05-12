import os
import random
import datetime
import pandas as pd

# Settings
topics = [
    "Animals","Arts","Business","Science","Nature","Demography","Geography", 
    "History","Culture","Education","AI","Languages","Education and Society",
    "Entertainment" "Environment","Health","Politics","Travel","Books","Space"
]

output_folder = "fake_articles"
output_csv = "fake_articles.csv"
num_articles = 5000  # Change this number as you want

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Helper functions
def generate_fake_title(topic):
    templates = [
        "The Untold Secrets of {}",
        "How {} Changed the World Forever",
        "Discovering the Hidden Side of {}",
        "10 Shocking Facts About {}",
        "The Mythical Origins of {}",
        "Why {} Might Be Different Than You Think",
        "Exploring the Ancient History of {}",
        "How {} Will Shape Our Future"
    ]
    template = random.choice(templates)
    return template.format(topic)

def generate_realistic_fake_body(topic, base_min_words=500, base_max_words=1000):
    min_words = random.randint(base_min_words, base_min_words + 200)
    max_words = random.randint(base_max_words - 200, base_max_words)
    num_words = random.randint(min_words, max_words)

    intro_sentences = [
        f"Many experts have debated the true nature of {topic} for decades.",
        f"Recent discoveries suggest that our understanding of {topic} might be completely wrong.",
        f"Throughout history, {topic} has played a crucial role in shaping civilizations.",
        f"New theories propose astonishing insights into {topic}."
    ]
    middle_sentences = [
        f"In the early 20th century, several groundbreaking studies on {topic} were conducted, but many have since been debunked.",
        f"Legends and folklore surrounding {topic} hint at a much deeper significance.",
        f"Researchers now believe that the true story behind {topic} is far more complex and fascinating than previously thought.",
        f"According to a fictitious report, the dynamics of {topic} are influenced by unknown cosmic forces."
    ]
    conclusion_sentences = [
        f"Ultimately, the mystery of {topic} continues to captivate and bewilder scholars around the globe.",
        f"Although the facts remain elusive, the fascination with {topic} is unlikely to fade anytime soon.",
        f"As new fake discoveries are made, our perception of {topic} may change forever.",
        f"Whether myth or reality, {topic} remains a topic of endless debate."
    ]

    paragraphs = []
    current_word_count = 0

    # Build body paragraphs
    while current_word_count < num_words - 100:
        paragraph = random.choice(intro_sentences) + " "
        paragraph += " ".join(random.choice(middle_sentences) for _ in range(3)) + " "
        paragraph = paragraph.strip()
        if not paragraph.endswith("."):
            paragraph += "."
        paragraphs.append(paragraph)
        current_word_count += len(paragraph.split())

    # Add a final conclusion paragraph
    conclusion_paragraph = random.choice(conclusion_sentences)
    if not conclusion_paragraph.endswith("."):
        conclusion_paragraph += "."
    paragraphs.append(conclusion_paragraph)

    body_text = "\n\n".join(paragraphs)
    words = body_text.split()[:num_words]
    final_body = " ".join(words)

    if not final_body.endswith("."):
        final_body += "."

    return final_body

def generate_summary(topic):
    templates = [
        f"An insightful exploration into the lesser-known aspects of {topic}.",
        f"This article challenges traditional views about {topic} with fictional evidence.",
        f"A creative journey uncovering myths and fabricated facts about {topic}.",
        f"Discover the surprising fictional history and future of {topic}."
    ]
    return random.choice(templates)

def generate_fake_author():
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Quinn", "Jamie", "Skyler"]
    last_names = ["Smith", "Johnson", "Brown", "Williams", "Jones", "Garcia", "Miller", "Davis"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_fake_date():
    start_date = datetime.date(2010, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    return random_date.strftime("%B %d, %Y")

# Store articles data
articles_data = []

# Generate articles
for i in range(num_articles):
    topic = random.choice(topics)
    title = generate_fake_title(topic)
    body = generate_realistic_fake_body(topic)
    summary = generate_summary(topic)
    author = generate_fake_author()
    date = generate_fake_date()

    article_content = f"Title: {title}\n"
    article_content += f"Author: {author}\n"
    article_content += f"Date: {date}\n"
    article_content += f"Topic: {topic}\n"
    article_content += f"Summary: {summary}\n"
    article_content += "\n"
    article_content += body

    # Save to individual file
    filename = os.path.join(output_folder, f"article_{i+1}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(article_content)

    # Save to list for CSV
    articles_data.append({
        "Title": title,
        "Author": author,
        "Date": date,
        "Topic": topic,
        "Summary": summary,
        "Body": body
    })

# Save all articles into one CSV file
df = pd.DataFrame(articles_data)
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"{num_articles} fake articles with summaries generated in folder '{output_folder}' and saved to '{output_csv}'!")
