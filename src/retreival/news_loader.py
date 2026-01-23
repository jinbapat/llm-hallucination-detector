from gdelt import gdelt
from datetime import datetime, timedelta

gd = gdelt.GDELT(version=2)

def fetch_news_articles(query: str, days_back: int = 30, max_records: int = 50):
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    results = gd.Search(
        query=query,
        table="events",
        start_date=start.strftime("%Y %m %d"),
        end_date=end.strftime("%Y %m %d"),
        maxrecords=max_records
    )

    if results is None or results.empty:
        return []

    articles = []
    for _, row in results.iterrows():
        articles.append({
            "title": row.get("SOURCEURL", ""),
            "date": row.get("SQLDATE", ""),
            "url": row.get("SOURCEURL", "")
        })

    return articles
