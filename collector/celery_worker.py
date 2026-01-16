from celery import Celery
from config import Config
from vk_collector import VKCollector
from ok_collector import OKCollector
from pymongo import MongoClient
import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

celery_app = Celery('collector', broker=f'redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0')

mongo_client = MongoClient(Config.get_mongo_url())
mongo_db = mongo_client[Config.MONGO_DB]


def get_postgres_connection():
    return psycopg2.connect(
        host=Config.POSTGRES_HOST,
        port=Config.POSTGRES_PORT,
        database=Config.POSTGRES_DB,
        user=Config.POSTGRES_USER,
        password=Config.POSTGRES_PASSWORD
    )


@celery_app.task(name='collect_reviews_task')
def collect_reviews_task(sources, keywords, count=100):
    vk_collector = VKCollector()
    ok_collector = OKCollector()

    all_posts = []

    for keyword in keywords:
        if 'vk' in sources:
            vk_posts = vk_collector.search_posts(keyword, count)
            all_posts.extend(vk_posts)

            for post in vk_posts[:10]:
                comments = vk_collector.get_comments(
                    post['owner_id'],
                    post['post_id'],
                    count=50
                )
                post['comments'] = comments

        if 'ok' in sources:
            ok_posts = ok_collector.search_posts(keyword, count)
            all_posts.extend(ok_posts)

    if all_posts:
        mongo_db.raw_posts.insert_many(all_posts)
        logger.info(f"Сохранено {len(all_posts)} постов в MongoDB")

    try:
        conn = get_postgres_connection()
        cur = conn.cursor()

        for post in all_posts:
            cur.execute("""
                INSERT INTO collected_posts 
                (source, post_id, text, date, likes, comments_count, query)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, post_id) DO NOTHING
            """, (
                post['source'],
                str(post.get('post_id', '')),
                post['text'],
                post['date'],
                post.get('likes', 0),
                post.get('comments_count', 0),
                post['query']
            ))

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Метаданные сохранены в PostgreSQL")
    except Exception as e:
        logger.error(f"Ошибка при сохранении в PostgreSQL: {e}")

    return {
        'status': 'success',
        'collected': len(all_posts)
    }