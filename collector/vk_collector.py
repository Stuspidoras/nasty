import vk_api
from vk_api.exceptions import ApiError
import time
from datetime import datetime
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VKCollector:
    def __init__(self):
        self.vk_session = vk_api.VkApi(token=Config.VK_ACCESS_TOKEN)
        self.vk = self.vk_session.get_api()

    def search_posts(self, query, count=100):
        posts = []
        offset = 0

        try:
            while len(posts) < count:
                response = self.vk.newsfeed.search(
                    q=query,
                    count=min(200, count - len(posts)),
                    offset=offset
                )

                if not response['items']:
                    break

                for item in response['items']:
                    post_data = self._parse_post(item, query)
                    posts.append(post_data)

                offset += 200
                time.sleep(0.34)  # Ограничение API: 3 запроса в секунду

            logger.info(f"Собрано {len(posts)} постов по запросу: {query}")
            return posts

        except ApiError as e:
            logger.error(f"Ошибка VK API: {e}")
            return posts

    def search_wall_posts(self, owner_id, query, count=100):
        posts = []
        offset = 0

        try:
            while len(posts) < count:
                response = self.vk.wall.get(
                    owner_id=owner_id,
                    count=min(100, count - len(posts)),
                    offset=offset,
                    filter='all'
                )

                if not response['items']:
                    break

                for item in response['items']:
                    # Фильтрация по ключевым словам
                    if query.lower() in item.get('text', '').lower():
                        post_data = self._parse_post(item, query)
                        posts.append(post_data)

                offset += 100
                time.sleep(0.34)

            logger.info(f"Собрано {len(posts)} постов со стены {owner_id}")
            return posts

        except ApiError as e:
            logger.error(f"Ошибка VK API: {e}")
            return posts

    def get_comments(self, owner_id, post_id, count=100):

        comments = []
        offset = 0

        try:
            while len(comments) < count:
                response = self.vk.wall.getComments(
                    owner_id=owner_id,
                    post_id=post_id,
                    count=min(100, count - len(comments)),
                    offset=offset,
                    sort='asc'
                )

                if not response['items']:
                    break

                for item in response['items']:
                    comment_data = self._parse_comment(item)
                    comments.append(comment_data)

                offset += 100
                time.sleep(0.34)

            return comments

        except ApiError as e:
            logger.error(f"Ошибка при получении комментариев: {e}")
            return comments

    def _parse_post(self, item, query):
        """Парсинг данных поста"""
        return {
            'source': 'vk',
            'post_id': item.get('id'),
            'owner_id': item.get('owner_id'),
            'text': item.get('text', ''),
            'date': datetime.fromtimestamp(item.get('date')),
            'likes': item.get('likes', {}).get('count', 0),
            'reposts': item.get('reposts', {}).get('count', 0),
            'views': item.get('views', {}).get('count', 0),
            'comments_count': item.get('comments', {}).get('count', 0),
            'query': query,
            'collected_at': datetime.now()
        }

    def _parse_comment(self, item):

        return {
            'comment_id': item.get('id'),
            'from_id': item.get('from_id'),
            'text': item.get('text', ''),
            'date': datetime.fromtimestamp(item.get('date')),
            'likes': item.get('likes', {}).get('count', 0),
            'reply_to_user': item.get('reply_to_user'),
            'reply_to_comment': item.get('reply_to_comment')
        }