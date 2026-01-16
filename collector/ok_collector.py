import requests
import hashlib
import time
from datetime import datetime
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OKCollector:
    def __init__(self):
        self.access_token = Config.OK_ACCESS_TOKEN
        self.application_key = Config.OK_APPLICATION_KEY
        self.session_secret_key = Config.OK_SESSION_SECRET_KEY
        self.api_url = 'https://api.ok.ru/fb.do'

    def _generate_sig(self, params):
        param_str = ''.join([f"{k}={v}" for k, v in sorted(params.items())])
        sig_str = param_str + self.session_secret_key
        return hashlib.md5(sig_str.encode()).hexdigest()

    def _make_request(self, method, params):
        params['application_key'] = self.application_key
        params['method'] = method
        params['access_token'] = self.access_token
        params['format'] = 'json'

        sig_params = {k: v for k, v in params.items() if k != 'access_token'}
        params['sig'] = self._generate_sig(sig_params)

        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка OK API: {e}")
            return None

    def search_posts(self, query, count=100):
        posts = []

        params = {
            'count': min(100, count)
        }

        response = self._make_request('stream.get', params)

        if response:
            for item in response.get('stream', []):
                if query.lower() in item.get('text', '').lower():
                    post_data = self._parse_post(item, query)
                    posts.append(post_data)

        logger.info(f"Собрано {len(posts)} постов по запросу: {query} из OK")
        return posts

    def get_group_posts(self, group_id, query, count=100):
        posts = []

        params = {
            'gid': group_id,
            'count': min(100, count)
        }

        response = self._make_request('group.getTopics', params)

        if response:
            for item in response.get('topics', []):
                if query.lower() in item.get('text', '').lower():
                    post_data = self._parse_post(item, query)
                    posts.append(post_data)

        logger.info(f"Собрано {len(posts)} постов из группы {group_id}")
        return posts

    def _parse_post(self, item, query):
        return {
            'source': 'ok',
            'post_id': item.get('id'),
            'author_id': item.get('author_id'),
            'text': item.get('text', ''),
            'date': datetime.fromtimestamp(item.get('created_ms', 0) / 1000),
            'likes': item.get('like_summary', {}).get('count', 0),
            'shares': item.get('share_count', 0),
            'comments_count': item.get('discussion_summary', {}).get('comments_count', 0),
            'query': query,
            'collected_at': datetime.now()
        }