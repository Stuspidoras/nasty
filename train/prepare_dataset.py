import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:


    def __init__(self):
        self.mongo_client = MongoClient(Config.get_mongo_url())
        self.db = self.mongo_client[Config.MONGO_DB]

    def load_labeled_data(self):

        logger.info("Загрузка размеченных данных...")

        cursor = self.db.labeled_posts.find({}, {'text': 1, 'sentiment': 1})

        data = []
        for doc in cursor:
            data.append({
                'text': doc.get('text', ''),
                'label': doc.get('sentiment', 'neutral')
            })

        df = pd.DataFrame(data)
        logger.info(f"Загружено {len(df)} размеченных записей")

        return df

    def create_synthetic_dataset(self):

        logger.info("Создание синтетического датасета...")

        positive_examples = [
            "Отличный продукт! Очень доволен покупкой",
            "Превосходное качество, рекомендую всем",
            "Замечательный сервис, быстрая доставка",
            "Лучшее, что я покупал за последнее время",
            "Идеально подходит, буду заказывать ещё",
        ] * 200

        negative_examples = [
            "Ужасное качество, не рекомендую",
            "Разочарован покупкой, зря потратил деньги",
            "Плохой сервис, долгая доставка",
            "Не соответствует описанию",
            "Полный провал, никому не советую",
        ] * 200

        neutral_examples = [
            "Обычный товар, ничего особенного",
            "Нормальное качество за свою цену",
            "Средненько, можно было и лучше",
            "Приемлемо, но есть нюансы",
            "Стандартный продукт",
        ] * 200

        data = []

        for text in positive_examples:
            data.append({'text': text, 'label': 'positive'})

        for text in negative_examples:
            data.append({'text': text, 'label': 'negative'})

        for text in neutral_examples:
            data.append({'text': text, 'label': 'neutral'})

        df = pd.DataFrame(data)
        df = df.sample(frac=1).reset_index(drop=True)  # Перемешивание

        logger.info(f"Создано {len(df)} синтетических примеров")

        return df

    def prepare_dataset(self):

        df = self.load_labeled_data()

        if len(df) < 100:
            logger.warning("Мало размеченных данных, добавление синтетических...")
            synthetic_df = self.create_synthetic_dataset()
            df = pd.concat([df, synthetic_df], ignore_index=True)

        df = df.drop_duplicates(subset=['text'])

        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['label'] = df['label'].map(label_map)

        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df