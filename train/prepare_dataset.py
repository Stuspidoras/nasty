import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from config import Config
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    def __init__(self):
        try:
            self.mongo_client = MongoClient(Config.get_mongo_url())
            self.db = self.mongo_client[Config.MONGO_DB]
            logger.info(f"Подключение к MongoDB успешно: {Config.MONGO_DB}")
        except Exception as e:
            logger.warning(f"Не удалось подключиться к MongoDB: {e}")
            self.db = None

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9\s.,!?-]', '', text)

        return text.strip()

    def load_labeled_data(self):
        logger.info("Загрузка размеченных данных")

        if self.db is None:
            logger.warning("MongoDB недоступна, используем только синтетические данные")
            return pd.DataFrame(columns=['text', 'label'])

        try:
            cursor = self.db.labeled_posts.find({}, {'text': 1, 'sentiment': 1})

            data = []
            for doc in cursor:
                text = self.clean_text(doc.get('text', ''))
                sentiment = doc.get('sentiment', 'neutral')

                if text and sentiment in ['negative', 'neutral', 'positive']:
                    data.append({
                        'text': text,
                        'label': sentiment
                    })

            df = pd.DataFrame(data)
            logger.info(f"Загружено {len(df)} размеченных записей")

            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return pd.DataFrame(columns=['text', 'label'])

    def create_synthetic_dataset(self):
        logger.info("Создание синтетического датасета")

        positive_examples = [
            "Отличный продукт! Очень доволен покупкой",
            "Превосходное качество, рекомендую всем",
            "Замечательный сервис, быстрая доставка",
            "Лучшее, что я покупал за последнее время",
            "Идеально подходит, буду заказывать ещё",
            "Великолепно! Превзошло все ожидания",
            "Супер качество, цена полностью оправдана",
            "Очень рад покупке, всё на высшем уровне",
            "Прекрасный товар, советую всем друзьям",
            "Потрясающе! Именно то, что нужно",
        ] * 150

        negative_examples = [
            "Ужасное качество, не рекомендую",
            "Разочарован покупкой, зря потратил деньги",
            "Плохой сервис, долгая доставка",
            "Не соответствует описанию",
            "Полный провал, никому не советую",
            "Отвратительный товар, деньги на ветер",
            "Ужасно разочарован, верну обратно",
            "Кошмарное качество, не стоит своих денег",
            "Худшая покупка в моей жизни",
            "Категорически не рекомендую это барахло",
        ] * 150

        neutral_examples = [
            "Обычный товар, ничего особенного",
            "Нормальное качество за свою цену",
            "Средненько, можно было и лучше",
            "Приемлемо, но есть нюансы",
            "Стандартный продукт",
            "На троечку, не более",
            "Сойдёт, но ожидал большего",
            "Ничего выдающегося, обычная вещь",
            "Качество среднее, цена адекватная",
            "В целом нормально, но есть недочёты",
        ] * 150

        data = []

        for text in positive_examples:
            data.append({'text': self.clean_text(text), 'label': 'positive'})

        for text in negative_examples:
            data.append({'text': self.clean_text(text), 'label': 'negative'})

        for text in neutral_examples:
            data.append({'text': self.clean_text(text), 'label': 'neutral'})

        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Создано {len(df)} синтетических примеров")
        logger.info(f"Позитивных: {len([x for x in data if x['label'] == 'positive'])}")
        logger.info(f"Негативных: {len([x for x in data if x['label'] == 'negative'])}")
        logger.info(f"Нейтральных: {len([x for x in data if x['label'] == 'neutral'])}")

        return df

    def prepare_dataset(self):
        logger.info("Подготовка датасета...")

        df = self.load_labeled_data()

        if len(df) < 100:
            logger.warning(f"Мало размеченных данных ({len(df)}), добавление синтетических...")
            synthetic_df = self.create_synthetic_dataset()
            df = pd.concat([df, synthetic_df], ignore_index=True)
        else:
            logger.info(f"Достаточно размеченных данных: {len(df)}")

        initial_len = len(df)
        df = df.drop_duplicates(subset=['text'])
        logger.info(f"Удалено дубликатов: {initial_len - len(df)}")

        df = df[df['text'].str.len() > 5]
        logger.info(f"Отфильтровано коротких текстов: осталось {len(df)}")

        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['label'] = df['label'].map(label_map)

        label_counts = df['label'].value_counts().sort_index()
        logger.info("Распределение классов:")
        for label, count in label_counts.items():
            label_name = ['negative', 'neutral', 'positive'][label]
            logger.info(f"   {label_name}: {count} ({count/len(df)*100:.1f}%)")

        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            random_state=42,
            stratify=df['label']
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['label']
        )

        logger.info(f"\nДатасет подготовлен:")
        logger.info(f"   📚 Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"   🎯 Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"   🧪 Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def __del__(self):
        if hasattr(self, 'mongo_client') and self.mongo_client:
            self.mongo_client.close()