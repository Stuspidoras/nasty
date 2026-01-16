import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
    EPOCHS = int(os.getenv('EPOCHS', 3))

    MODEL_NAME = os.getenv('MODEL_NAME', 'DeepPavlov/rubert-base-cased')
    MAX_LENGTH = int(os.getenv('MAX_LENGTH', 128))
    NUM_LABELS = 3  # negative, neutral, positive

    MODEL_DIR = os.getenv('MODEL_DIR', '/app/models')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/app/models/sentiment')

    SENTIMENT_MODEL_PATH = os.path.join(OUTPUT_DIR, 'pytorch_model.bin')

    MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
    MONGO_PORT = int(os.getenv('MONGO_PORT', 27017))
    MONGO_USER = os.getenv('MONGO_USER', 'mongo')
    MONGO_PASSWORD = os.getenv('MONGO_PASSWORD', 'mongo123')
    MONGO_DB = os.getenv('MONGO_DB', 'reviews')

    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres123')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'reviewsdb')

    @staticmethod
    def get_mongo_url():
        return f"mongodb://{Config.MONGO_USER}:{Config.MONGO_PASSWORD}@{Config.MONGO_HOST}:{Config.MONGO_PORT}/{Config.MONGO_DB}?authSource=admin"

    @staticmethod
    def get_postgres_url():
        return f"postgresql://{Config.POSTGRES_USER}:{Config.POSTGRES_PASSWORD}@{Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}"

    @staticmethod
    def validate():
        required_params = {
            'MODEL_NAME': Config.MODEL_NAME,
            'OUTPUT_DIR': Config.OUTPUT_DIR,
            'MONGO_HOST': Config.MONGO_HOST,
            'MONGO_DB': Config.MONGO_DB,
        }

        for param_name, param_value in required_params.items():
            if not param_value:
                raise ValueError(f"Параметр {param_name} не установлен!")

        print("Конфигурация валидна")
        print(f"Модель: {Config.MODEL_NAME}")
        print(f"Путь сохранения: {Config.OUTPUT_DIR}")
        print(f"Параметры: batch={Config.BATCH_SIZE}, lr={Config.LEARNING_RATE}, epochs={Config.EPOCHS}")