import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from prepare_dataset import DatasetPreparer
from config import Config
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentModelTrainer:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Используется устройство: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=3  # negative, neutral, positive
        )
        self.model.to(self.device)

        self.preparer = DatasetPreparer()
        self.train_df, self.val_df, self.test_df = self.preparer.prepare_dataset()

    def create_data_loaders(self):

        train_dataset = ReviewDataset(
            self.train_df['text'].values,
            self.train_df['label'].values,
            self.tokenizer,
            Config.MAX_LENGTH
        )

        val_dataset = ReviewDataset(
            self.val_df['text'].values,
            self.val_df['label'].values,
            self.tokenizer,
            Config.MAX_LENGTH
        )

        test_dataset = ReviewDataset(
            self.test_df['text'].values,
            self.test_df['label'].values,
            self.tokenizer,
            Config.MAX_LENGTH
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, optimizer, scheduler):

        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return avg_loss, accuracy, f1

    def evaluate(self, data_loader):

        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return avg_loss, accuracy, f1, predictions, true_labels

    def train(self):
        logger.info("Начало обучения модели...")

        train_loader, val_loader, test_loader = self.create_data_loaders()

        optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)

        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_val_f1 = 0

        for epoch in range(Config.EPOCHS):
            logger.info(f"\n{'='*50}")
            logger.info(f"Эпоха {epoch + 1}/{Config.EPOCHS}")
            logger.info(f"{'='*50}")

            train_loss, train_acc, train_f1 = self.train_epoch(
                train_loader, optimizer, scheduler
            )

            logger.info(f"\nTrain Loss: {train_loss:.4f}")
            logger.info(f"Train Accuracy: {train_acc:.4f}")
            logger.info(f"Train F1: {train_f1:.4f}")

            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader)

            logger.info(f"\nVal Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_acc:.4f}")
            logger.info(f"Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model()
                logger.info(f"Новая лучшая модель сохранена! F1: {val_f1:.4f}")

        logger.info("\n" + "="*50)
        logger.info("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ")
        logger.info("="*50)

        test_loss, test_acc, test_f1, predictions, true_labels = self.evaluate(test_loader)

        logger.info(f"\nTest Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test F1: {test_f1:.4f}")

        label_names = ['negative', 'neutral', 'positive']
        report = classification_report(true_labels, predictions, target_names=label_names)
        logger.info(f"\nКлассификационный отчет:\n{report}")

        logger.info("\n✓ Обучение завершено успешно!")

    def save_model(self):

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        self.model.save_pretrained(Config.OUTPUT_DIR)
        self.tokenizer.save_pretrained(Config.OUTPUT_DIR)

        logger.info(f"Модель сохранена в {Config.OUTPUT_DIR}")

    def load_model(self):

        self.model = AutoModelForSequenceClassification.from_pretrained(Config.OUTPUT_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.OUTPUT_DIR)
        self.model.to(self.device)

        logger.info(f"Модель загружена из {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        trainer = SentimentModelTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}", exc_info=True)