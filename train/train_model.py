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
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        Config.validate()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Используется устройство: {self.device}")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        logger.info(f"Загрузка модели: {Config.MODEL_NAME}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                Config.MODEL_NAME,
                num_labels=Config.NUM_LABELS
            )
            self.model.to(self.device)
            logger.info("Модель загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

        self.preparer = DatasetPreparer()
        self.train_df, self.val_df, self.test_df = self.preparer.prepare_dataset()

    def create_data_loaders(self):
        logger.info("Создание DataLoader'ов...")

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
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        logger.info(f"DataLoader'ы созданы: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)} batches")

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")

        for batch_idx, batch in enumerate(progress_bar):
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

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return avg_loss, accuracy, f1

    def evaluate(self, data_loader, phase="Validation"):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=phase):
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
        logger.info(f"Модель: {Config.MODEL_NAME}")
        logger.info(f"Параметры: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}, Epochs={Config.EPOCHS}")

        train_loader, val_loader, test_loader = self.create_data_loaders()

        optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)

        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% разогрев
            num_training_steps=total_steps
        )

        best_val_f1 = 0
        best_epoch = 0

        for epoch in range(Config.EPOCHS):
            logger.info(f"\n{'='*60}")
            logger.info(f"📚 Эпоха {epoch + 1}/{Config.EPOCHS}")
            logger.info(f"{'='*60}")

            train_loss, train_acc, train_f1 = self.train_epoch(
                train_loader, optimizer, scheduler, epoch
            )

            logger.info(f"\n📊 Train Results:")
            logger.info(f"   Loss: {train_loss:.4f}")
            logger.info(f"   Accuracy: {train_acc:.4f}")
            logger.info(f"   F1: {train_f1:.4f}")

            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader, "Validation")

            logger.info(f"\n🎯 Validation Results:")
            logger.info(f"   Loss: {val_loss:.4f}")
            logger.info(f"   Accuracy: {val_acc:.4f}")
            logger.info(f"   F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                self.save_model()
                logger.info(f"\nНовая лучшая модель сохранена F1: {val_f1:.4f}")

        logger.info(f"\nЛучшая модель: Epoch {best_epoch}, F1: {best_val_f1:.4f}")

        # Финальное тестирование
        logger.info("\n" + "="*60)
        logger.info("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ")
        logger.info("="*60)

        test_loss, test_acc, test_f1, predictions, true_labels = self.evaluate(
            test_loader, "Testing"
        )

        logger.info(f"\nTest Results:")
        logger.info(f"Loss: {test_loss:.4f}")
        logger.info(f"Accuracy: {test_acc:.4f}")
        logger.info(f"F1: {test_f1:.4f}")

        label_names = ['negative', 'neutral', 'positive']
        report = classification_report(
            true_labels,
            predictions,
            target_names=label_names,
            digits=4
        )
        logger.info(f"\nКлассификационный отчет:\n{report}")

        logger.info("\nОбучение завершено")

    def save_model(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        self.model.save_pretrained(Config.OUTPUT_DIR)
        self.tokenizer.save_pretrained(Config.OUTPUT_DIR)

        config_file = os.path.join(Config.OUTPUT_DIR, 'training_config.txt')
        with open(config_file, 'w') as f:
            f.write(f"MODEL_NAME={Config.MODEL_NAME}\n")
            f.write(f"MAX_LENGTH={Config.MAX_LENGTH}\n")
            f.write(f"BATCH_SIZE={Config.BATCH_SIZE}\n")
            f.write(f"LEARNING_RATE={Config.LEARNING_RATE}\n")
            f.write(f"EPOCHS={Config.EPOCHS}\n")

        logger.info(f"Модель сохранена в {Config.OUTPUT_DIR}")

    def load_model(self):
        if not os.path.exists(Config.OUTPUT_DIR):
            raise FileNotFoundError(f"Модель не найдена в {Config.OUTPUT_DIR}")

        self.model = AutoModelForSequenceClassification.from_pretrained(Config.OUTPUT_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.OUTPUT_DIR)
        self.model.to(self.device)

        logger.info(f"Модель загружена из {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        trainer = SentimentModelTrainer()
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("\nОбучение прервано пользователем")
    except Exception as e:
        logger.error(f"\nОшибка при обучении: {e}", exc_info=True)
        raise