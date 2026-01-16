#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   ML Model Training Service${NC}"
echo -e "${BLUE}============================================${NC}"

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}docker-compose не установлен${NC}"
    exit 1
fi

cd "$(dirname "$0")/../infra" || exit 1

echo -e "\n${YELLOW} Проверка зависимостей...${NC}"

if ! docker-compose ps mongodb | grep -q "Up"; then
    echo -e "${YELLOW} MongoDB не запущен, запускаем...${NC}"
    docker-compose up -d mongodb
    sleep 10
fi

if ! docker-compose ps postgres | grep -q "Up"; then
    echo -e "${YELLOW} PostgreSQL не запущен, запускаем...${NC}"
    docker-compose up -d postgres
    sleep 10
fi

echo -e "${GREEN}Все зависимости запущены${NC}"

echo -e "\n${YELLOW} Проверка данных для обучения...${NC}"
DATA_COUNT=$(docker-compose exec -T mongodb mongosh -u mongo -p mongo123 --authenticationDatabase admin reviews --eval "db.labeled_posts.countDocuments()" --quiet)

if [ "$DATA_COUNT" -lt 100 ]; then
    echo -e "${YELLOW}Обнаружено только $DATA_COUNT размеченных записей${NC}"
    echo -e "${YELLOW} Будут использованы синтетические данные${NC}"
else
    echo -e "${GREEN}Найдено $DATA_COUNT размеченных записей${NC}"
fi

echo -e "\n${BLUE}Выберите режим запуска:${NC}"
echo "1) Обучение с нуля (build + train)"
echo "2) Использовать существующий образ (train only)"
echo "3) Только сборка образа (build only)"
read -p "Ваш выбор [1-3]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Сборка Docker образа...${NC}"
        docker-compose build train-service

        echo -e "\n${YELLOW}Запуск обучения модели...${NC}"
        docker-compose run --rm train-service
        ;;
    2)
        echo -e "\n${YELLOW}Запуск обучения модели...${NC}"
        docker-compose run --rm train-service
        ;;
    3)
        echo -e "\n${YELLOW}Сборка Docker образа...${NC}"
        docker-compose build train-service
        echo -e "${GREEN}Образ собран успешно${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Неверный выбор${NC}"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN} Обучение завершено успешно!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "\n${BLUE}Модель сохранена в volume: model_data${NC}"
    echo -e "${BLUE}Логи обучения: docker logs train-service${NC}"

    echo -e "\n${YELLOW}Информация о модели:${NC}"
    docker-compose run --rm train-service ls -lh /app/models/sentiment/
else
    echo -e "\n${RED}============================================${NC}"
    echo -e "${RED} Обучение завершилось с ошибкой${NC}"
    echo -e "${RED}============================================${NC}"
    echo -e "\n${YELLOW}Проверьте логи:${NC}"
    echo -e "${BLUE}docker logs train-service${NC}"
    exit 1
fi