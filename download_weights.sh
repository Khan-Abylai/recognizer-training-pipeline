#!/usr/bin/env bash

pip install --quiet gdown

FOLDER_URL="https://drive.google.com/drive/folders/1ZtPeoPoECFeXsZJncfN7W7qKDh3lvRTT?usp=drive_link"

DEST_DIR="weights"

mkdir -p "$DEST_DIR"

echo "Скачиваю содержимое папки из Google Drive в '$DEST_DIR'..."

gdown --folder "$FOLDER_URL" -O "$DEST_DIR"

echo "Скачивание завершено. Проверьте содержимое в папке '$DEST_DIR'."
