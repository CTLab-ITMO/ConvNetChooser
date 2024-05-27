import sys
import os

# Добавить корневую папку в sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from extractors.extractor import Features1065
from utils.utils import predict, load_model_and_encoder
import os
import gdown

# Путь к изображениям
image_dir = 'data/'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
image_files = image_files[:5]

# Путь к сохраненной модели и LabelEncoder
# Проверка, существует ли файл
model_path = 'experiments/model_with_params.pth'
if not os.path.exists(model_path):
    print("Model file not found. Downloading from Google Drive...")
    url = 'https://drive.google.com/uc?id=1G-p3pTZmLVL5Cq3jr7yY0M2iS_fY2DYr'

    gdown.download(url, model_path, quiet=False)

encoder_path = 'experiments/label_encoder.pkl'

feature_extractor = Features1065(by=2, device='cpu')

extracted_features = []
# Извлечение признаков из изображений
for image_file in image_files:
    # Извлекаем признаки для текущего изображения
    image_features = feature_extractor._get_single_image_features(image_file)
    # Добавляем метафичи в общий список
    extracted_features.extend(image_features)

# Загрузка обученной модели и LabelEncoder
model, label_encoder = load_model_and_encoder(model_path, encoder_path)

# Предсказание классов на новых данных
predicted_labels = predict(model, [extracted_features])
predicted_classes = label_encoder.inverse_transform(predicted_labels)

print("Predicted Model:", predicted_classes)
