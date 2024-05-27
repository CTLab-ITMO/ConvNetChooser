import os
import gdown
from extractors.extractor import Features1065
from utils.utils import predict, load_model_and_encoder, find_images_in_directory

# Получение пути к директории, содержащей текущий исполняемый файл
current_directory = os.path.dirname(__file__)

# Формирование пути к директории experiments внутри текущей директории
experiments_directory = os.path.join(current_directory, 'experiments')


class ModelSelection:
    def __init__(self, by=2, device='cpu'):
        # Инициализация пути к модели и LabelEncoder
        self.model_path = os.path.join(experiments_directory, 'model_with_params.pth')
        self.encoder_path = os.path.join(experiments_directory, 'label_encoder.pkl')

        # Проверка, существует ли файл модели
        if not os.path.exists(self.model_path):
            print("Model file not found. Downloading from Google Drive...")
            url = 'https://drive.google.com/uc?id=1G-p3pTZmLVL5Cq3jr7yY0M2iS_fY2DYr'
            gdown.download(url, self.model_path, quiet=False)

        # Инициализация объекта для извлечения признаков
        self.feature_extractor = Features1065(by=by, device=device)

    def predict_model(self, image_dir='/data'):
        # Извлечение признаков из изображений в указанной папке
        extracted_features = []
        image_files = find_images_in_directory(image_dir)
        for image_file in image_files:
            # Извлечение признаков для текущего изображения
            image_features = self.feature_extractor._get_single_image_features(image_file)
            extracted_features.extend(image_features)

        # Загрузка обученной модели и LabelEncoder
        model, label_encoder = load_model_and_encoder(self.model_path, self.encoder_path)

        # Предсказание классов на новых данных
        predicted_labels = predict(model, [extracted_features])
        predicted_model = label_encoder.inverse_transform(predicted_labels)

        return predicted_model
