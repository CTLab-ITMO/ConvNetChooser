import torch
from models import TwoLayerClassifier
import joblib
import os


# Использование модели для предсказания на новых данных
def predict(model, new_data):
    model.eval()  # Перевод модели в режим оценки
    new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(new_data_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()


# Функция для загрузки обученной модели и LabelEncoder
def load_model_and_encoder(model_path, encoder_path):
    model_params = torch.load(model_path, map_location=torch.device('cpu'))
    input_size = model_params['input_size']
    hidden_size = model_params['hidden_size']
    num_classes = model_params['num_classes']
    model = TwoLayerClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(model_params['state_dict'])
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

def find_images_in_directory(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
                image_files.append(os.path.join(root, file))
    return image_files
