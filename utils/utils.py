import torch
from models import TwoLayerClassifier
import joblib


# Использование модели для предсказания на новых данных
def predict(model, new_data):
    model.eval()  # Перевод модели в режим оценки
    new_data = new_data.values
    new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(new_data_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()


# Функция для загрузки обученной модели и LabelEncoder
def load_model_and_encoder(model_path, encoder_path):
    model_params = torch.load(model_path)
    input_size = model_params['input_size']
    hidden_size = model_params['hidden_size']
    num_classes = model_params['num_classes']
    model = TwoLayerClassifier(input_size, hidden_size, num_classes).to('cuda')
    model.load_state_dict(model_params['state_dict'])
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder
