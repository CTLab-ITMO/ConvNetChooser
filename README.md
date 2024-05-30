# Convolutional Architecture Selection For Classification Task

Этот проект представляет собой инструмент для выбора архитектуры модели машинного обучения.

## Установка

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/your_username/Architecture_selection.git
    ```

2. Перейдите в каталог проекта:

    ```bash
    cd Architecture_selection
    ```

3. Установите зависимости:

    ```bash
    pip install -r requirements.txt
    ```

## Использование

1. Создайте экземпляр класса `ModelSelection`:

    ```python
    from Architecture_selection import ModelSelection

    Selection = ModelSelection()
    ```

2. Укажите путь к директории с изображениями:

    ```python
    image_dir = '/path/to/your/image/directory'
    ```

3. Выберите архитектуру модели, запустив метод `select_model`:

    ```python
    selected_model = Selection.predict_model(image_dir=image_dir)
    ```

4. Получите результат выбора:

    ```python
    print("Selected model:", selected_model)
    ```
   
## Запуск на Google Colab

Есть руководство по запуску проекта на Google Colab. Вы можете найти его в папке `notebooks/guide.ipynb`.

## Вклад

- Если вы хотите сообщить об ошибке или предложить улучшение, создайте issue на GitHub.
- При желании вы можете внести свой вклад, создав pull request.

# Convolutional Architecture Selection For Classification Task

This project is a tool for selecting a machine learning model architecture.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/Architecture_selection.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Architecture_selection
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Create an instance of the `ModelSelection` class:

    ```python
    from Architecture_selection import ModelSelection

    Selection = ModelSelection()
    ```

2. Specify the path to the directory containing images:

    ```python
    image_dir = '/path/to/your/image/directory'
    ```

3. Choose the model architecture by running the `select_model` method:

    ```python
    selected_model = Selection.predict_model(image_dir=image_dir)
    ```

4. Get the selection result:

    ```python
    print("Selected model:", selected_model)
    ```
   
## Running on Google Colab

There's a guide available for running the project on Google Colab. You can find it in the `notebooks/guide.ipynb` folder.

## Contribution

- If you want to report a bug or suggest an improvement, create an issue on GitHub.
- If you'd like, you can contribute by creating a pull request.

