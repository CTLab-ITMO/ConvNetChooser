# Architecture Selection

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
