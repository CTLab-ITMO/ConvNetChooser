from setuptools import setup, find_packages

setup(
    name='my_ml_library',
    version='0.1.0',
    author='Kamil Shakirov',
    author_email='shakirowkamil2001@gmail.com',
    description='Model selection library',
    url='https://github.com/ваш_проект',  # URL вашего проекта
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'scikit-learn>=0.24.1',
        'opencv-python>=4.5.1',
        'scikit-image>=0.18.1',
        'numpy>=1.19.5',
        'Pillow>=8.1.0',
        'imutils>=0.5.4',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
