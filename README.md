## Sentiment-App

Приложение, определяющее эмоциональный окрас отзывов на товары. Train set включает 2000 отзывов на товары англоязычных пользователей. Test set состоит из 500 предложений. Анализ тональности осуществляется с помощью модели BERT. BERT основана на использовании архитектуры attention. Тенсор, извлекаемый из последнего слоя attention первого токена, проганяетсячерез линный слой с двумя нейронами, которые подсоединяются к слою softmax. В проекте используется библиотека transformers с pretrained эмбеддингами, токенизатором и моделью, рассчитанную на два класса. Для построения модели используется pytorch. Модель файтюнится на вышеупомянутом тренировочном наборе.

Для запуска приложения, скопируйте репозиторий в свою директиву, используя команду:
```
git clone https://github.com/Artyom112/Sentiment-App.git
```
Затем распакуйте папку trained_model_parameters и установитке необходимые библиотеки с импользоанием pip. Перейдите в файл app.py и запустите:
```
if __name__=="__main__":
    app.run(port=5001, debug=True)
```

**Архитектура BERT**:
![Image of Bert architecture](http://jalammar.github.io/images/bert-output-vector.png)
