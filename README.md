## Sentiment-App

Приложение, определяющее эмоциональный окрас отзывов на товары. Train set включает 2000 отзывов на товары англоязычных пользователей. Test set состоит из 500 предложений. BERT основана на использовании архитектуры attention. Тенсор, извлекаемый из последнего слоя attention первого токена, проганяетсячерез линный слой с двумя нейронами, которые подсоединяются к слою softmax. В проекте используется библиотека transformers с pretrained эмбеддингами, токенизатором и моделью, рассчитанной на два класса. Для построения модели используется pytorch. 

![Image of Bert architecture](https://octodex.github.com/images/yaktocat.png)
