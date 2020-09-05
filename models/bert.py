#%%
#Загружаем и обрабатываем исзодные данные
file_obj = open('/Users/artyomkholodkov/Downloads/products_sentiment_train.tsv')
test_file_obj = open('/Users/artyomkholodkov/Downloads/products_sentiment_test.tsv')


def generate_text_and_targets(file_obj):
    data = [line.strip() for line in file_obj.readlines()]
    targets = [int(sentence[-1]) for sentence in data]
    sentences = [sentence[:-1].strip() for sentence in data]
    return sentences, targets


def generate_text(test_file_obj):
    data = [line.strip() for line in test_file_obj.readlines()][1:]
    sentences = [line[len(str(num)):].strip() for num, line in enumerate(data)]
    return sentences

train_sentences, targets = generate_text_and_targets(file_obj)
test_sentences = generate_text(test_file_obj)
print(len(train_sentences))

#%%

#Загружаем библиотеки. Выбранная архитектура приведена ниже.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('stopwords')

scores = cross_val_score(Pipeline([('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), stop_words='english',
                    min_df=4)), ('classifier', LogisticRegression())]), train_sentences, targets, cv=5, scoring='accuracy')


#%%

#Выберем порог бинаризации на основании roc auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

classifier = Pipeline([('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), stop_words='english', min_df=4)),
                        ('classifier', LogisticRegression())])

X_train, X_test, y_train, y_test = train_test_split(train_sentences, targets, test_size=0.3)

classifier.fit(X_train, y_train)
predict_probs = classifier.predict_proba(X_test)

fpr, tpr, thr = roc_curve(y_test, predict_probs[:, 1])
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

#%%

#Предсказания на основе нового порога бинаризации
import math

def treshold(targets, probs):
  fpr, tpr, thr = roc_curve(targets, probs[:, 1])
  distances = [math.sqrt(math.pow(fp - 0, 2) + math.pow(tp - 1, 2)) for fp, tp in zip(fpr, tpr)]
  treshold = thr[distances.index(min(distances))]
  return treshold

treshold = treshold(y_test, predict_probs)

#%%

def predict_test(t, text):
  classifier.fit(train_sentences, targets)
  test_probs = classifier.predict_proba(text)
  adjusted_preds = [0 if x < t else 1 for x in test_probs[:, 1]]
  return adjusted_preds

predictions = predict_test(treshold, test_sentences)


#%%

"""
Во избежании проведения кросс-валидации для нахождения нужной комбинации гиперпараметров для tfidf, можно
воспользоваться более продвинутой архитектурой для предсказания тональности, называемой BERT. BERT основана на
использовании архитектуры attention. Тенсор, извлекаемый из последнего слоя attention первого токена, проганяется
через линный слой с двумя нейронами, которые подсоединяются к слою softmax. Мы будем пользоваться библиотекой
transformers с pretrained эмбеддингами, токенизатором и моделью, рассчитанной на два класса. Для построения модели
используется pytorch. Для ускорения вычислений, нижепреведенный код можно запустить в colab с gpu кернелом.
"""

import torch
import transformers
from transformers import BertTokenizer


#%%

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('device name:', torch.cuda.get_device_name(0))

#%%

#Пользуемя pretrained токенизатором
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#%%

def find_length(sentences):
  return max([len(tokenizer.encode(sent)) for sent in sentences])


#%%

#length = 120
train_seq_length = find_length(train_sentences)

#%%

"""
Все токенизированные последовательности должны быть приведены к общей длине при помощи токена ['PAD']. Также к
последовательностям должны быть добавлены токены '[CLS]' и '[SEP]' в начале и конце (до паддинга). Функция encode_plus
возвращает токенизированные последовательности со специальными токенами и attention masks, имеющие такую же форму, как
и получаемый тенсор токенизированных последовательностей и состоящий из 1 и 0, где 0 соответсвует токену ['PAD'], 1 -
всем остальным.
"""
def return_ids_masks(sentences):
  input_ids = []
  attention_masks = []

  for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 120,           # Pad & truncate all sentences.
                        truncation=True,
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  return input_ids, attention_masks

#%%

train_ids, train_masks = return_ids_masks(train_sentences)
targets = torch.tensor(targets)
print(targets.shape)

#%%

#Создаем датасеты для тренировки и валидации
from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(train_ids, train_masks, targets)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


#%%

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

#%%

from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

model.cuda()

#%%

#Тренировочный алгоритм - AdamW (модификация градиентого спуска)
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

#%%

#Применеям learning rate scheduling
from transformers import get_linear_schedule_with_warmup

epochs = 2
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

#%%

#Функция для вычисление accuracy на одном пакете
import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#%%

#Фунция для форматирования времени
import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

#%%

#Файнтюним pretrained модель на двух эпохах. Валидация происходит в конце каждой эпохи.
import random
import numpy as np

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

total_t0 = time.time()

for epoch_i in range(0, epochs):
  print("")
  print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
  print('Training...')
  t0 = time.time()

  total_train_loss = 0

  model.train()
  for step, batch in enumerate(train_dataloader):
    if step % 40 == 0 and not step == 0:
      elapsed = format_time(time.time() - t0)
      print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    model.zero_grad()

    loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

    total_train_loss += loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    scheduler.step()

  avg_train_loss = total_train_loss / len(train_dataloader)
  training_time = format_time(time.time() - t0)

  print("")
  print("  Average training loss: {0:.2f}".format(avg_train_loss))
  print("  Training epcoh took: {:}".format(training_time))
  print("")
  print("Running Validation...")

  t0 = time.time()
  model.eval()

  total_eval_accuracy = 0
  total_eval_loss = 0
  nb_eval_steps = 0

  for batch in validation_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
      (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

    total_eval_loss += loss.item()

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    total_eval_accuracy += flat_accuracy(logits, label_ids)

  avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
  print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

  avg_val_loss = total_eval_loss / len(validation_dataloader)

  validation_time = format_time(time.time() - t0)

  print("  Validation Loss: {0:.2f}".format(avg_val_loss))
  print("  Validation took: {:}".format(validation_time))

  training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
  )

  print("")
  print("Training complete!")

  print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

#%%
#Сохраняем модель
model.save_pretrained('trained_model_params')
tokenizer.save_pretrained('trained_model_params')

#%%
model = BertForSequenceClassification.from_pretrained("../trained_model_params")
tokenizer = BertTokenizer.from_pretrained('trained_model_params')

#%%
model.to(device)
#%%
import pandas as pd

pd.set_option('precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats

#%%
#Dataloader для тестовых данных
test_ids, test_masks = return_ids_masks(test_sentences)

batch_size = 32

test_data = TensorDataset(test_ids, test_masks)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
#%%
def predict(loader, is_probs=False, is_targets=False):
  model.eval()
  predictions = []
  probabilities = [] #
  targets = []
  softmax = torch.nn.Softmax(dim=1)

  for batch in loader:
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
      logits = model(batch[0], token_type_ids=None,
                  attention_mask=batch[1])

      predictions.append(torch.argmax(logits[0], dim=1).detach().cpu().numpy())

      if is_probs:
        probs = softmax(logits[0]).detach().cpu().numpy()
        probabilities.append(probs)
        if is_targets:
          targets.append(batch[2].to('cpu').numpy())

  if is_probs and not is_targets:
    return np.concatenate(probabilities, axis=0)
  elif is_probs and is_targets:
    return np.concatenate(probabilities, axis=0), np.concatenate(targets)
  else:
    return np.concatenate(predictions)

#%%
#Выбираем порог бинаризации по валидационной выборке
val_probs, val_targets = predict(validation_dataloader, is_probs=True, is_targets=True)
#%%
def treshold(targets, probs):
  fpr, tpr, thr = roc_curve(targets, probs[:, 1])
  distances = [math.sqrt(math.pow(fp - 0, 2) + math.pow(tp - 1, 2)) for fp, tp in zip(fpr, tpr)]
  treshold = thr[distances.index(min(distances))]
  return treshold
#%%
t = treshold(val_targets, val_probs)
print(t)
#%%
def adjusted_preds(treshold, probs):
  return [0 if prob < t else 1 for prob in probs[:, 1]]
#%%
test_probs = predict(test_dataloader, is_probs=True)
adjusted_predictions = adjusted_preds(t, test_probs)
#%%
import pandas as pd

predictions_df = pd.DataFrame(adjusted_predictions, columns=['y'])
predictions_df.index.name = 'Id'
#%%
predictions_df.to_csv('adjusted_predictions_df_bert.csv')
#%%
#Предсазания без бинаризации
test_probs = predict(test_dataloader)
#%%
predictions_df = pd.DataFrame(test_probs, columns=['y'])
predictions_df.index.name = 'Id'
#%%
predictions_df.to_csv('not_adjusted_predictions_df_bert.csv')
#%%
#Порог бинаризации по всем данным
all_data_dataloader = DataLoader(
            dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

#%%
alldata_probs, alldata_targets = predict(all_data_dataloader, is_probs=True, is_targets=True)
#%%
t = treshold(alldata_targets, alldata_probs)
#%%
test_probs = predict(test_dataloader, is_probs=True)
adjusted_predictions_all = adjusted_preds(t, test_probs)
#%%
predictions_df = pd.DataFrame(adjusted_predictions_all, columns=['y'])
predictions_df.index.name = 'Id'
#%%
predictions_df.to_csv('allset_adjusted_predictions_df_bert.csv')

