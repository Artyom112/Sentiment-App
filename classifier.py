from transformers import BertForSequenceClassification, BertTokenizer
import torch


class Classifier():
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained("trained_model_params")
        self.tokenizer = BertTokenizer.from_pretrained("trained_model_params")

    def return_ids_masks(self, sentences):
        input_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=120,  # Pad & truncate all sentences.
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def predict(self, sentence):
        softmax = torch.nn.Softmax(dim=1)
        ids, masks = self.return_ids_masks(sentence)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ids, token_type_ids=None, attention_mask=masks)
            probs = softmax(logits[0])
            prob_pos = probs[0, 1].item()
        if prob_pos < 0.05:
            return 'negative'
        elif prob_pos < 0.3:
            return 'mostly negative'
        elif prob_pos < 0.5:
            return 'between negative and neutral'
        elif prob_pos < 0.7:
            return 'between positive and neutral'
        elif prob_pos < 0.95:
            return 'mostly positive'
        else:
            return 'positive'

