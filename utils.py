# utils.py
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import re

class Summarizer:
    def __init__(self, classifier_model_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = tf.keras.models.load_model("bert_extractive_classifier_day8.h5")

    def sentence_tokenizer(self, text):
        return [s.strip() for s in re.split(r'۔|\n|\.', text) if s.strip()]

    def get_cls_embeddings(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors='tf', padding=True, truncation=True, max_length=128)
        outputs = self.bert_model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embeddings.numpy()

    def predict(self, text, threshold=0.5):
        sentences = self.sentence_tokenizer(text)
        if not sentences:
            return ""

        embeddings = self.get_cls_embeddings(sentences)
        probs = self.classifier.predict(embeddings).flatten()

        # Select sentences with probability > threshold
        selected_sents = [sent for sent, prob in zip(sentences, probs) if prob > threshold]

        # If none selected, fallback to first sentence
        if not selected_sents:
            selected_sents = [sentences[0]]

        summary = "۔ ".join(selected_sents)
        return summary
