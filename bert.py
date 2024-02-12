import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tqdm import tqdm

# Charger le dataset
df = pd.read_csv('/Users/nadine/Desktop/MIAGE/Fake-News/fake_news_tweets.csv', delimiter=';')
print("Colonnes du dataset :", df.columns)

# Charger le modèle BERT et le tokenizer
model_name = "bert-base-uncased"
model = TFBertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Définir les étiquettes
labels = ['real news', 'fake news']

# Fonction pour classer le tweet comme "fake news" ou "real news"
def classify_tweet(tweet):
    # Encodage du tweet avec le tokenizer BERT
    inputs = tokenizer(tweet, return_tensors="tf", truncation=True, padding=True)
    # Passage du tweet encodé au modèle BERT pour obtenir les prédictions
    outputs = model(inputs)
    # Récupération des scores de chaque classe
    logits = outputs.logits
    # Sélection de la classe avec le score le plus élevé comme prédiction
    predicted_class_index = tf.argmax(logits, axis=1).numpy()[0]
    # Récupération de l'étiquette correspondant à la classe prédite
    predicted_class = labels[predicted_class_index]
    return predicted_class

# Appliquer la classification à chaque tweet
classified_tweets = []
# Utilisation de tqdm pour afficher une barre de progression
for tweet in tqdm(df['Tweet']):
    classified_tweet = classify_tweet(tweet)
    classified_tweets.append(classified_tweet)

# Ajouter les résultats de la classification à la DataFrame
df['Classification'] = classified_tweets

# Sauvegarder les résultats dans un nouveau fichier CSV
output_file_path = '/Users/nadine/Desktop/classified_fake_news_tweets.csv'
df.to_csv(output_file_path, index=False)
print("Résultats sauvegardés dans :", output_file_path)
