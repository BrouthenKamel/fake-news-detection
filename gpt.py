import openai
import pandas as pd

# Charger la clé API
openai.api_key = 'your-api-key'

# Charger le dataset
df = pd.read_csv('/Users/nadine/Desktop/Nad-2/fake_news_tweets.csv', delimiter=';')
print(df.columns)

# Fonction pour interroger le modèle LLM
def classify_tweet(tweet):
    response = openai.Completion.create(
        engine="text-davinci-003",  # ou un autre modèle disponible
        prompt=f"Classify the following tweet as 'fake news' or 'real news':\n\n{tweet}\n\nClassification:",
        max_tokens=1
    )
    classification = response['choices'][0]['text'].strip().lower()
    return 'fake' if 'fake' in classification else 'real'



# Appliquer la classification à chaque tweet
df['LLM_Classification'] = df.apply(lambda row: classify_tweet(row['Tweet']), axis=1)

# Sauvegarder les résultats dans un nouveau fichier CSV
df.to_csv('/Users/nadine/Desktop/Nad-2/classified_fake_news_tweets.csv', index=False)
