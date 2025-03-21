import os
import nltk
import time
import torch
import random
import shutil
import numpy as np
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import nlpaug.augmenter.word as naw
from spellchecker import SpellChecker
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,Trainer, TrainingArguments, EarlyStoppingCallback


beginning_time = time.time()
# Télécharger les ressources NLTK nécessaires
nltk.download('averaged_perceptron_tagger')
nltk.download('ppdb')
nltk.download('punkt')
nltk.download('punkt_tab')
spell = SpellChecker()

# ============================
# 1️ CHARGEMENT DU DATASET
# ============================
start_time = time.time()
print("\n Chargement du dataset...")
dataset = load_dataset("dair-ai/emotion")

# Diviser le dataset  (avant augmentation)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_data = split_dataset['train']
val_data = split_dataset['test']
test_data = dataset["test"]

# Vérification des données initiales
label_counts = Counter(train_data["label"])
print("\n Distribution des labels AVANT augmentation:")
for label, count in sorted(label_counts.items()):
    print(f"Label {label}: {count}")

print(f"\n Dataset chargé en {time.time() - start_time:.2f} secondes.\n")

# ============================
# 2️ AUGMENTATION DES DONNÉES
# ============================
start_time = time.time()
# Dictionnaire des labels pour affichage des émotions
label_names = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}
# Définir les règles d'augmentation pour chaque label
augmentation_rules = {
    2: 2,  # Love → 2 phrases supplémentaires
    3: 1,
    4: 1,
    5: 3  # Surprise → 4 phrases supplémentaires
}

# Générer dynamiquement le message d'augmentation
augmentation_message = ", ".join([f"'{label_names[label]}' (x{times})" for label, times in augmentation_rules.items()])

# Affichage dynamique
print(f"\n Augmentation en cours : {augmentation_message}...")

# Initialiser wordnet
aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=1)

# Fonction pour détecter les mots importants
def get_important_words(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    exclude_words = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
    important_words = [word for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ', 'RB')) and word.lower() not in exclude_words ]

    return important_words

# Fonction pour vérifier si le mot existe
def is_valid_word(word):
    return word.lower() in spell

# Fonction pour générer plusieurs variations d'une phrase
def augment_text_multiple(text, num_variations):
    important_words = get_important_words(text)

    if not important_words:
        return [text] * num_variations  #  Retourne le texte original si il n'y as pas de mot valide

    augmented_texts = set()

    for _ in range(10):  # Essayer différents mots jusqu'à 10 fois
        word_to_replace = random.choice(important_words)
        augmented_version = aug.augment(word_to_replace)

        if isinstance(augmented_version, list):
            augmented_version = augmented_version[0]

        # Vérifier si le mot augmenté est valide
        if not is_valid_word(augmented_version):
            continue  # Ignorer les remplacements invalides

        new_text = text.replace(word_to_replace, augmented_version, 1)

        #  Correction : S'assurer que le nouveau texte est valide et unique
        if new_text != text and new_text not in augmented_texts and new_text.strip():
            augmented_texts.add(new_text)

        if len(augmented_texts) >= num_variations:
            break

    return list(augmented_texts) if augmented_texts else [text]  #  S'assurer qu'au moins le texte original est retourné





def log_augmentation_examples(train_data, augmentation_rules, num_examples=10, log_file_path="augmentation_log.txt"):
    """
    Journalise des exemples sélectionnés aléatoirement avant et après l'augmentation.

    Paramètres :
    - train_data : Le dataset d'entraînement original avant augmentation.
    - augmentation_rules : Dictionnaire spécifiant combien de fois chaque label doit être augmenté.
    - num_examples : Nombre d'exemples aléatoires par label à enregistrer.
    - log_file_path : Chemin du fichier de journalisation.
    """
    log_entries = []

    for label, num_variations in augmentation_rules.items():
        if num_variations == 0:
            continue  # Ignorer les labels qui ne sont pas augmentés

        # Filtrer le dataset pour le label actuel
        label_subset = train_data.filter(lambda example: example["label"] == label)

        # Convertir le dataset en liste avant l'échantillonnage
        label_subset_list = list(label_subset)

        # Vérifier qu'il y a des exemples à échantillonner
        if len(label_subset_list) == 0:
            continue

        # Sélectionner aléatoirement des exemples pour le journal
        selected_examples = random.sample(label_subset_list, min(num_examples, len(label_subset_list)))

        for example in selected_examples:
            new_texts = augment_text_multiple(example["text"], num_variations)

            for new_text in new_texts:
                if new_text != example["text"]:  # Vérifier que le texte a réellement changé
                    log_entries.append(f"\n Before: {example['text']}\n After: {new_text}\n")

    # Enregistrer les entrées de journal dans un fichier
    if log_entries:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n=== LOGGING {num_examples} EXAMPLES PER LABEL ===\n")
            log_file.writelines(log_entries)
            log_file.write("\n" + "=" * 50 + "\n")

    print(f"\n Exemples d'augmentation enregistrés dans {log_file_path}\n")


def augment_target_labels(train_data, augmentation_rules):
    augmented_data = []

    for label, num_variations in augmentation_rules.items():
        if num_variations == 0:
            continue
        label_subset = train_data.shuffle(seed=42).filter(lambda example: example["label"] == label)

        # Ajout d'une barre de progression tqdm pour l'augmentation
        for example in tqdm(label_subset, desc=f" Augmentation du label {label} ({num_variations}x)"):
            new_texts = augment_text_multiple(example["text"], num_variations)

            for new_text in new_texts:
                if new_text != example["text"]:  # Ajouter uniquement si le texte a réellement changé
                    new_example = example.copy()
                    new_example["text"] = new_text
                    augmented_data.append(new_example)

    # Si aucune donnée n'a été augmentée, on retourne simplement le dataset original
    if not augmented_data:
        print("Aucune augmentation de données appliquée. Retour au dataset original.")
        return train_data

    augmented_dataset = Dataset.from_list(augmented_data, features=train_data.features)
    final_dataset = concatenate_datasets([train_data, augmented_dataset])
    return final_dataset


log_augmentation_examples(train_data, augmentation_rules, num_examples=10)

# Appliquer l'augmentation
augmented_train_dataset = augment_target_labels(train_data, augmentation_rules)

# Vérification finale
print("\n Distribution des labels APRÈS augmentation :")
label_counts_new = Counter(augmented_train_dataset["label"])
for label, count in sorted(label_counts_new.items()):
    print(f"Label {label}: {count}")

print(f"\n Augmentation terminée en {time.time() - start_time:.2f} secondes.\n")

# ============================
# 3 ENTRAÎNEMENT DU MODÈLE
# ============================
start_time = time.time()
print("\n Entraînement du modèle en cours...")

# Charger le tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Fonction de tokenisation
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# Tokenization avec tqdm bar de chargement bar
print("\n Tokenisation des datasets...")
tokenized_train = augmented_train_dataset.map(tokenize_function, batched=True, desc="Tokenizing train")
tokenized_val = val_data.map(tokenize_function, batched=True, desc="Tokenizing validation")
tokenized_test = test_data.map(tokenize_function, batched=True, desc="Tokenizing test")

# Charger le modèle
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./augmented_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1", #accuracy
)

# Fonction de calcul des métriques
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Entraînement du modèle
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

save_dir = "./fine_tuned_model_best_preprocess_text_"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir) # Supprime le dossier existant
    print("fichier fine_tuned_model_best_preprocess_text déjà existant, fichier supprimé")
os.makedirs(save_dir)  # Crée un dossier vide
trainer.save_model(save_dir)
print("fichier fine_tuned_model_best_preprocess_text crée")


print(f"\n Entraînement terminé en {time.time() - start_time:.2f} secondes.\n")

# ============================
# 4 ÉVALUATION DU MODÈLE
# ============================
start_time = time.time()
print("\n Évaluation du modèle sur le jeu de test...")

predictions_output = trainer.predict(tokenized_test)
preds = np.argmax(predictions_output.predictions, axis=1)
labels = predictions_output.label_ids

# Classification report
target_names = tokenized_test.features["label"].names
report = classification_report(labels, preds, target_names=target_names)

print("\n=== RAPPORT FINAL DE CLASSIFICATION ===")
print(report)

# Matrice de confusion
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Final Confusion Matrix")
plt.show()

end_time = time.time()

print(f"\n Évaluation terminée en {time.time() - start_time:.2f} secondes.\n")

print(f"\n Tout l'entrainement a pris {end_time - beginning_time:.2f} secondes.\n")