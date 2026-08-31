import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")

# Read raw text
with open("../Dataset/raw_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Original Text:\n")
print(text)

# Cleaning
text = text.lower()
text = re.sub(r"\d+", "", text)
text = re.sub(r"[^\w\s]", "", text)   # remove punctuation, keep spaces
text = re.sub(r"\s+", " ", text).strip()

print("\nCleaned Text:\n")
print(text)

# Tokenization
sentences = sent_tokenize(text)
words = word_tokenize(text)

# Save outputs
pd.DataFrame({"Clean_Text": [text]}).to_csv("../Dataset/clean_text.csv", index=False)
pd.DataFrame({"Token": words}).to_csv("../Dataset/tokens.csv", index=False)

print("\nFiles saved successfully.")


