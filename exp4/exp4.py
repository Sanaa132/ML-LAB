
# ------- DATASET -------
dataset = [
    ("very good movie", "Positive"),
    ("excellent acting", "Positive"),
    ("good story", "Positive"),
    ("amazing direction", "Positive"),
    ("enjoyable film", "Positive"),
    ("bad movie", "Negative"),
    ("poor acting", "Negative"),
    ("boring story", "Negative"),
    ("terrible direction", "Negative"),
    ("not enjoyable", "Negative")
]

# ------- STEP 1: CALCULATE PRIORS -------
total = len(dataset)
positive_docs = sum(1 for text, label in dataset if label == "Positive")
negative_docs = total - positive_docs

P_pos = positive_docs / total
P_neg = negative_docs / total

print("Prior Probabilities")
print("P(Positive) =", P_pos)
print("P(Negative) =", P_neg)
print("-" * 40)


# ------- STEP 2: BUILD VOCABULARY + WORD COUNTS -------
from collections import defaultdict

pos_words = []
neg_words = []

for text, label in dataset:
    words = text.split()
    if label == "Positive":
        pos_words.extend(words)
    else:
        neg_words.extend(words)

vocab = set(pos_words + neg_words)
V = len(vocab)

print("Vocabulary =", vocab)
print("Vocabulary Size =", V)
print("-" * 40)

# Word frequency dictionaries
pos_freq = defaultdict(int)
neg_freq = defaultdict(int)

for word in pos_words:
    pos_freq[word] += 1

for word in neg_words:
    neg_freq[word] += 1

total_pos_words = len(pos_words)
total_neg_words = len(neg_words)


# ------- FUNCTION: CONDITIONAL PROBABILITY WITH LAPLACE SMOOTHING -------
def word_likelihood(word, label):
    if label == "Positive":
        return (pos_freq[word] + 1) / (total_pos_words + V)
    else:
        return (neg_freq[word] + 1) / (total_neg_words + V)


# ------- STEP 3 & 4: CLASSIFY NEW SENTENCE -------
test_sentence = "good acting"
words = test_sentence.split()

P_test_pos = P_pos
P_test_neg = P_neg

for w in words:
    P_test_pos *= word_likelihood(w, "Positive")
    P_test_neg *= word_likelihood(w, "Negative")

print("Sentence:", test_sentence)
print("\nProbability it is POSITIVE =", P_test_pos)
print("Probability it is NEGATIVE =", P_test_neg)

print("\nFinal Prediction:")
if P_test_pos > P_test_neg:
    print(">>> Review is POSITIVE ")
else:
    print(">>> Review is NEGATIVE ")
