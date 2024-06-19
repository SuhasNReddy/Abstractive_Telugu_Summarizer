import string
import re
import json
import numpy as np
from Lemmatization.custom_lemmatizer import telugu_custom_lemmatizer
from Lemmatization.replacing_original_lemma import replacing_original_lemma
from NER.ner import is_named_entity


# Open a file for writing the output
output_file_path = 'output.txt'
output_file = open(output_file_path, 'w', encoding='utf-8')

with open('input_text.json', 'r', encoding='utf-8') as input_file:
    input_data = json.load(input_file)

with open("D:\\sem 5\\NLP\\project\\NLP_END\\NLP_END\\Stop_Words\\c_jsonformat.json", 'r', encoding='utf-8') as stop_words_file:

    stop_words_data = json.load(stop_words_file)

InputText = input_data["text"]
stopwords = stop_words_data["stopwords"]


def remove_punctuations(InputText):
    regex = r"[!\"#\$%&\'\(\)\*\+,-\/:;<=>\?@\[\\\]\^_`{\|}~]"
    replaced_with = " "
    InputText = re.sub(regex, lambda m: "." if m.group(
        0) == "." else " ", InputText, 0, re.MULTILINE)
    return InputText


def remove_sw(sentences):
    all_words_without_sw = []
    sentences_without_sw = []

    for sentence in sentences:
        words = sentence.split(' ')
        words_without_sw = [word for word in words if not word in stopwords]

        all_words_without_sw.extend(words_without_sw)

        sentences_without_sw.append(
            [word for word in words_without_sw if word.strip() != ''])

    for w in all_words_without_sw:
        if ((w == ' ') or (w == '')):
            all_words_without_sw.remove(w)
    return all_words_without_sw, sentences_without_sw


InputText = remove_punctuations(InputText)
InputText = InputText.replace('\u200c', '')
output_file.write('---------------------------------------------------------------------------\nInput Text\n---------------------------------------------------------------------------\n')
output_file.write(InputText + '\n')
sentences = InputText.split('. ')
output_file.write("\nTotal No. of sentences : " + str(len(sentences)) + '\n')

tockens_without_sw, sentences_without_sw = remove_sw(sentences)
output_file.write('---------------------------------------------------------------------------\nAll words without stop words\n---------------------------------------------------------------------------\n')
output_file.write(str(tockens_without_sw) + '\n')

# Lemmatization
sentences_without_sw_lemmatized = [
    [replacing_original_lemma(word) for word in sentence] for sentence in sentences_without_sw]

lemmatized_tokens = []
for sentence in sentences_without_sw_lemmatized:
    for word in sentence:
        lemmatized_tokens.append(word)

output_file.write('---------------------------------------------------------------------------\nAll words After lemmatization\n---------------------------------------------------------------------------\n')
output_file.write(str(lemmatized_tokens) + '\n')

output_file.write('---------------------------------------------------------------------------\nSentences After lemmatization\n---------------------------------------------------------------------------\n')
output_file.write(str(sentences_without_sw_lemmatized) + '\n')

# TF-IDF
def calculate_tf(tokens):
    tf_dict = {}
    for token in tokens:
        tf_dict[token] = tf_dict.get(token, 0) + 1
    return tf_dict

# Calculate TF for lemmatized_tokens
tf = calculate_tf(lemmatized_tokens)

def calculate_idf(documents, term):
    document_count = len(documents)
    term_occurrences = sum(1 for doc in documents if term in doc)
    if term_occurrences > 0:
        return 1 + np.log(document_count / term_occurrences)
    else:
        return 1  # Avoid division by zero

# We have multiple documents (sentences) for IDF calculation
# documents = [lemmatized_tokens_sentence_1, lemmatized_tokens_sentence_2, ...]
idf = {term: calculate_idf(sentences_without_sw_lemmatized, term) for term in set(lemmatized_tokens)}



# Apply normalization to TF values
normalized_tf = {term: tf[term] / max(tf.values()) for term in tf}

output_file.write('---------------------------------------------------------------------------\nNormalized TF\n---------------------------------------------------------------------------\n')
output_file.write(str(normalized_tf) + '\n')

# Calculate sentence scores using TF-IDF and Named Entity Recognition
sentence_scores_combined = []

for sentence in sentences_without_sw_lemmatized:
    score_tfidf = sum(normalized_tf[word] * idf.get(word, 1) for word in sentence)
    score_ner = sum(1 for word in sentence if is_named_entity(word))
    print(score_ner)
    # Combine TF-IDF and NER scores (you can adjust the weights as needed)
    combined_score = 0.7 * score_tfidf + 0.3 * score_ner
    sentence_scores_combined.append(combined_score)


# Select the top four sentences using combined scores
top_sentences_combined = sorted(range(len(sentence_scores_combined)), key=lambda i: sentence_scores_combined[i], reverse=True)[:4]

output_file.write('---------------------------------------------------------------------------\nSentence - Combined Score (TF-IDF + NER)\n---------------------------------------------------------------------------\n')
for i, score in enumerate(sentence_scores_combined):
    output_file.write(sentences[i] + ' ---- ' + str(score) + '\n')

# Display the selected top sentences for combined scores and write to the output file
output_file.write('\n---------------------------------------------------------------------------\nTop Sentences (Combined Score)\n---------------------------------------------------------------------------\n')
summary = ""
for index in top_sentences_combined:
    output_file.write(sentences[index] + ' ---- ' + str(sentence_scores_combined[index]) + '\n')
    summary += sentences[index] + '. '

output_file.write('\n---------------------------------------------------------------------------\nFinal Summary\n\n\n')
output_file.write(summary)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def train_classifier(train_data):
    # Tokenization and vectorization
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([sent for sent, _ in train_data])
    y_train = [sense for _, sense in train_data]

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    return classifier, vectorizer

def apply_word_sense_disambiguation(text, classifier, vectorizer):
    # Tokenize and vectorize the input text
    text_vectorized = vectorizer.transform([text])
    
    # Predict the sense using the trained classifier
    predicted_sense = classifier.predict(text_vectorized)
    
    return predicted_sense[0]

def disambiguate_text(input_text, classifier, vectorizer):
    sentences = input_text.split('.')
    disambiguated_sentences = []

    for sentence in sentences:
        if sentence.strip():  # Check if the sentence is not empty
            disambiguated_sense = apply_word_sense_disambiguation(sentence, classifier, vectorizer)
            disambiguated_sentences.append(f"{sentence.strip()} ({disambiguated_sense})")

    return '. '.join(disambiguated_sentences)

# Example Telugu dataset with annotated senses
telugu_data = [
    ("పాకిస్థాన్ జరిగిన మ్యాచ్ లో విరాట్ కోహ్లి అసాధారణ పోరాటంతో భారత్ విజయాన్ని అందించాడు.", "క్రికెట్_మ్యాచ్"),
    ("హౌస్లో ఉన్న కంటెస్టెంట్సే పనికి మాలిన వాళ్లని నానా తిట్టిన నాగార్జున.", "బిగ్_బాస్"),
    ("ప్రపంచవ్యాప్తంగా ప్రశంసల వర్షం కురుస్తోంది.", "ప్రశంస"),
    ("చిన్నవి బానిసత్తాలు అందించాడు.", "బానిసత్తాలు"),
    ("టీ20 వరల్డ్ కప్‌లో ఛేజింగ్‌లో అసాధారణమైన బ్యాటింగ్ రికార్డ్ ఉంది.", "బ్యాటింగ్_రికార్డ్"),
    ("అతను సినిమా నటుడి.", "నటుడు"),
    ("అతను ఎండు తరహా ఆధునిక కళాకారుడు.", "కళాకారుడు"),
    ("అతను రచనా సారధి.", "రచనా_సారధి"),
    ("అతను ప్రసార ప్రధికారి.", "ప్రసార_ప్రధికారి"),
]

# Extended Telugu dataset with annotated senses
extended_telugu_data = [
    ("హౌస్లో ఉన్న కంటెస్టెంట్సే పనికి మాలిన వాళ్లని నానా తిట్టులు తిట్టిన నాగార్జున", "బిగ్_బాస్"),
    ("శనివారం నానిగారు బిగ్ బాస్ 7 ఎపిసోడ్ 28లో హోస్ట్ నాగార్జున కల్ట్ సినిమా చూపించేశారు", "టెలివిజన్_షో"),
    ("ఆ పనికి మాలిన వాళ్ల అభిప్రాయాన్ని తీసుకుని జడ్జిమెంట్ ఇవ్వడం తప్పనిపించింది.", "న్యాయాధీశు"),
]

# Combine both the original and extended Telugu datasets
telugu_data.extend(extended_telugu_data)

# Split the combined dataset into training and testing sets
train_data, _ = train_test_split(telugu_data, test_size=0.2, random_state=42)

# Train the classifier
classifier, vectorizer = train_classifier(train_data)

# Example text for disambiguation
example_text =summary
disambiguated_text = disambiguate_text(example_text, classifier, vectorizer)


def summarize_with_disambiguation(disambiguated_text):
    # Split the disambiguated text into sentences
    sentences = disambiguated_text.split('. ')

    # Create a dictionary to store sentences for each sense
    sense_sentences = {}

    # Extract sentences for each unique sense
    for sentence in sentences:
        # Extract the sense from the sentence
        parts = sentence.split('(')
        if len(parts) == 2:
            sense = parts[1].rstrip(')')
            sense_sentences.setdefault(sense, []).append(parts[0])

    # Generate a summary by selecting one sentence for each sense
    summary_sentences = [f"{sense}: {sentences[0]}" for sense, sentences in sense_sentences.items()]

    return '. '.join(summary_sentences)

# Example usage
disambiguated_summary = summarize_with_disambiguation(disambiguated_text)

output_file.write('\n\n\n--------------------------------Disambigute Summary\n\n')
output_file.write(disambiguated_summary)

def summarize_and_disambiguate(disambiguated_text):
    # Split the input text into sentences
    return disambiguated_summary

# Close the output file
output_file.close()
# summarizer.py
