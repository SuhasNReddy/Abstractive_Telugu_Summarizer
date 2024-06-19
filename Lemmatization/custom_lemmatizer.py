import re

def telugu_custom_lemmatizer(word):
    # Define custom lemmatization rules using regular expressions
    rules = [
        (r'లు$', ''),       
        (r'ను$', ''),       
        (r'ం$', ''),        
        (r'లో$', ''),      
        (r'కి$', ''),      
        (r'ని$', ''),      
        (r'కు$', ''),      
        (r'తో$', ''),      
        (r'డు$', 'డు'),   
        (r'డి$', 'డు'),   
        (r'డుకు$', 'డు'),
    ]

    # Apply rules to the word
    for pattern, replacement in rules:
        word = re.sub(pattern, replacement, word)

    return word

# Sample Telugu words (add more as needed)
# telugu_words = ["ప్రతిష్ఠానం", "పర్వం", "కూడలు", "కోడి", "పాఠశాల", "పడి", "పడుకు"]

# # Lemmatize Telugu words
# lemmatized_words = [telugu_custom_lemmatizer(word) for word in telugu_words]

# # Define the file path where you want to store the lemmatized words
# output_file_path = "lemmatized_telugu_words.txt"

# # Write the lemmatized words to the file
# with open(output_file_path, "w", encoding="utf-8") as file:
#     for word in lemmatized_words:
#         file.write(word + "\n")

# # Print a message to confirm that the words have been stored
# print(f"Lemmatized words have been written to {output_file_path}")
