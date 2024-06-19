import json

with open('collected_stopwords.txt', 'r',encoding='utf-8') as file:
    words = file.read().split()

# Wrapping each word in double quotes
quoted_words = ['"{}"'.format(word) for word in words]

# Creating a dictionary with words as keys and indices as values
word_dict = {f'Word_{i+1}': word for i, word in enumerate(quoted_words)}

# Creating the final JSON object with a "stopwords" key
final_json = {"stopwords": words}

# Writing the final JSON object to a file
with open('c_jsonformat.json', 'w',encoding='utf-8') as json_file:
    json.dump(final_json, json_file, ensure_ascii=False,indent=4,separators=(',', ': '))
