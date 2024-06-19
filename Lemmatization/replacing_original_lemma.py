def replacing_original_lemma(input_word):
    # Define the path to the modified Telugu lemma file
    file_path = "D:\\sem 5\\NLP\\project\\NLP_END\\NLP_END\\Lemmatization\\preprocessed_source_lemma.txt"

    # Initialize a dictionary to store word replacements
    word_replacements = {}

    # Read data from the modified Telugu lemma file and populate the dictionary
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" : ")
            if len(parts) == 2:
                word_from_file, word_to_file = parts[0], parts[1]
                word_replacements[word_from_file] = word_to_file

    # Check if the input word is in the dictionary, and replace it if found
    if input_word in word_replacements:
        replaced_word = word_replacements[input_word]
    else:
        replaced_word = input_word

    # Write the input word and its replacement to the specified output file``
    # with open(output_file_path, "a", encoding="utf-8") as output_file:
    #     output_file.write(f"Input Word: {input_word}\n")
    #     output_file.write(f"Replaced Word: {replaced_word}\n\n")

    # Return the replaced word
    return replaced_word

# Example usage:
# input_word = "క్షణాలకన్నా"
# output_file_path = "word_replacements.txt"
# result = replacing_original_lemma(input_word, output_file_path)
