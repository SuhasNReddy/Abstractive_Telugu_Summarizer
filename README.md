# Abstractive Text Summarization for Telugu Language

## Description

This project focuses on implementing abstractive text summarization for the Telugu language using Natural Language Processing (NLP). Abstractive summarization involves taking long Telugu text inputs and generating concise summaries in a few lines.

## How to Run

### Step-by-Step Instructions:

1. **Download and Extract the Project:**
   - Download the project zip file and extract it to your desired location.

2. **Install Required Libraries:**
   - Ensure `scikit-learn` is installed by running the following command:
     ```
     pip install scikit-learn
     ```

3. **Configure Stop Words File:**
   - Replace the path of `stop_words_file` in `Text_summarization.py` with the absolute path of the stop words file located on your system.

4. **Configure Lemmatization File:**
   - Replace the path of `source_lemma_preprocessing.py` in `replacing_original_lemma.py` located in the `lemmatization` folder with the absolute path of `source_lemma_preprocessing.py` on your system.

5. **Run the Website:**
   - Execute the following command to start the website:
     ```
     python run.py
     ```
   - This command will launch the website where you can input Telugu text and view the generated summaries on the webpage.

## Additional Notes

- Ensure all dependencies are installed and paths are correctly configured before running the application.
- This project uses abstractive summarization techniques tailored for the Telugu language, leveraging NLP libraries and preprocessing scripts.
