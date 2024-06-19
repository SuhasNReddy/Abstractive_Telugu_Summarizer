from flask import Flask, render_template, request, redirect, url_for
import json

app = Flask(__name__)

# Path to your JSON file
json_file_path = 'D:\\sem 5\\NLP\\project\\NLP_END\\NLP_END\\input_text.json'

def load_data():
    try:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {"text": ""}
    return data

def save_data(data):
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def summarize_text(input_text):
    # Replace this with your actual summarization logic
    # For now, it echoes the input text as a placeholder
    return input_text

@app.route('/')
def index():
    data = load_data()
    return render_template('index.html', input_text=data.get('text', ''), summary='')

@app.route('/update', methods=['POST'])
def update():
    input_text = request.form['inputText']
    data = load_data()
    data['text'] = input_text
    save_data(data)

    # Summarize the entered text
    summary = summarize_text(input_text)

    return render_template('index.html', input_text=input_text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
