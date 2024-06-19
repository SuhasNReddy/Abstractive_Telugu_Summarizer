
import json
# from summary_dataset import generatesummary

# Create an empty list to store parsed JSON objects
parsed_data = []

# Read data from input file line by line
with open('telugu_test.json', 'r', encoding='utf-8') as file:
    # Read and process each line as a separate JSON object
    for line in file:
        try:
            # Parse JSON object from the line and append it to the list
            data = json.loads(line)
            # Remove specific keys from the JSON object
            del data['url']
            del data['title']
            # del data['summary']
            
            parsed_data.append(data)
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")



# Create a new JSON file and write the updated data (list of parsed JSON objects)
with open('preprocessed_Dataset.json', 'w', encoding='utf-8') as file:
    # Convert the list of parsed JSON objects to a JSON string and write it to the file
    json.dump(parsed_data, file, ensure_ascii=False, indent=4)

print("Data added and saved to preprocessed_Dataset.json")
