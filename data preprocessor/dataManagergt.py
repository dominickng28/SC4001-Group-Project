import json
import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()
import time

import openai
import weave

# Initialize OpenAI API with the API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize your Weave project
weave.init('generate-gt-labelled-imdbdataset')

@weave.op()  # Keep this for tracking LLM interactions
def safe_generate_explanations(review, sentiment):
    try:
        return generate_explanations(review, sentiment)
    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return "Reasoning generation failed.", "Reasoning generation failed."
    
@weave.op()
def generate_explanations(review, sentiment):
    """
    Generate both abstractive and extractive explanations for a given review and sentiment.
    """ 
    try:
        # Generate abstractive explanation
        abstractive_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Review: '{review}'\nSentiment: {sentiment}\nWrite a short rationale for why the sentiment is {sentiment}."}
            ]
        )
        rationale = abstractive_response.choices[0].message.content

        return rationale
    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return "Reasoning generation failed."

def process_entry(entry):
    """
    Process each entry to fit the desired format, including generating reasoning.
    """
    review = entry["review"]
    sentiment = entry["sentiment"].capitalize()

    choices = ["Positive", "Negative"]
    rationale = generate_explanations(review, sentiment)
    
    formatted_entry = {
        "id": os.urandom(16).hex(),  # Generate a pseudo-random ID
        "review": review,
        "choices": choices,
        "answer": sentiment,
        "rationale": rationale
    }
    return formatted_entry

# Update the function signatures and content according to the corrections
# The main structure of `convert_csv_to_json` remains the same
def convert_csv_to_json(input_csv_path, output_train_path):
    """
    Convert a CSV file to train and test JSON files with generated reasoning.
    """

    # Load the entire DataFrame
    train_df = pd.read_csv(input_csv_path, nrows=10000)  # nrows counts only data rows
    
    # Process and save the train set
    with open(output_train_path, 'w') as f:
        for _, row in train_df.iterrows():
            processed_entry = process_entry(row.to_dict())
            f.write(json.dumps(processed_entry) + '\n')
    
    print("Finished processing.")

# Ensure you replace the paths with actual paths where your CSV is located and where you want the JSONL files to be saved
input_csv_path = './IMDB Dataset.csv'
output_train_path = './gt_output_train.json'

# Execute the conversion
convert_csv_to_json(input_csv_path, output_train_path)