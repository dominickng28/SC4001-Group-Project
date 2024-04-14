import json
import os
import time

import openai
import pandas as pd
import weave
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

# Initialize OpenAI and Weave APIs
openai.api_key = os.getenv('OPENAI_API_KEY')
weave.init('generate-gt-Yelpdataset')

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
    if entry["sentiment"] == 1:
        sentiment = "Negative"
    else:
        sentiment = "Positive" 

    choices = ["Positive", "Negative"]
    rationale = generate_explanations(review, sentiment)
    
    formatted_entry = {
        "id": os.urandom(16).hex(),  # Generate a pseudo-random ID
        "question": review,
        "choices": choices,
        "answer": sentiment,  
        "rationale": rationale,
    }
    return formatted_entry

def convert_csv_to_json(input_csv_path, output_train_path):
    """
    Convert a CSV file to train and test JSON files with generated reasoning.
    """
    train_df = pd.read_csv(input_csv_path, header=None, names=["sentiment","review"])
    
    # Process and save the train set starting from row 701
    with open(output_train_path, 'w') as f:
        for idx, row in train_df.iloc[2966:].iterrows():
            processed_entry = process_entry(row.to_dict())
            f.write(json.dumps(processed_entry) + '\n')


    
    print("Finished processing.")

# Paths for CSV and output JSONL files
input_csv_path = './train.csv'
output_train_path = './gtoutput_train2.json'

# Execute the conversion
convert_csv_to_json(input_csv_path, output_train_path)