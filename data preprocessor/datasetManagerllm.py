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

# Initialize OpenAI API with the API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize your Weave project
weave.init('generate-llm-labelled-newimdbdataset')

@weave.op()
def generate_rationale(review):
    """
    Generate an abstractive rationale for the sentiment of a review.
    """
    try:
        # Construct the prompt
        prompt = (
            "Your task is to understand a text, explain succinctly in one sentence why it either shows positive or negative emotion, "
            "then give a predicted emotion. For instance, for the given text 'Have just seen the Australian premiere of Shower [Xizhao] "
            "at the Sydney Film Festival. The program notes said it was - A perfect delight -deftly made, touching, amusing, dramatic "
            "and poignantly meaningful. I couldn't agree more.' explanation:  'it represents positive emotion as it notes how the premier "
            "was meaningful and amusing'"
        )
        prompt += f"\nReview: '{review}'\nWrite a rationale for your predicted sentiment."
        
        # Generate the rationale
        rationale_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate rationale"}
            ]
        )
        
        # Extract the rationale from the response
        rationale = rationale_response.choices[0].message.content
        
        return rationale
    except Exception as e:
        print(f"Error generating rationale: {e}")
        return "Rationale generation failed."

@weave.op()
def predict_sentiment_from_rationale(rationale):
    """
    Predict the sentiment based on the generated rationale.
    """
    try:
        # Construct the prompt
        prompt = f"The rationale for the sentiment prediction is: '{rationale}'. Predict the sentiment either 'Positive' or 'Negative'."
        
        # Generate the sentiment prediction
        sentiment_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Predict sentiment"}
            ]
        )
        
        # Extract the sentiment prediction from the response
        sentiment = sentiment_response.choices[0].message.content
        
        return sentiment
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return "Sentiment prediction failed."



def process_entry(entry):
    """
    Process each entry to fit the desired format, including generating rationale and predicting sentiment.
    """
    review = entry["review"]
    rationale = generate_rationale(review)
    sentiment = predict_sentiment_from_rationale(rationale)
    
    formatted_entry = {
        "id": os.urandom(16).hex(),
        "question": review,
        "choices": ["Positive", "Negative"],
        "answer": sentiment,
        "rationale": rationale,
    }
    return formatted_entry

def convert_csv_to_json(input_csv_path, output_train_path):
    """
    Convert a CSV file to train and test JSON files with generated reasoning and predicted sentiments.
    """
    # If the CSV does not have a header
    train_df = pd.read_csv(input_csv_path, nrows=10000)  # nrows counts only data rows
    
    with open(output_train_path, 'w') as f:
        for idx, row in train_df.iloc[9110:].iterrows():
            processed_entry = process_entry(row.to_dict())
            f.write(json.dumps(processed_entry) + '\n')

    print("Finished processing.")

# Paths to the CSV and output JSONL files
input_csv_path = './IMDB Dataset.csv'
output_train_path = '.llm_output_train3.json'

# Execute the conversion
convert_csv_to_json(input_csv_path, output_train_path)
