import openai
import pandas as pd
import argparse
import os

# Function to query GPT-4 for terms
def query_gpt4_for_terms(column_name, sample_data, verbose=True):
    prompt = f"Based on the following data sample from the column '{column_name}':\n{sample_data}\nSuggest appropriate terms for fuzzy sets in the format of a Python list."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    terms = response.choices[0].text.strip()
    if verbose:
        print(f"GPT-4 suggested terms for '{column_name}': {terms}")
    return eval(terms)  # Caution: This is assuming GPT-4 returns valid Python list syntax.

# Main function to process dataset and generate terms dictionary
def generate_terms_dict(file_path, api_key, verbose=True):
    openai.api_key = api_key
    data = pd.read_csv(file_path)
    terms_dict = {}

    for column in data.columns:
        if data[column].dtype not in [float, int]:
            continue

        sample_data = data[column].sample(n=min(10, len(data))).to_list()
        terms = query_gpt4_for_terms(column, sample_data, verbose)
        terms_dict[column] = terms

    return terms_dict

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Generate terms dictionary for a dataset using GPT-4.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing the dataset.")
    parser.add_argument("api_key", type=str, help="OpenAI API key.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    
    args = parser.parse_args()
    
    terms_dict = generate_terms_dict(args.file_path, args.api_key, args.verbose)

    if args.verbose:
        print("Generated terms dictionary:")
        for column, terms in terms_dict.items():
            print(f"'{column}': {terms}")
    
    # Save the terms dictionary to a file
    with open("terms_dict.py", "w") as f:
        f.write("terms_dict = {\n")
        for column, terms in terms_dict.items():
            f.write(f"    '{column}': {terms},\n")
        f.write("}\n")

if __name__ == "__main__":
    main()
