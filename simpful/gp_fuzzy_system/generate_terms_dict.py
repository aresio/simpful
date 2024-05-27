import openai
import pandas as pd
import argparse
import re

# Function to construct the prompt for GPT-3.5-turbo-instruct-0914
def construct_prompt(column_name, stats):
    prompt = f"""
You are an expert in fuzzy logic systems. Based on the following statistics for the column '{column_name}':
Min: {stats['min']}
Max: {stats['max']}
Mean: {stats['mean']}
Std: {stats['std']}

Suggest appropriate fuzzy terms for this column in the format of a Python list, adhering to a Likert scale. Use terms like "VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH". If the data suggests a constant or negligible change, suggest "CONSTANT". Consider using 1, 3, 4, or 5 terms based on the data distribution. Ensure the terms are ordered from smallest to largest.
"""
    return prompt

# Function to extract the list of terms from GPT-3.5-turbo-instruct-0914 response
def extract_terms(response_text):
    try:
        # Find the first Python list in the response
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            terms_str = match.group(0)
            terms = eval(terms_str)
            return [term.upper() for term in terms]  # Ensure all terms are in uppercase
    except Exception as e:
        print(f"Error extracting terms: {e}")
    return []

# Function to query GPT-3.5-turbo-instruct-0914 for terms
def query_gpt_for_terms(column_name, stats, verbose=True):
    prompt = construct_prompt(column_name, stats)
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct-0914",
        prompt=prompt,
        max_tokens=200,
        temperature=0.5
    )

    response_text = response.choices[0].text.strip()
    if verbose:
        print(f"GPT-3.5-turbo-instruct-0914 suggested terms for '{column_name}': {response_text}")
    
    terms = extract_terms(response_text)
    return terms

# Function to analyze the data and generate fuzzy terms based on statistics
def analyze_data_and_generate_terms(data, verbose=True):
    skip_columns = {'year', 'month', 'day', 'hour'}
    terms_dict = {}
    
    term_order = {"CONSTANT": 0, "VERY_LOW": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "VERY_HIGH": 5}
    
    for column in data.columns:
        if column.lower() in skip_columns:
            continue

        if data[column].dtype in [float, int]:
            stats = {
                'min': data[column].min(),
                'max': data[column].max(),
                'mean': data[column].mean(),
                'std': data[column].std()
            }
            terms = query_gpt_for_terms(column, stats, verbose)
            
            # Ensure Likert scale consistency and remove redundant "CONSTANT" terms
            if "CONSTANT" in terms and len(set(terms)) == 1:
                terms = ["CONSTANT"]
            else:
                valid_likert_terms = {"VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"}
                terms = [term for term in terms if term in valid_likert_terms]
                terms = list(dict.fromkeys(terms))  # Remove duplicates, preserving order

                if not terms:
                    terms = ["CONSTANT"]
                else:
                    terms.sort(key=lambda term: term_order[term])

            terms_dict[column] = terms

    return terms_dict

# Main function to process dataset and generate terms dictionary
def generate_terms_dict(file_path, api_key, verbose=True):
    openai.api_key = api_key
    data = pd.read_csv(file_path)
    
    terms_dict = analyze_data_and_generate_terms(data, verbose)
    
    # Save the terms dictionary to a file
    with open("terms_dict.py", "w") as f:
        f.write("terms_dict = {\n")
        for column, terms in terms_dict.items():
            f.write(f"    '{column}': {terms},\n")
        f.write("}\n")
    
    return terms_dict

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Generate terms dictionary for a dataset using GPT-3.5-turbo-instruct-0914 based on data analysis.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing the dataset.")
    parser.add_argument("api_key", type=str, help="OpenAI API key.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    
    args = parser.parse_args()
    
    terms_dict = generate_terms_dict(args.file_path, args.api_key, args.verbose)

    if args.verbose:
        print("Generated terms dictionary:")
        for column, terms in terms_dict.items():
            print(f"'{column}': {terms}")

if __name__ == "__main__":
    main()

# Usage
# python generate_terms_dict.py tests/gp_data_x_train.csv sk-proj-token -v
