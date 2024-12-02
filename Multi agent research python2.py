#hi2

import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline, T5Tokenizer, T5ForConditionalGeneration

# Initialize BART and T5 models and tokenizers for summarization
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Initialize Summarization pipeline for BART
summarizer = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)

# Healthcare and AI keyword lists
healthcare_keywords = ["disease", "cancer", "patient", "treatment", "health", "illness", "medicine", "symptom", "diagnosis", "epidemic", "infection"]
ai_keywords = ["algorithm", "artificial intelligence", "machine learning", "neural network", "AI", "model", "deep learning", "prediction", "data"]

# Function to classify the domain (Healthcare, AI, or both)
def classify_domain(title, abstract):
    healthcare_detected = any(keyword.lower() in (title + abstract).lower() for keyword in healthcare_keywords)
    ai_detected = any(keyword.lower() in (title + abstract).lower() for keyword in ai_keywords)

    if healthcare_detected and ai_detected:
        return "Healthcare, AI"  # Both healthcare and AI
    elif healthcare_detected:
        return "HealthCare"
    elif ai_detected:
        return "AI"
    return "General"

# Function to generate extractive summaries using BART
def extractive_summary(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Healthcare Agent to enhance healthcare-related content (focusing on diseases and treatments)
def healthcare_agent(abstract):
    # Check if healthcare-related keywords are in the abstract
    healthcare_relevant_text = " ".join([sentence for sentence in abstract.split('.') if any(keyword in sentence.lower() for keyword in healthcare_keywords)])

    # If healthcare-related sentences are found, generate a summary
    if healthcare_relevant_text:
        healthcare_summary = extractive_summary(healthcare_relevant_text)
        return healthcare_summary
    else:
        return "Not related to Healthcare"

# AI Agent to enhance AI-related content (focusing on algorithms and machine learning)
def ai_agent(abstract):
    # Check if AI-related keywords are in the abstract
    ai_relevant_text = " ".join([sentence for sentence in abstract.split('.') if any(keyword in sentence.lower() for keyword in ai_keywords)])
    if ai_relevant_text:
        ai_summary = extractive_summary(ai_relevant_text)
        return ai_summary
    else:
        return "Not related to AI"

# Function to generate a paraphrased general summary (with key points from the abstract)
def generate_general_summary(abstract):
    # Use T5 model for paraphrasing the abstract (retaining key points, but rephrased)
    input_text = f"summarize: {abstract}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Ensure that the general summary is clear and paraphrased correctly, retaining the meaning of the original abstract
    return summary

# Function to generate collaborative insights between healthcare and AI
def generate_collaborative_insights(abstract, title, domain):
    # Initialize summary placeholders
    general_summary = generate_general_summary(abstract)
    healthcare_summary = healthcare_agent(abstract)
    ai_summary = ai_agent(abstract)

    # Collaborative summary if both healthcare and AI are involved
    if domain == "Healthcare, AI":
        collaborative_summary = f"Collaborative Insights between Healthcare and AI: {healthcare_summary} {ai_summary}"
    else:
        collaborative_summary = "Not related to both Healthcare and AI"  # Collaborative insight will not be generated if the domain does not match

    # If domain doesn't match healthcare or AI, use general summary and not related for the respective fields
    if domain == "General":
        healthcare_summary = "Not related to Healthcare"
        ai_summary = "Not related to AI"

    return general_summary, healthcare_summary, ai_summary, collaborative_summary

# Function to process a CSV file
def process_csv(input_file, output_file):
    # Load CSV file
    df = pd.read_csv(input_file, encoding='latin-1')

    # Ensure the required columns are present
    if 'Title' not in df.columns or 'Abstract' not in df.columns:
        print("CSV file must contain 'Title' and 'Abstract' columns.")
        return

    # Prepare a list to store results
    results = []

    # Process each row in the CSV file
    for index, row in df.iterrows():
        title = row['Title']
        abstract = row['Abstract']

        # Classify the domain (Healthcare, AI, or both)
        domain = classify_domain(title, abstract)

        # Generate summaries
        general_summary, healthcare_summary, ai_summary, collaborative_summary = generate_collaborative_insights(abstract, title, domain)

        # Store the results
        results.append({
            'Title': title,
            'Abstract': abstract,
            'Domain': domain,
            'General Summary': general_summary,
            'HealthCare Summary': healthcare_summary,
            'AI Summary': ai_summary,
            'Collaborative Summary': collaborative_summary
        })

    # Create a DataFrame from the results and save it as a new CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Example Usage
input_csv = 'paper.csv'  # Path to input CSV file
output_csv = 'output_research_papers_with_summaries.csv'  # Path to output CSV file

# Process the input CSV and generate the results
process_csv(input_csv, output_csv)
