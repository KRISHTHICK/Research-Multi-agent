#hi 3

# pip install transformers gradio pandas
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline, T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline, T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import os

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
    healthcare_relevant_text = " ".join([sentence for sentence in abstract.split('.') if any(keyword in sentence.lower() for keyword in healthcare_keywords)])
    if healthcare_relevant_text:
        healthcare_summary = extractive_summary(healthcare_relevant_text)
        return healthcare_summary
    else:
        return "Not related to Healthcare"

# AI Agent to enhance AI-related content (focusing on algorithms and machine learning)
def ai_agent(abstract):
    ai_relevant_text = " ".join([sentence for sentence in abstract.split('.') if any(keyword in sentence.lower() for keyword in ai_keywords)])
    if ai_relevant_text:
        ai_summary = extractive_summary(ai_relevant_text)
        return ai_summary
    else:
        return "Not related to AI"

# Function to generate general summary using T5 model for rephrasing
def generate_general_summary(abstract):
    input_text = f"summarize: {abstract}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to generate collaborative summary when both Healthcare and AI are involved
def generate_collaborative_summary(abstract, title, domain):
    general_summary = generate_general_summary(abstract)
    healthcare_summary = healthcare_agent(abstract)
    ai_summary = ai_agent(abstract)

    if domain == "Healthcare, AI":
        collaborative_summary = f"Collaborative Insights between Healthcare and AI: {healthcare_summary} {ai_summary}"
    else:
        collaborative_summary = "Not related to both Healthcare and AI"

    if domain == "General":
        healthcare_summary = "Not related to Healthcare"
        ai_summary = "Not related to AI"

    return general_summary, healthcare_summary, ai_summary, collaborative_summary

# Function to process a CSV file and return results for download
def process_csv(input_file):
    # Load CSV file
    df = pd.read_csv(input_file, encoding='latin-1')

    # Ensure the required columns are present
    if 'Title' not in df.columns or 'Abstract' not in df.columns:
        return "CSV file must contain 'Title' and 'Abstract' columns."

    # Prepare a list to store results
    results = []

    # Process each row in the CSV file
    for index, row in df.iterrows():
        title = row['Title']
        abstract = row['Abstract']

        # Classify the domain (Healthcare, AI, or both)
        domain = classify_domain(title, abstract)

        # Generate summaries
        general_summary, healthcare_summary, ai_summary, collaborative_summary = generate_collaborative_summary(abstract, title, domain)

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

    # Create a DataFrame from the results
    result_df = pd.DataFrame(results)

    # Save to CSV and return file path for download
    output_file = "summarized_results.csv"
    result_df.to_csv(output_file, index=False)
    return output_file

# Function to generate summary for a single abstract
def generate_single_summary(title, abstract):
    domain = classify_domain(title, abstract)
    general_summary, healthcare_summary, ai_summary, collaborative_summary = generate_collaborative_summary(abstract, title, domain)
    return general_summary, healthcare_summary, ai_summary, collaborative_summary

# Gradio Interface functions and file handling
def create_gradio_interface():
    # Create Gradio Interface for a single abstract
    input_abstract = gr.Textbox(label="Abstract", lines=5, placeholder="Enter the abstract of the research paper here...")  
    input_title = gr.Textbox(label="Title", placeholder="Enter the title of the research paper here...")

    # Outputs: General Summary, Healthcare Summary, AI Summary, Collaborative Summary
    output_general_summary = gr.Textbox(label="General Summary")
    output_healthcare_summary = gr.Textbox(label="HealthCare Summary")
    output_ai_summary = gr.Textbox(label="AI Summary")
    output_collaborative_summary = gr.Textbox(label="Collaborative Summary")

    # Create a Gradio interface for single abstract using gr.Row for horizontal layout
    iface_single = gr.Interface(
        fn=generate_single_summary,
        inputs=[input_title, input_abstract],  # Pass inputs as a list 
        outputs=[output_general_summary, output_healthcare_summary, output_ai_summary, output_collaborative_summary],  # Pass outputs as a list
        live=True,
        title="Interdisciplinary Research Collaboration System",  # Title of the app
        description="Surat Stormers: A tool for generating research summaries and collaborative insights between healthcare and AI"  # Subtitle/description
    )

    # Gradio Interface for CSV file upload
    def file_input_function(file):
        return process_csv(file.name)  # Get the processed CSV and its path

    file_input = gr.File(label="Upload CSV with Title and Abstract")
    output_file = gr.File(label="Download Summarized CSV")

    # Interface to return the processed CSV file
    iface_csv = gr.Interface(
        fn=file_input_function,
        inputs=file_input,
        outputs=output_file,
        live=False  # Disable live updating
    )

    return iface_single, iface_csv

# Running Gradio app
iface_single, iface_csv = create_gradio_interface()

# Launch the individual and file processing interfaces
iface_single.launch(share=True)
iface_csv.launch(share=True)
