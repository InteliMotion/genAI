from langchain.chains import ConversationChain, LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from huggingface_hub import login
import fitz
from langchain_anthropic import Anthropic
import re


llm = Anthropic(
    model="claude-3.5-20240229",
    temperature=0,
    max_tokens=8000,
    api_key = "api_key"
)

# Step 3: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    """
    pdf_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                pdf_text += page.get_text()  # Extract text from the page
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return pdf_text

# Step 4: Create Embeddings and Build a Vector Store
def create_embeddings(pdf_text):
    """
    Create embeddings for the extracted PDF text and store them in FAISS for retrieval.
    """
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Split text into chunks for better embedding
    chunk_size = 512
    pdf_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]

    # Create a FAISS vector store
    vector_store = FAISS.from_texts(pdf_chunks, embedding=embeddings)
    return vector_store

# Step 5: Retrieve Relevant Context
def retrieve_context(vector_store, query):
    """
    Retrieve the most relevant context from the vector store for the given query.
    """
    results = vector_store.similarity_search(query, k=3)
    return " ".join([result.page_content for result in results])

# Step 6: Generate Verilog/VHDL Code
import time

# Step 6: Generate Verilog/VHDL Code
def generate_code_with_llm(user_query, context):
    """
    Generates Verilog/VHDL code using LangChain and ChatGroq based on the user query and retrieved context.
    """
    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=(
            """ You are an assistant generating Verilog/VHDL code based on user input and extracted context.
            - Always take care of user-provided details: pin range/size, input/output/clock/enable/reset names.
            - Context can also contain part of prompts used, alayse the instructions is there are any.
            - Carefully read the context before responding.
            - In case of mux without pin range:
                    - Total number of pins is equal to <mux_type>, starting from 0 to (mix_type - 1).
            - In case of mux with pin range:
                    - Total number of pins is equal to <mux_type>, starting from 0 to (mix_type - 1).
                    - Suppose, pin_<n-1>_range=[x:y]
                        pin_<n>_range=[(x+<addition_factor>):(y+<addition_factor>)]   # <n> is pin number
                - Add <addition_factor> correctly and take care of mathematical summation for each pin.
            - Recheck all mathematical calculations and values.
            - Always respond with complete code. do not skip any lines.
            - Respond only with complete Verilog/VHDL code.
            Context: {context}
            Query: {query}
            """
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    retries = 3
    delay = 5  # Initial delay in seconds

    for attempt in range(retries):
        try:
            # Use invoke for better input handling
            response = chain.invoke({"context": context, "query": user_query})
            return response
        except Exception as e:
            if "503" in str(e) or "Service Unavailable" in str(e):
                print(f"Attempt {attempt + 1}/{retries}: Service unavailable. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Unexpected error: {e}")
                break

    # If all retries fail
    return "Error: The service is currently unavailable. Please try again later."


# Step 7: Interactive Interface
def interact_with_user(pdf_path):
    """
    Interactive interface for users to input their queries and generate Verilog/VHDL code.
    """
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Create embeddings and build the vector store
    vector_store = create_embeddings(pdf_text)


    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == "quit":
            break

        # Retrieve relevant context
        context = retrieve_context(vector_store, user_query)

        # Generate code
        response = generate_code_with_llm(user_query, context)
        print("AI :")
        print(response['text'])

# Step 8: Run the Program
if __name__ == "__main__":
    # Replace with the path to your PDF file
    pdf_path = "_path_to_pdf"  # Change to your uploaded file path
    interact_with_user(pdf_path)


