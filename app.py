import os
import warnings
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session , url_for
from langchain_community.llms import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

# Suppress the pydantic warning
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4-turbo"

# Initialize the model and embeddings
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
embeddings = OpenAIEmbeddings()
parser = StrOutputParser()

# Define the prompt template
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "It is not within my Scope".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")
chain = prompt | model | parser

# Load and split the PDF
loader = PyPDFLoader("User_Guide.pdf")
pages = loader.load_and_split()

# Create the vector store from the document
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)

# Function for generating LLM response
def format_answer(answer):
    steps = answer.split('. ')
    formatted_answer = '<ol>'
    for step in steps:
        if step:
            formatted_answer += f'<li>{step.strip()}</li>'
    formatted_answer += '</ol>'
    return formatted_answer

def generate_response(input):
    result = chain.invoke({'question': input})
    answer = result.split("Answer: ")[-1].strip() if "Answer: " in result else result.strip()
    return format_answer(answer)

@app.route('/')
def index():
    if 'messages' not in session:
        session['messages'] = [{"role": "assistant", "content": "Welcome to Infyz Solutions"}]
    return render_template('index.html', messages=session['messages'])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    session['messages'].append({"role": "", "content": user_input})
    
    response = generate_response(user_input)
    
    session['messages'].append({"role": "", "content": response})
    
    return jsonify({"message": response})



if __name__ == '__main__':
    app.run(debug=True)
