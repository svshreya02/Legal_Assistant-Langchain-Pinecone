from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Set OpenAI API key
openai.api_key = "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"

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone index
pinecone.init(api_key='14b2909a-c00c-4ff8-9b79-87eb51b9d891', environment='gcp-starter')
index = pinecone.Index('langchain-chatbot')

# Function to find the most relevant match in Pinecone index
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

# Function to refine a user query using OpenAI's Completion API
def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

# Function to get the conversation string for display
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

def transliterate_tamil_to_english(text):
    return transliterate(text, sanscript.TAMIL, sanscript.ITRANS)

# Function to find the most relevant match in Pinecone index
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

# ... (your existing functions)

# Streamlit app
def main():
    st.title("LangChain Chatbot")

    # User input for the conversation in Tamil
    user_input_tamil = st.text_input("User Input (Tamil):", "")

    # Transliterate Tamil input to English for processing
    user_input_english = transliterate_tamil_to_english(user_input_tamil)

    # Retrieve refined query using OpenAI
    refined_query = query_refiner(get_conversation_string(), user_input_english)

    # Find the most relevant match using Sentence Transformers and Pinecone
    match_result = find_match(refined_query)

    # Display results
    st.text("User Input (Tamil): " + user_input_tamil)
    st.text("User Input (English): " + user_input_english)
    st.text("Refined Query: " + refined_query)
    st.text("Top Matches:")
    st.text(match_result)
    st.title("LangChain Chatbot")

    # User input for the conversation
    user_input = st.text_input("User Input:", "")

    # Retrieve refined query using OpenAI
    refined_query = query_refiner(get_conversation_string(), user_input)

    # Find the most relevant match using Sentence Transformers and Pinecone
    match_result = find_match(refined_query)

    # Display results
    st.text("Refined Query: " + refined_query)
    st.text("Top Matches:")
    st.text(match_result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
