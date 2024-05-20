from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Set OpenAI API key (you can use .env file)
openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxx"

model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone index (pass your credentials)
pinecone.init(api_key='xxxxxxxxxxxxxxxxxxx', environment='xxxxxxxxxxxxxxx')
index = pinecone.Index('xxxxxxxxxxxxx')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

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

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

def transliterate_tamil_to_english(text):
    return transliterate(text, sanscript.TAMIL, sanscript.ITRANS)

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']



# Streamlit app
def main():
    st.title("LangChain Chatbot")

    user_input_tamil = st.text_input("User Input (Tamil):", "")

    user_input_english = transliterate_tamil_to_english(user_input_tamil)
    refined_query = query_refiner(get_conversation_string(), user_input_english)

    match_result = find_match(refined_query)

    # Display the results
    st.text("User Input (Tamil): " + user_input_tamil)
    st.text("User Input (English): " + user_input_english)
    st.text("Refined Query: " + refined_query)
    st.text("Top Matches:")
    st.text(match_result)
    st.title("LangChain Chatbot")

 
    user_input = st.text_input("User Input:", "")
    refined_query = query_refiner(get_conversation_string(), user_input)

    # Find the most relevant match using Sentence Transformers and Pinecone
    match_result = find_match(refined_query)

    st.text("Refined Query: " + refined_query)
    st.text("Top Matches:")
    st.text(match_result)

if __name__ == "__main__":
    main()
