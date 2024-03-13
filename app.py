# Import statements
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

# Streamlit setup
st.subheader("Legal Guardian")

# Session state initialization
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Initialize ChatOpenAI and conversation
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

system_msg_template = SystemMessagePromptTemplate.from_template("""
    Legal Guardian' is a GPT designed to assist with a broad range of legal questions related to children's issues, focusing on laws in India...
    ...It asks for clarification on vague questions to ensure accurate and relevant responses, and treats each query independently for focused assistance.'
""")

human_msg_template = HumanMessagePromptTemplate.from_template("{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Streamlit UI components
response_container = st.container()
textcontainer = st.container()

# Handle user input and display conversation
with textcontainer:
    query = st.text_input("Query: ", key="input")
    if st.button("Submit"):
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Display conversation history
with response_container:
    if st.session_state['responses']:
        st.subheader("Chat History:")
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
