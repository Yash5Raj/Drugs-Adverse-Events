# importing libraries
import os
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

class Utils():
    def __init__(self):
        self.embedding_model_path = embedding_model_path = "C:\\Users\\yraj\\Work\\POCs\\Drugs & Adverse Events\\embedding_model"
        self.model_name = 'gpt-3.5-turbo'
        self.vector_store_path = 'C:\\Users\\yraj\\Work\\POCs\\Drugs & Adverse Events\\data\\test_vector_store'
        self.system_message_template = """
        You are a helpful assistant built by Yash, you are good at helping classification of drug and it's affect. 
        """
        self.human_message_template = """
        '/n' means next-line thourghout the prompt

        understand the statements for any adverse events and predict the [nature]. 'is_ADE' means [statement] reports an adverse event medically and 'not_ADE' means not adverse event.

        {context} 
        [statement]: {statement} 
        [nature]: ''

        '###' means end of line.

        return output for last 'statement' in this way:
        [nature]: 'is_ADE' (if adverse avtivity present)
        [nature]: 'not_ADE' (if adverset avtivity not present)
        [nature]: 'I can't Identify'
        """
    def get_retriever(self, n_documents=5):
        # encoder path

        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_path)

        # load vector data if not defined already
        vector_store = FAISS.load_local(embeddings=embedding_model, folder_path=self.vector_store_path)
        # configure document retrieval 
        retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) 
        return retriever

    def get_prompt(self):

        # define system-level instructions
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_message_template)
        # define human-driven instructions
        human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_message_template)
        # combine instructions into a single prompt
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        return chat_prompt

    def get_llm(self):
        # define model to respond to prompt
        llm = ChatOpenAI(model_name=self.model_name, temperature=0.9)
        return llm
