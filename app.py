# importing libraries
# importing mlflow library
import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import json
import os
from include.env_values import *
from include.utils import *
import openai

# setting up environment
os.environ["OPENAI_API_KEY"] = openai_api_key
with open("./include/config.json") as file:
    config = json.load(file)

### Defining MLflow wrapper for model
class MLflowADEClassifier_FT(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name):
        self.model = model_name
    def predict(self, prompt):
        res = openai.Completion.create(model=self.model, prompt=prompt, max_tokens=1, temperature=0)
        return res['choices'][0]['text']

st.title("💊 Adverse Drug Affect Classifier")

selected_model = st.selectbox('Select Model....', ('Few-Shot Encoding', 'Fine-Tuned'))

# importing data for demonstration
data = pd.read_json(f"{config['base_path']}/data/curie_fine_tuning_data/train_data_prepared_valid.jsonl", lines=True)

prompt = st.selectbox("Select an example prompt.", (f"(is_ADE) {data['prompt'][0]}", f"(not_ADE) {data['prompt'][1]}"))

if prompt:
    if selected_model == 'Fine-Tuned':
        # retrieveing model
        model = MLflowADEClassifier_FT(config['fine_tuned_model'])
        if prompt[0:4] =='(is':
            output = model.predict(prompt=data['prompt'][0])
        else:
            output = model.predict(prompt=data['prompt'][1])
        if output.find('not') == 1:
            st.write('not_ADE')
        else:
            st.write('is_ADE')
    else:
        utils = Utils()
        retriever=utils.get_retriever()
        # retrieve model from mlflow
        model = mlflow.pyfunc.load_model(f"models:/ade_llm_classifier/Production")
        if prompt[0:4] =='(is':
            dct = {k:[v] for k,v in {"statement": f"{data['prompt'][0]}"}.items()}
            statement = pd.DataFrame(dct)
            docs = retriever.get_relevant_documents(data['prompt'][0])
        else:
            dct = {k:[v] for k,v in {"statement": f"{data['prompt'][1]}"}.items()}
            statement = pd.DataFrame(dct)
            docs = retriever.get_relevant_documents(data['prompt'][1])
        # get response
        output = model.predict(statement)

        st.write(output)
        
        
        context = ""
        for doc in docs:
            # get document text
            context = context + "\n" + doc.page_content + "\n" + "###"

        input_prompt = f"""
        understand the statements for any adverse events and predict the [nature]. 'is_ADE' means [statement] reports an adverse event medically and 'not_ADE' means not adverse event.

        {context} \n
        [statement]: {prompt} \n
        [nature]: ''\n

        '###' means end of line.

        return output for last 'statement' in this way:
        [nature]: 'is_ADE' (if adverse avtivity present)
        [nature]: 'not_ADE' (if adverset avtivity not present)
        [nature]: 'I can't Identify'
        """
        with st.expander('Generated Input Prompt'):
            st.code(input_prompt) 
        
    