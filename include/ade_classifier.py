# importing libraries
from langchain import LLMChain
class ADE_Classifier():
    def __init__(self, llm, retriever, prompt):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.ade_classifier = LLMChain(llm=self.llm, prompt=prompt)
        
    def _get_class(self, context, statement, timeout_sec=60):
        '''get answer from llm with timeout handling'''
        
        # default result
        result = None
        import time
        import openai
        # define end time
        end_time = time.time() + timeout_sec
        
        # try timeout
        while time.time() < end_time:
            # attempt to get a response
            try:
                result = self.ade_classifier.generate([{'context': context, 'statement':statement}])
                break # if successful response, stop lopping
                
            # if rate limit error...
            except openai.error.RateLimitError as rate_limit_error:
                if time.time() < end_time: # if time permits, sleep
                    time.sleep(2)
                    continue
                else: # otherwise, raise the exception
                    raise rate_limit_error
            
            # if other error, raise it
            except Exception as e:
                print(f'LLM ADE Classifier Chain encountered unexpected error: {e}')
                raise e
        return result
    
    def get_class(self, statement):
        '''get answer to provide question'''
        
        # default result
        result = {'generated_output':None}
        # get relevant documents
#         retriever = vector_store.as_retriever(search_kwargs={'k': n_examples}) # configure retrieval mechanism
        docs = self.retriever.get_relevant_documents(statement)
    
        context = ""
        for doc in docs:
            # get document text
            context = context + "\n" + doc.page_content + "\n" + "###"

        # get an answer from llm
        output = self._get_class(context=context, statement=statement)
        
        # get output from results
        generation = output.generations[0][0]
        answer = generation.text
        
        result['generated_output'] = answer
        
        return result
