'''LangChain is a popular Python framework designed to help developers build applications 
that use large language models (LLMs) more effectively.'''

'''LangChain Groq refers to an integration that allows LangChain to use the Groq model, 
a high-performance, specialized processor designed for AI workloads,'''

from langchain_groq import ChatGroq

API_KEY = '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'

LLM = ChatGroq(
    model ='llama-3.1-8b-instant',
    groq_api_key = API_KEY,
    temperature = 0,     #This temperature parameter controls the randomness of generated output - 0 means more deterministic ouput, closer to 1 means more random, diverse and creative output
    max_tokens = 250,    #This will limit the generated output to 250 words.
    # timeout = None

)    #This object can directly take input and using API of groqcloud, process the input and return the output
    # You can try changing various inputs to get different outputs just like a any LLM model.

input = 'Generate an expanation on transformers in LLM'

response = LLM.invoke(input)

print(response.content)