from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# **Step 1: Load the PDF File from Data Path****
loader = DirectoryLoader('data/',
                         glob="*j.pdf",
                         loader_cls=PyPDFLoader)

documents = loader.load()

# print(documents)

# ***Step 2: Split Text into Chunks***

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50)

text_chunks = text_splitter.split_documents(documents)

print(len(text_chunks))
# **Step 3: Load the Embedding Model***


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

# **Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
vector_store = FAISS.from_documents(text_chunks, embeddings)

# **Step 5: Find the Top 3 Answers for the Query***

query = "Description in project Asyncio Restaurant Simulation"
docs = vector_store.similarity_search(query)

# print(docs)
llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 128,
                            'temperature': 0.01})

template = """Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

# start=timeit.default_timer()

chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': qa_prompt})
start = time.time()
response = chain.invoke({'query': query})
print(f"Here is the complete Response: {response}")
print(str(time.time() - start))

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                           retriever=vector_store.as_retriever(), memory=memory)
start = time.time()
response = conversation_chain.invoke({'question': query})
# end=timeit.default_timer()
print(f"Here is the complete Response: {response}")
print(str(time.time() - start))

# print(f"Here is the final answer: {response['result']}")

# print(f"Time to generate response: {end-start}")

"""
while True:
    user_input = input(f"prompt:")
    if user_input == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = chain.invoke({'query': user_input})
    print(f"Answer:{result['result']}")
"""
"""
llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 128,
                            'temperature': 0.01})


answer = llm.invoke("translate English to German: How old are you?")
print(answer)
what is a mirrored-strategy?
"""
