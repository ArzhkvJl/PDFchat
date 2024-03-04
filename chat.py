from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.llms import CTransformers
import customtkinter as ctk
from customtkinter import filedialog
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_file_name():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", ".pdf")])
    entry.insert(0, file_path)


def get_pdf_text():
    text = ""
    pdf_reader = PdfReader(entry.get())
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_conversation_chain(vector_store):
    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        config={'max_new_tokens': 128,
                                'temperature': 0.05})

    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vector_store.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def generate():
    # Load the PDF File from Data Path
    text = get_pdf_text()
    # Split Text into Chunks
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=500,
                                          chunk_overlap=50, length_function=len)
    text_chunks = text_splitter.split_text(text)
    # Load the Embedding Model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
    vector_store = FAISS.from_texts(text_chunks, embeddings)

    # Find the Top 3 Answers for the Query
    query = question_entry.get()  # user question
    docs = vector_store.similarity_search(query)

    template = """Use the following pieces of information to answer the user's question.
    If you dont know the answer just say you know, don't try to make up an answer.

    Context:{context}
    Question:{question}

    Only return the helpful answer below and nothing else
    Helpful answer
    """

    qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    # Create conversation chain
    conversation = get_conversation_chain(vector_store)
    answer = conversation.invoke({'question': query})
    chat_history = answer['chat_history']
    result.insert("0.0", answer)
    print("Done!")


root = ctk.CTk()
root.geometry("750x550")
ctk.set_appearance_mode("light")
root.title("Llama2 Answers Geterator")
title_label = ctk.CTkLabel(root,
                           text="Answers about PDFs",
                           font=ctk.CTkFont(size=30, family="Bai Jamjuree"))
title_label.pack(padx=10, pady=(40, 20))
frame = ctk.CTkFrame(root)
frame.pack(fill="x", padx=100)
# First frame with PDF loader
pdf_path_frame = ctk.CTkFrame(frame)
pdf_path_frame.pack(fill="both", padx=100, pady=(20, 5))
pdf_path_label = ctk.CTkLabel(pdf_path_frame,
                              text="PDF file path",
                              font=ctk.CTkFont(size=20, family="Bai Jamjuree"))
pdf_path_label.pack()
# Create a button to open the file dialog box
button = ctk.CTkButton(pdf_path_frame, text="Browse", command=lambda: get_file_name())
button.pack(pady=(10, 10))
entry = ctk.CTkEntry(pdf_path_frame, width=500)
entry.pack(fill="both", padx=10, pady=(10, 10))
# Second frame with the question
question_frame = ctk.CTkFrame(frame)
question_frame.pack(fill="both", padx=100, pady=(20, 5))
question_label = ctk.CTkLabel(question_frame,
                              text="Ask a question",
                              font=ctk.CTkFont(size=20, family="Bai Jamjuree"))
question_label.pack()
question_entry = ctk.CTkEntry(question_frame, width=500)
question_entry.pack(fill="both", padx=10, pady=(10, 10))
# Create a button to start text generation
start_button = ctk.CTkButton(frame, text="Generate an answer",
                             command=generate)
start_button.pack(padx=100, fill="x", pady=(5, 20))
result = ctk.CTkTextbox(root, font=ctk.CTkFont(size=15))
result.pack(pady=10, fill="x", padx=100)

root.mainloop()
