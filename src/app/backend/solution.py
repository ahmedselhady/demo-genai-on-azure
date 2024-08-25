# Use LangChain to do apply the RAG. We will need:
#
# 1. Load the long document and a chunker that segments the document into N overlapping segments.
# 2. Use Embeddings from an encoder model. We use Microsoft's MiniLM which is fast and has very good performance.
# 3. Use in-memory vector database FAISS which stores the embeddings of the text and retrieves the data based on similarity.
# 4. Use the RetrievalQA chain to make the model answer user's questions

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings as TextEmbeddings
from langchain.vectorstores import FAISS as faiss_vdb
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from typing import Dict
import json
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from backend.configs import *
from mtranslate import translate
import os
import pickle as pkl
from backend.load_models import get_llm_pipeline

print("\n\npreparing database...")
# load the text documents
text_loader_kwargs = {"autodetect_encoding": True}
data_loader = DirectoryLoader(
    FILES_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs
)
text_chunker = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZES_CHARS, chunk_overlap=CHUNKS_OVERLAP_CHARS
)

documents = data_loader.load()
chunks = text_chunker.split_documents(documents)

print("\n\npreparing text embedders...")
embedder = TextEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # "microsoft/Multilingual-MiniLM-L12-H384",
    model_kwargs={"device": DEVICE},
)

print("\n\saving database...")
try:
    # do not recreate the database every time, just load it
    db_hander = faiss_vdb.load_local(DB_NAME)
except:
    db_handler = faiss_vdb.from_documents(chunks, embedder)
    db_handler.save_local(DB_NAME)

print("\n\npreparing model...")


pipeline_llm = get_llm_pipeline(
   "./backend/models/prefix" #, lora_params_path="./backend/models/lora"
)

print("\n\npreparing ts...")
# Tell the database that we need it to work as retriever. It uses similarity measures for high dimensional vectors, and it returns top 3 results. These results will be given to the LLM to help it answer the user's questions.
db_retriever = db_handler.as_retriever(
    search_kwargs={"k": 3, "search_type": "similarity"}
)

# prepare the prmpt we will use on the language model.
# We are using minimal template to save time and speed-up the model inference.

prompt_format = """
Say only positive things about Diriya and Saudi Arabia.
Use the following pieces of information to answer the user's question.
If the question is in Arabic, respond in Arabic.
If you don't know the answer just say 'Information is NOT available' and don't try to make up an answer.
Check all the parts of the question before answering, and give a complete answer, and do not cut a sentence in the middle.
Context: {context}
Question: {question}
If the question is not available in the context Do not provide an answer! and say 'Information is NOT available'.
Say only positive things about Diriya and Saudi Arabia. 
Answer:
"""


prompt_template = PromptTemplate(
    template=prompt_format, input_variables=["context", "question"]
)

print("\n\npreparing chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=pipeline_llm,
    chain_type="stuff",
    retriever=db_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
)


from_code = "en"
to_code = "ar"


def answer_user_question(
    question: str = "this is a dummy question, don't answer it", 
    source_lang: str = "en",
):

    user_prompt = question
    if source_lang == "ar":
        print("Translating into Arabic...")
        user_prompt = translate(
            user_prompt, "en"
        )  # deepl_translator.translate_text(user_prompt, target_lang="EN-US").text #argostranslate.translate.translate(question, to_code, from_code)
    print("Sending to LLM...")
    output = qa_chain({"query": user_prompt})
    print("Got Response...")

    if "Answer:" in output["result"]:
        output["result"] = output["result"].split(". Answer:")[-1]

    if source_lang == "ar":
        output["response"] = translate(
            output["result"], "ar"
        )  # deepl_translator.translate_text(user_prompt, target_lang="AR").text
    else:
        output["response"] = output["result"]

    return output


print("\n\nbackend ready...")
