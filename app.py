import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

OPENAI_API_KEY = "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"

#UI

st.header("Inteligent chatbot with Generative AI")
with st.sidebar : 
    st.title("your docs")
    file = st.file_uploader("Download a PDF file",type="pdf")

if file is not None :
    #Text extraction from the PDF
    st.write("File downloaded succesfully")
    st.write(f"file's name : {file.name}")
else:
    st.write("choose a PDF file to start")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""

    #Manage the case where the extraction is not possible (images/graphs...)


    for page in pdf_reader.pages:
        page_text = page.extract_text() 
        if page_text:
            text += page_text + "\n"
        else:
            st.warning("one or more page not contain an exploitable text in your file")


    #Splitting the text to Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", "!", "?", " ", ""], #to not cut the sentence in the middle
        chunk_size = 1000, #num of characters in each chunks 
        chunk_overlap = 100, #to not lose informations that include in the previous/next chunk but important to understand the current chunk
        length_function=len #unit of chunk_size, here is character
    )
    chunks = text_splitter.split_text(text)
    
 #Generate the embedding

    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    #DB creation (Faiss)
    vector_bd = FAISS.from_texts(chunks,embeddings) #generate the embedding of each chuck an store it in the db 

    #Question input 

    user_query = st.text_input("Ask your question Here")

    #Find chunks similirity with the user's question

    if user_query:
        results = vector_bd.similarity_search(user_query,top_k = 2)

        st.write("Chunks similarity founds")

        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature=0, #best accuracy
            max_tokens=100, #answer of 20 words
            model_name="gpt-4"
        )

        #load question-answers chain
        chain = load_qa_chain(llm, chain_type="stuff") #we concat the similar chinks before send to the llm model

        response = chain.run(input_documents = results, question = user_query)

        #Display answer

        st.write("Generated answer")
        st.write(response)

        #Download teh answer 
        st.download_button(
            label="Downloas answer",
            data=response,
            file_name="answer_chatbot.txt",
            mime="text/plain"
        )
  