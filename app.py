import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH="chroma_subtitle_data/"  #path where the chromadb data is present
EMBED_MODEL="all-MiniLM-L6-v2"  #bert based embedding model
COLLECTION_NAME="subtitle_movie_data" #collection name"


client=chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
col = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)



st.title("⌨️Subtitle Search Bar")
bg="""
<style>
[data-testid="stAppViewContainer"]{
  background-color:	#8C90C8;
}
</style>
"""
st.markdown(bg,unsafe_allow_html=True)

label = "Enter the subtitle"
query = st.text_input(label)
if st.button("Find the movie/series name"):
    query_results = col.query(query_texts=[query], n_results=5)
    st.write("Top 5 similar")
    st.write(query_results["metadatas"])
  