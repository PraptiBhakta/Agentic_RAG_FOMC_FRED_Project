# %%
## import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from typing import List
import contextlib
import io, re, shutil, os, time, json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, models
import openai
from openai import OpenAI
import anthropic
import matplotlib.dates as mdates

#%%
## Load FRED master file and pre-processed FOMC chunk master file.
fomc_chunk_new_df=pd.read_pickle("fomc_chunk_new_df.pkl")

# %%
## To get OpenAI and Anthropic api key
with open("OpenAI_API_Key.txt", "r") as f:
  openai_api_key = ' '.join(f.readlines())

#%%
def fomc_collection_load():
    # Path of stored collection database.
    coll_path = os.path.join(os.getcwd(), "FOMC_ChromaDB_Data_v2")
    collection_name = "FOMC_Chroma_Client1"
    model_name = "mukaj/fin-mpnet-base"

    embedd_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="mukaj/fin-mpnet-base")

    client = chromadb.PersistentClient(path=coll_path)

    chroma_collection = client.get_collection(name=collection_name,embedding_function=embedd_function)

    return chroma_collection
#%%
## Function to return date range of given user query

def extract_years(user_query):
    years = re.findall(r"\b(?:19|20)\d{2}\b", user_query)
    yrs=[int(y) for y in years]

    if len(yrs)==1:
        start_dt=f'{yrs[0]}-01-01'
        end_dt=f'{yrs[0]}-12-31'

    elif len(yrs)>=2:
        start_dt=f'{yrs[0]}-01-01'
        end_dt=f'{yrs[1]}-12-31'

    else:
        start_dt=str('2021-01-01')
        end_dt=str('2025-12-31')
    
    return start_dt,end_dt

#%%

## Defining Function for query search in chroma vector DB.

def query_engine(user_query,chroma_collection):

    ## Obtain date range from query for optimal result
    start_dt,end_dt=extract_years(user_query)
    start_year = pd.to_datetime(start_dt).year
    end_year = pd.to_datetime(end_dt).year

    ## Call collection database for retrieval
    if start_year is not None and end_year is not None:
      ## Perform query on choromadb based on user query and retrieve top 30 relevant search from vector db based on date range.
      query_results = chroma_collection.query(query_texts=user_query,n_results=30,\
                                              where={"$and": [{"meeting_year": {"$gte": start_year}},{"meeting_year": {"$lte": end_year}}]},\
                                              include=["documents", "metadatas", "distances"])
    else:
      query_results = chroma_collection.query(query_texts=[user_query],n_results=30,\
                                              where={"meeting_year": {"$gte": 2021}},\
                                              include=["documents", "metadatas", "distances"])
      
  ## Creating Dataframe of query results.
    query_results_df=pd.DataFrame(columns=['Document','Metadatas','Distance'])
    query_results_df['Document']=query_results['documents'][0]
    query_results_df['Metadatas']=query_results['metadatas'][0]
    query_results_df['Distance']=query_results['distances'][0]
    query_results_df['Document_ID']=[i['doc_id'] for i in query_results_df['Metadatas']]
    query_results_df['Chunk_ID']=[i['chunk_id'] for i in query_results_df['Metadatas']]
    query_results_df['Meeting_Date']=[i['meeting_date'] for i in query_results_df['Metadatas']]
     
    top_15_similar_document= query_results_df.sort_values(by='Distance',ascending=True).head(15)
    return(top_15_similar_document)
#%%

#### Retrieval function to retrieve relevant documents to user query.

def user_query_retreival(user_query):

  text_question_answer_for_llm={}

  ## Call collection load function for semantic retrieval of text.
  chroma_collection=fomc_collection_load()

  ## Call query engine function to retrieve top 15 semantic chunks from embed collection.
  top_15_similar_document=query_engine(user_query,chroma_collection)
  document_retreived_for_query = top_15_similar_document.sort_values(by='Distance')
  text_question_answer_for_llm['user_query']=user_query

  metadatas_list = document_retreived_for_query['Metadatas'].tolist()
  documents_list = document_retreived_for_query['Document'].tolist()

  combined_retrieval_list = []

  for meta, doc in zip(metadatas_list, documents_list):
    combined_retrieval_list.append({**meta,"document_text": doc})

  text_question_answer_for_llm['documents_retrieved_for_question_answer'] = combined_retrieval_list
  return text_question_answer_for_llm
#%%

# Function to generate response of user query using llm prompt.
def textual_query_response_generation_llm(output_json,openai_api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic and policy analysis of FOMC textual data to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC minutes and statements document textual query.
  You will be provided list of relevant chunks of FOMC documents including both statement and minute type for a specific period.
  Your task is to generate answer to original user query related to FOMC communication documents from retrieved Federal Open Market Committee (FOMC) documents provided in <<documents_retrieved_for_question_answer>>.

  <<documents_retrieved_for_question_answer>> from <<output_json>>

  You must follow below instructions for generating the summary response:
  1. Read and understand all the relevant documents excerpts from <<documents_retrieved_for_question_answer>>.
  2. Read and understand the context of user original query from <<user_query>> in <<documents_retrieved_for_question_answer>>.
  3. If any overlapping insights or ideas across multiple documents then combine the information to generate answer to user original query.
  4. Use only information provided in the retrieved documents and do not add any information of your own. The explanation in your answer must be simple, clear, concise, and factually grounded.
  5. You must not assume facts that are not present in the context and clearly mention in response if the answer generated from context document is partial.
  6. Whenever possible, mention the economic condition and policy stance.
  7. Write a direct answer to the user original question in 120–180 words and do not exceed more than 200 words. If required, you can use bullet points to explain the supporting information from documents.
  8. You must write as a professional and economic analyst.
  9. You must use '%' when refering to percentages and instead of writing the word 'percent' provide '%'.
      Example:3% instead of 3 percent.
  10. **VERY IMPORTANT** , if there is year or month or date provided in user query, then retrieve ONLY that specific year document chunks. IF there are no documents on those date, then generate response based on retrieved documents.
  10. At the end of response, you must provide citation of the documents that you used for generating answer to user original query.

  Note that user query type and relevant documents details are provided in {output_json} and you must return a response.

  Citation Evidence:

  Mention the meeting dates referenced in the retrieved documents in format yyyy-mm-dd along with the source type as Minutes (M) or Statement(S) based on value provided in 'doc_type' parameter of <<documents_retrieved_for_summary>>.
  You must also provide citation link for each document.
  You must ensure that for FOMC statement documents, the url link is 'https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm'
  and for FOMC minute document url link is 'https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm'
  You must use doicument type value based on 'doc_type' patrameter of <<documents_retrieved_for_question_answer>>.
  List all document citations using the STRICT format below:
  Document Date: YYYY-MM-DD, Source: Minute/Statement,  https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm or https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm based on document type.
  Document Date: YYYY-MM-DD, Source: Minute/Statement, https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm or https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm based on document type.

  Replace YYYYMMDD in the url link  with values present in document timestamp of document in <<documents_retrieved_for_question_answer>>.

  """
  # Initialize client with API key
  client = OpenAI(api_key=openai_api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content
   
#%%

## User Query response generator function
def chat_query_response(user_query):
    ## Load personal key files from the local path
    with open("OpenAI_API_Key.txt", "r") as f:
        openai_api_key = ' '.join(f.readlines())
    # To calculate total time taken
    strt=time.time()
  
    # Initializing return variables
    final_response = None
    fig = None
    fig1 = None
    fig2 = None
    retrieve_generate_time=0
    text_data=user_query_retreival(user_query)
    output_json=json.dumps(text_data, indent=1)
    final_response=textual_query_response_generation_llm(output_json,openai_api_key)

    end=time.time()
    retrieve_generate_time=round(end-strt,4)

    ## Return variables to Streamlit.
    return {
    "response": final_response,
    "chart": fig,
    "chart1": fig1,
    "chart2": fig2,
    "time_taken": retrieve_generate_time        
    }
#%%
