#%%
## import python code of agentic rag pipeline.

import json
from agentic_rag_pipeline_vscode import query_intent_parser, query_orchestrator_for_retreival
from traditional_rag_pipeline_vsccode import user_query_retreival

#%%
## To perform evaluation on chunks retreieved from FOMC for Precision@k metrics.(applicable only for Question_answer type and topic summary type)

## Define 8 queries for tetsing 

query_lst= ['What concerns has the Fed expressed related to unemployment?',
          "What was the Fed's view of on economic growth in 2025?", 
          'Provide a topic-wise summary of inflation discussions',
          'What are the economic risks highlighted by the Fed',
          'What was the major economic concerns discussed in latest FOMC meetings',
          'Summarize GDP growth',
          'What does FED says about inflation?',
          'What policy changes are expected based on latest meeting discussion?']


#%%

## Retrieval of chunks for all queries from Agentic RAG 
## Retrieval function applicable only for QA and topic_summary type

def retrieval_relevance_evaluation(query_parse, user_query):
    if query_parse['query_type']=='text' and query_parse['query_task_type']=='summary_topic':
      text_data=query_orchestrator_for_retreival(query_parse,user_query)['documents_retrieved_for_summary'][0:5]
      print('User query --- ',user_query)
      print('Below is the JSON output of retrieved chunks:,\n')
      print(json.dumps(text_data, indent=1))

    elif query_parse['query_type']=='text' and query_parse['query_task_type']=='question_answer':
      text_data=query_orchestrator_for_retreival(query_parse,user_query)
      print('User query --- ',user_query)
      print('Below is the JSON output of retrieved chunks:,\n')
      print(json.dumps(text_data, indent=1))

print('Below are the chunks retrieved for all eight queries through Agentic RAG system :\n\n')

for query in query_lst:
    retrieval_relevance_evaluation(query_intent_parser(query),query)

#%%
# Retrieval of chunks for all queries from Traditional RAG 
print('Below are the chunks retrieved for all eight queries through traditional RAG system :\n\n')

for query in query_lst:
    text_data=user_query_retreival(query)['documents_retrieved_for_question_answer'][0:5]
    output_json=json.dumps(text_data, indent=1)
    print('\n\nRetrieved chunks/data for query : ',query)
    print(output_json)

#%%
## Evaluation of reasoning agent based on JSON output or query intent/parser prompt
## This is applicable only for agentic RAG system

# Define new query list to include numeric queries that require retrieval if data from FRED database.

query_lst= [
           'Compare GDP rate before and after Covid-19',
           "What was the 75th percentile of employment rate in 2025?", 
           'Provide a topic-wise summary of inflation discussions',
           'Summarize FOMC statement in 2024',
           'What was the major economic concerns discussed in latest FOMC meetings',
          'Summarize GDP growth',
           'Compare Fed funds rate and inflation rate',
           'Provide a summary of most recent FOMC meeting conducted in 2025'
          ]

## Retrieval function to get the retreieved data for evaluation of systemn flow

def retrieval_relevance_evaluation1(query_parse, user_query):
    if query_parse['query_type']=='text' and query_parse['query_task_type']=='summary_topic':
      text_data=query_orchestrator_for_retreival(query_parse,user_query)['documents_retrieved_for_summary'][0:5]
      print('User query --- ',user_query)
      print('Below is the JSON output of retrieved chunks:,\n')
      print(json.dumps(text_data, indent=1))

    if ((query_parse['query_type']=='text') and (query_parse['query_task_type']=='summary_fomc_minute')) or\
          ((query_parse['query_type']=='text') and (query_parse['query_task_type']=='summary_fomc_statement')):
      text_data=query_orchestrator_for_retreival(query_parse,user_query)['document_retreived_for_sumamry'][0:5]
      print('User query --- ',user_query)
      print('Below is the JSON output of retrieved chunks:,\n')
      print(json.dumps(text_data, indent=1))

    elif query_parse['query_type']=='text' and query_parse['query_task_type']=='question_answer':
      text_data=query_orchestrator_for_retreival(query_parse,user_query)
      print('User query --- ',user_query)
      print('Below is the JSON output of retrieved chunks:,\n')
      print(json.dumps(text_data, indent=1))

    elif query_parse['query_type']=='numeric' and query_parse['query_task_type']=='single':
      summ=query_orchestrator_for_retreival(query_parse,user_query)
      print('Below is the JSON output of retrieved data:,\n')
      print(summ)

    elif query_parse['query_type']=='numeric' and query_parse['query_task_type']=='timeseries':
      summ,data=query_orchestrator_for_retreival(query_parse,user_query)
      print('Below is the JSON output of retrieved data:,\n')
      print(summ)
      print('\n', print(json.dumps(data, indent=1)))

    elif query_parse['query_type']=='numeric' and query_parse['query_task_type']=='compare_indicator':
      summ,data1,data2=query_orchestrator_for_retreival(query_parse,user_query)
      print('Below is the JSON output of retrieved data:,\n')
      print(summ)
      print(data1)
      print(data2)

    elif query_parse['query_type']=='numeric' and query_parse['query_task_type']=='compare_period':
      summ,data1,data2=query_orchestrator_for_retreival(query_parse,user_query)
      print('Below is the JSON output of retrieved data:,\n')
      print(summ)
      print(data1)
      print(data2)

for query1 in query_lst:
    print('\n\nOutput from reasoning agent for query : ',query1)
    query_parse=query_intent_parser(query1)
    print(query_parse)
    retrieval_relevance_evaluation1(query_parse,query1)
