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
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, models
import openai
from openai import OpenAI
import anthropic
import matplotlib.dates as mdates

# %%
## Load FRED master file and pre-processed FOMC chunk master file.
fred_master_df = pd.read_pickle("fred_master_df.pkl")
fomc_chunk_new_df=pd.read_pickle("fomc_chunk_new_df.pkl")

# %%
## To get OpenAI and Anthropic api key
with open("claude_anthropic_api_key.txt", "r") as f:
  anthropic_api_key = ' '.join(f.readlines())
with open("OpenAI_API_Key.txt", "r") as f:
  openai_api_key = ' '.join(f.readlines())

## Call of cross encoder model
reranker = CrossEncoder("cross-encoder/stsb-roberta-large")

# %% 
## Function to generate parameter from user query for further execution. (JSON output generated from LLM prompt)

def query_parser_prompt_llm(user_query):
  system_message = """ You are an query parser assistant who is expertise in Macroeconomic analysis domain especially with FRED economic indcator and FOMC will help\
  to generate parameters from query in json format. """

  user_message = f"""
  You have knowledge in Economic Domain with FRED and FOMC and your job is to geneerate structured JSON from a given input query.
  You are given the query asked by user in variable <<user_query>> as below:
  <<user_query>> = '{user_query}'
  Your task is to list out the values of paramaters query_type, query_task_type, indicator, aggregation_method, start_dt, end_dt, chart_required from the input query provided in variable <<user_query>> and output these parameter value in JSON format.
  You should only use below values for query_type paramater:
  1. text
  2. numeric
  3. hybrid

  You should only use below values for <<query_task_type>> paramater:
  1. single
  2. timeseries
  3. compare_indicator
  4. compare_period
  5. summary_topic
  6. summary_fomc_statement
  7. summary_fomc_minute
  8. question_answer

   You should only use below values for <<indicator>> paramater:
  1. gdp_rate
  2. employment_rate
  3. unemployment_rate
  4. inflation_5year
  5. inflation_10year
  6. fed_fund_rate

  You should only use below values for <<aggregation_method>> paramater:
  1. Average
  2. Min
  3. Max
  4. Latest
  5. Oldest
  6. Median
  7. standard_deviation
  8. quantile

  You should only use below values for <<chart_required>> paramater:

  You must follow below rules to generate JSON output:
  1. If query can be answered only from FOMC minutes and statements, then 'query_type' parameter value should be 'text'.
  2. If query can be answered only from FRED numeric data, then 'query_type' parameter value should be 'numeric'.
  3. If query needs both FOMC text and FRED numeric, then 'query_type' parameter value should be 'hybrid'.
  4. If the query needs numeric FRED data and asks for a single value at a specific date, then 'query_task_type' should be 'single'.
  5. If the query needs numeric FRED data and ask about trend, pattern, change over timeperiod, then 'query_task_type' should be 'timeseries'.
  6. If the query needs numeric FRED data and asks comparisions between two or more indicators, then 'query_task_type' should be 'compare_indicator'.
  7. If the query needs numeric FRED data and and asks comparisons between different periods of an indicator, then 'query_task_type' should be 'compare_period'.
  8. If the query needs numeric FRED data and ask about separate two or more indicators and there is no word compare, then 'query_task_type' should be 'single'.
  9. If the query needs numeric FRED data and asks for chart or there is a need for chart plot, then 'chart_required' should be 'yes' else it should be 'no'.
  10. If the query needs numeric FRED data and is about gdp growth rate, then 'indicator' should be 'gdp_rate'.
  11. If the query needs numeric FRED data and is about employment growth rate, then 'indicator' should be 'employment_rate'.
  12. If the query needs numeric FRED data and is about unemployment growth rate, then 'indicator' should be 'unemployment_rate'.
  13. If the query needs numeric FRED data and is about the inflation rate in the short term, then 'indicator' should be 'inflation_5year'.
  14. If the query needs numeric FRED data and is about the inflation rate in the long term, then 'indicator' should be 'inflation_10year'.
  15. If the query needs numeric FRED data and is about the federal funds rate, then 'indicator' should be 'fed_fund_rate'.
  16. If the query needs numeric FRED data and asks for more than one indicator, then the 'indicator' parameter should have values in a list containing all the indicators.
  17. If the query needs text FOMC data and asks for a summary, brief, or overview of a topic or indicator, then 'query_task_type' should be 'summary_topic' and you need to identify the indicator and update the indicator in output dictionary item.
      Note that querry is categorised as 'summary' when the output should be a natural language generation  of retrieved data.
  18. If the query needs text FOMC data and asks about a summary, a brief, or an overview of FOMC minutes, then 'query_task_type' should be 'summary_fomc_minute'.
  19. If the query needs text FOMC data and asks about a summary and asks for a summary of FOMC statements, then 'query_task_type' should be 'summary_fomc_statement'.
  20. If the query needs text FOMC data and asks about information of indicators or questions of policy decision or question of facts, then 'query_task_type' should be 'question_answer' and 'indicator' can be empty.
  21. start_dt and end_dt should be in YYYY-MM-DD format.
  22. If start date and end date are not provided in the query, then you must take the last four to five years of data, where start_dt should be "2021-01-01" and end_dt should be "2025-12-31".
  23. If the query needs numeric FRED data and 'query_task_type' is 'compare_period', then identify the start date and end date of both periods and store in start_dt and end_dt as list type.
  24. Do not return values other than those specified in the above list.
  25. For summary_fomc_statement and summary_fomc_minute query_task_type, then you must take the last four to five years of data, where start_dt should be "2021-01-01" and end_dt should be "2025-12-31".
  26. If the query needs numeric FRED data and asks for a single value at a specific date, then 'query_task_type' should be 'single' and <<aggregation_method>> parameter must be only one of the below:
      a. mean : If the query asks for the average or a general value of an indicator within given specific time period.
      b. min : If the query asks for the minimum or lowest value of an indicator within given specific time period.
      c. max : If the query asks for the maximum value of an indicator within a given specific time period.
      d. latest : If the query asks for the latest value or most recent values of an indicator within given specific time period.
      e. oldest : If the query asks for the oldest or earliest value of an indicator within given specific time period.
      f. median : If the query asks for the median or typical value of an indicator.
      g. standard_deviation : If the query asks for the standard deviation or volatility of an indicator.
      h. quantile : If the query asks about quantile or percentile values of an indicator.
      If nothing is provided in the query then you must use 'mean' as aggregation method.
      Do not generate any other aggregation_method other than mean, min, max, latest, oldest, median, standard_deviation and quantile.
  27. For query type 'numeric' and query_task_type other than 'single', then <<aggregation_method>> parameter must be empty.
  28. For 'text' query type the <<aggregation_method>> parameter must be empty.
  You should double verify that output generated is in below exact JSON format:
  {{"query_type": "value",
  "query_task_type": "value",
  "indicator": list of value,
  "aggregation_method": "value",
  "start_dt": date value or list of date values,
  "end_dt": date value or list of date values,
  "chart_required": "value"}}

  Note that query is provided in {user_query} and you must return a response in valid JSON format only.

  Refer below few shot examples to generate accurate JSON response of input user query:

  Example 1:
  user_query : Give some insights about inflation rate.
  response:
  {{"query_type": "text",
  "query_task_type": "question_answer",
  "indicator": [],
  "start_dt": "2025-01-01",
  "end_dt": "2025-12-31",
  "chart_required": "no"}}

  Example 2:
  user_query : Summarize FOMC meeting conducted in September 2023.
  response:
  {{"query_type": "text",
  "query_task_type": "summary_fomc_minute",
  "indicator": [],
  "start_dt": "2023-09-01",
  "end_dt": "2023-09-30",
  "chart_required": "no"}}

  Example 3:
  user_query :What was gdp rate in 2024.
  response:
  {{"query_type": "numeric",
  "query_task_type": "single",
  "indicator": [gdp_rate],
  "aggregation_method": "Average",
  "start_dt": "2024-01-01",
  "end_dt": "2024-12-31",
  "chart_required": "no"}}

  Example 4:
  User query: Show the trend of inflation in the short term from 2020 to 2023.
  response:
  {{"query_type": "numeric",
  "task_type": "timeseries",
  "indicators": ["inflation_5year"],
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "chart_required": "yes"}}

  Example 4:
  User query: What was the FED's view regarding unemployment?
  response:
  {{"query_type": "text",
  "task_type": "question_answer",
  "indicators": [],
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "chart_required": "no"}}

  Example 5:
  User query: Compare employment and unemployment rate.
  Output:
  {{"query_type": "numeric",
  "task_type": "compare_indicator",
  "indicators": ["employment_rate", "unemployment_rate"],
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "chart_required": "yes"}}

  """

  ## To get OpenAI key
  with open("OpenAI_API_Key.txt", "r") as f:
   openai_api_key = ' '.join(f.readlines())

  ## Call of cross encoder model
  client = OpenAI(api_key=openai_api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response.choices[0].message.content

# %%
## Agent A : Query Intent Classification and Parser.

def query_intent_parser(user_query):
  ## Call Parser Prompt via LLM model to obtain query type details.
  llm_query_parse_reply = query_parser_prompt_llm(user_query)
  
  ## To convert string to dictionary type
  import json
  llm_query_parse_dict=json.loads(llm_query_parse_reply)
  return(llm_query_parse_dict)

# %%
## Retrieval function 1 : Numeric and Single query type

## Designing of function for each query intent type.
## Below function is to be used for numeric type of query and with intent of single query.

def numeric_single_query_retreival(llm_query_parse_dict,user_query):

  # Map the parser indicators to actual indicator labels
  indicator_map = {
        "gdp_rate": "GDP Growth Rate",
        "fed_fund_rate": "Federal Funds Rate",
        "employment_rate": "Employment Rate",
        "unemployment_rate": "Un-employment Rate",
        "inflation_5year": "Inflation Rate 5Year",
        "inflation_10year": "Inflation Rate 10Year"}

  start_dt = pd.to_datetime(llm_query_parse_dict['start_dt'])
  end_dt   = pd.to_datetime(llm_query_parse_dict['end_dt'])

  indicator_label=[indicator_map[i] for i in llm_query_parse_dict['indicator']]
  numeric_single_query_answer_for_llm={}
  numeric_single_query_answer_for_llm['query_type']=llm_query_parse_dict['query_type']
  numeric_single_query_answer_for_llm['query_task_type']=llm_query_parse_dict['query_task_type']
  numeric_single_query_answer_for_llm['start_dt']=str(start_dt.date())
  numeric_single_query_answer_for_llm['end_dt']=str(end_dt.date())
  numeric_single_query_answer_for_llm['indicator']=indicator_label
  numeric_single_query_answer_for_llm['chart_required']='no'
  numeric_single_query_answer_for_llm['user_query']=user_query
  query_result=[]
  # To obtain mean of data.
  if llm_query_parse_dict['aggregation_method'].lower()=='mean':
    res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].mean()
  # To obtain a minimum of data
  elif llm_query_parse_dict['aggregation_method'].lower()=='min':
    res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].min()
   # To obtain a maximum of data
  elif llm_query_parse_dict['aggregation_method'].lower()=='max':
    res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].max()
   # To obtain latest data
  elif llm_query_parse_dict['aggregation_method'].lower()=='latest':
    res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].sort_values(by='Date').tail(1)
   # To obtain oldest data
  elif llm_query_parse_dict['aggregation_method'].lower()=='oldest':
    res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].sort_values(by='Date').head(1)
  # To obtain median of data
  elif llm_query_parse_dict['aggregation_method'].lower()=='median':
    res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].median()
  # To obtain standard deviation of data
  elif llm_query_parse_dict['aggregation_method'].lower()=='standard_deviation':
    res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].std()
  # To obtain percentile of data
  elif llm_query_parse_dict['aggregation_method'].lower()=='quantile':
    import re
    pattern='(\d+)(th|nd|st|rd)*\s*(percentile|quantile|percent|percentage|%)'
    match=re.search(pattern,user_query.lower())
    if match:
      if int(match.group(1))>=1 and int(match.group(1))<=100:
        value=int(match.group(1))
        val_decimal= round(value/100,2)
        res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].quantile(val_decimal)
        numeric_single_query_answer_for_llm['percentile_value']=value
      else:
          user_query_correct=input("""Please re-type query with correct percentile/quantile value ranging from 1-100, for example '20th percentile', '75 percentile', '90%' etc.
          """)
          match=re.search(pattern,user_query_correct.lower())
          value=int(match.group(1))
          val_decimal= round(value/100,2)
          res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].quantile(val_decimal)
          numeric_single_query_answer_for_llm['percentile_value']=value
          numeric_single_query_answer_for_llm['user_query']=user_query_correct
    else:
        user_query_correct=input("Please provide correct percentile/quantile value, for example '20th percentile', '75 percentile', '90%' etc.")
        match=re.search(pattern,user_query_correct.lower())
        value=int(match.group(1))
        val_decimal= round(value/100,2)
        res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].quantile(val_decimal)
        numeric_single_query_answer_for_llm['percentile_value']=value
        numeric_single_query_answer_for_llm['user_query']=user_query_correct
  # To get average if none provided
  else:
    res=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)].groupby('economic_indicator_desc')['rate_value_%'].mean()


  for i,row in res.reset_index().iterrows():
    query_result.append({'indicator_label':row['economic_indicator_desc'],'rate_value_%':round(row['rate_value_%'],2), 'unit':'percentage'})

  numeric_single_query_answer_for_llm['aggregation_method']=llm_query_parse_dict['aggregation_method']
  numeric_single_query_answer_for_llm['numeric_query_result']=query_result

  return(numeric_single_query_answer_for_llm)



# %%
##### Retrieval function 2 : Numeric and Timeseries/trend query type

## Designing of function for each query intent type.
## Below function is to be used for numeric type of query and with intent of trend/timeseries type query.

def numeric_timeseries_query_retreival(llm_query_parse_dict,user_query):

  # Map the parser indicators to actual indicator labels
  indicator_map = {
        "gdp_rate": "GDP Growth Rate",
        "fed_fund_rate": "Federal Funds Rate",
        "employment_rate": "Employment Rate",
        "unemployment_rate": "Un-employment Rate",
        "inflation_5year": "Inflation Rate 5Year",
        "inflation_10year": "Inflation Rate 10Year"}

  start_dt = pd.to_datetime(llm_query_parse_dict['start_dt'])
  end_dt   = pd.to_datetime(llm_query_parse_dict['end_dt'])

  indicator_label=[indicator_map[i] for i in llm_query_parse_dict['indicator']]

  # To get the list of timeseries data
  timeseries_data=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)][['rate_value_%','Date']]
  # To get the slop of trend/timeseries data
  x = np.arange(len(timeseries_data))
  slope = round(np.polyfit(x, timeseries_data['rate_value_%'], 1)[0],3)
  # to get total peaks(lows and ups) to identoify volatility.
  lows=0
  ups = 0
  for i in range(1,len(timeseries_data)-1):
    if (timeseries_data['rate_value_%'].iloc[i]>timeseries_data['rate_value_%'].iloc[i-1]) & \
      (timeseries_data['rate_value_%'].iloc[i]>timeseries_data['rate_value_%'].iloc[i+1]):
      ups = ups+1
    if (timeseries_data['rate_value_%'].iloc[i]<timeseries_data['rate_value_%'].iloc[i-1]) & \
      (timeseries_data['rate_value_%'].iloc[i]<timeseries_data['rate_value_%'].iloc[i+1]):
      lows = lows+1
  peaks=lows+ups
  peak_ratio = round(peaks/(len(timeseries_data)-2),3)
  # to get coefficient variation to identify volatility.
  cv = np.std(timeseries_data['rate_value_%'])/abs(np.mean(timeseries_data['rate_value_%']))
  # to prepare timeseries data output in dictionary format for llm
  timeseries_values_for_llm={}
  slop_rel=slope/abs(np.mean(timeseries_data['rate_value_%']))
  # To identofy the slope trend
  if  abs(slop_rel)<0.01:
    timeseries_values_for_llm['trend']='Stable'
  elif abs(slop_rel)<0.05:
    if slope>0:
      timeseries_values_for_llm['trend']='Moderate Upward'
    else:
      timeseries_values_for_llm['trend']='Moderate Downward'
  else:
    if slope>0:
      timeseries_values_for_llm['trend']='Strong Upward'
    else:
      timeseries_values_for_llm['trend']='Strong Downward'

  # To identify volatility
  if len(timeseries_data)<3:
    timeseries_values_for_llm['volatility']='Insufficient'
  else:
    if (cv<0.05) & (peak_ratio<0.10):
      timeseries_values_for_llm['volatility']='Low Volatility'
    elif (cv<0.10) & (peak_ratio<0.20):
      timeseries_values_for_llm['volatility']='Moderate Volatility'
    else:
      timeseries_values_for_llm['volatility']='High Volatility'

  # Chart visulaisation is required for facts.
  timeseries_values_for_llm['chart_required']='yes'
  # Add other required information for llm to generate fact data.
  timeseries_values_for_llm['user_query']=user_query
  timeseries_values_for_llm['query_task_type']=llm_query_parse_dict['query_task_type']
  timeseries_values_for_llm['high_value']=timeseries_data['rate_value_%'].max()
  timeseries_values_for_llm['low_value']=timeseries_data['rate_value_%'].min()
  timeseries_values_for_llm['average_value']=float(timeseries_data['rate_value_%'].mean())
  timeseries_values_for_llm['first_value']=float(timeseries_data['rate_value_%'].iloc[0])
  timeseries_values_for_llm['last_value']=float(timeseries_data['rate_value_%'].iloc[-1])
  timeseries_values_for_llm['slope_value']=float(slope)
  timeseries_values_for_llm['start_date']=str(start_dt.date())
  timeseries_values_for_llm['end_date']=str(end_dt.date())
  timeseries_values_for_llm['indicator']=indicator_label

  return timeseries_values_for_llm,timeseries_data



# %%
#### Retrieval function 3 : Numeric and multiple indicator Comparision query type

## Designing of function for each query intent type.
## Below function is to be used for numeric type of query and with intent of comparision between two indicators query.

def numeric_compare_multiindicator_query_retreival(llm_query_parse_dict):

  # Map the parser indicators to actual indicator labels
  indicator_map = {
        "gdp_rate": "GDP Growth Rate",
        "fed_fund_rate": "Federal Funds Rate",
        "employment_rate": "Employment Rate",
        "unemployment_rate": "Un-employment Rate",
        "inflation_5year": "Inflation Rate 5Year",
        "inflation_10year": "Inflation Rate 10Year"}

  start_dt = pd.to_datetime(llm_query_parse_dict['start_dt'])
  end_dt   = pd.to_datetime(llm_query_parse_dict['end_dt'])

  indicator_label=[indicator_map[i] for i in llm_query_parse_dict['indicator']]

  if len(indicator_label)>1:

  # To get the list of timeseries data of both indicator separately for comparision
    timeseries_data_1=fred_master_df[(fred_master_df['economic_indicator_desc']==(indicator_label[0])) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)][['rate_value_%','Date']]
    timeseries_data_2=fred_master_df[(fred_master_df['economic_indicator_desc']==(indicator_label[1])) & \
                (fred_master_df['Date']>=start_dt)\
                & (fred_master_df['Date']<=end_dt)][['rate_value_%','Date']]
  # To get the slop of both indicator separately for comparision
  x1 = np.arange(len(timeseries_data_1))
  slope1 = round(np.polyfit(x1, timeseries_data_1['rate_value_%'], 1)[0],3)
  x2= np.arange(len(timeseries_data_2))
  slope2 = round(np.polyfit(x2, timeseries_data_2['rate_value_%'], 1)[0],3)
  # to get total peaks(lows and ups) to identoify volatility.
  lows=0
  ups = 0
  for i in range(1,len(timeseries_data_1)-1):
    if (timeseries_data_1['rate_value_%'].iloc[i]>timeseries_data_1['rate_value_%'].iloc[i-1]) & \
      (timeseries_data_1['rate_value_%'].iloc[i]>timeseries_data_1['rate_value_%'].iloc[i+1]):
      ups = ups+1
    if (timeseries_data_1['rate_value_%'].iloc[i]<timeseries_data_1['rate_value_%'].iloc[i-1]) & \
      (timeseries_data_1['rate_value_%'].iloc[i]<timeseries_data_1['rate_value_%'].iloc[i+1]):
      lows = lows+1
  peaks1=lows+ups
  peak_ratio1 = round(peaks1/(len(timeseries_data_1)-2),3)
  lows=0
  ups = 0
  for i in range(1,len(timeseries_data_2)-1):
    if (timeseries_data_2['rate_value_%'].iloc[i]>timeseries_data_2['rate_value_%'].iloc[i-1]) & \
      (timeseries_data_2['rate_value_%'].iloc[i]>timeseries_data_2['rate_value_%'].iloc[i+1]):
      ups = ups+1
    if (timeseries_data_2['rate_value_%'].iloc[i]<timeseries_data_2['rate_value_%'].iloc[i-1]) & \
      (timeseries_data_2['rate_value_%'].iloc[i]<timeseries_data_2['rate_value_%'].iloc[i+1]):
      lows = lows+1
  peaks2=lows+ups
  peak_ratio2 = round(peaks2/(len(timeseries_data_2)-2),3)
  # to get coefficient variation to identify volatility.
  cv1 = np.std(timeseries_data_1['rate_value_%'])/abs(np.mean(timeseries_data_1['rate_value_%']))
  cv2 = np.std(timeseries_data_2['rate_value_%'])/abs(np.mean(timeseries_data_2['rate_value_%']))
  # to prepare timeseries data output in dictionary format for llm
  comparision_values_for_llm={}

  # Add other required information for llm to generate fact data.

  comparision_values_for_llm['type_of_comparision']='multiple_indicator'
  comparision_values_for_llm['indicator_1']={'indicator_label':indicator_label[0],\
                                             'high_value':timeseries_data_1['rate_value_%'].max(),\
                                             'low_value':timeseries_data_1['rate_value_%'].min(),\
                                             'average_value':round(float(timeseries_data_1['rate_value_%'].mean()),2),\
                                             'first_value':float(timeseries_data_1['rate_value_%'].iloc[0]),\
                                             'last_value':float(timeseries_data_1['rate_value_%'].iloc[-1]),\
                                             'slope_value':float(slope1),\
                                             'start_date':str(start_dt.date()),\
                                             'end_date':str(end_dt.date())
                                            }
  comparision_values_for_llm['indicator_2']={'indicator_label':indicator_label[1],\
                                             'high_value':timeseries_data_2['rate_value_%'].max(),\
                                             'low_value':timeseries_data_2['rate_value_%'].min(),\
                                             'average_value':round(float(timeseries_data_2['rate_value_%'].mean()),2),\
                                             'first_value':float(timeseries_data_2['rate_value_%'].iloc[0]),\
                                             'last_value':float(timeseries_data_2['rate_value_%'].iloc[-1]),\
                                             'slope_value':float(slope2),\
                                             'start_date':str(start_dt.date()),\
                                             'end_date':str(end_dt.date())
                                            }
  # Chart visulaisation is required for facts.
  comparision_values_for_llm['chart_required']='yes'

  slop_rel1=slope1/abs(np.mean(timeseries_data_1['rate_value_%']))
  slop_rel2=slope1/abs(np.mean(timeseries_data_2['rate_value_%']))

  # To identify the slope trend for first indicator
  if  abs(slop_rel1)<0.01:
    comparision_values_for_llm['indicator_1']['trend']='Stable'
  elif abs(slop_rel1)<0.05:
    if slope1>0:
      comparision_values_for_llm['indicator_1']['trend']='Moderate Upward'
    else:
      comparision_values_for_llm['indicator_1']['trend']='Moderate Downward'
  else:
    if slope1>0:
      comparision_values_for_llm['indicator_1']['trend']='Strong Upward'
    else:
      comparision_values_for_llm['indicator_1']['trend']='Strong Downward'

    # To identify the slope trend for second indicator
  if  abs(slop_rel2)<0.01:
    comparision_values_for_llm['indicator_2']['trend']='Stable'
  elif abs(slop_rel2)<0.05:
    if slope2>0:
      comparision_values_for_llm['indicator_2']['trend']='Moderate Upward'
    else:
      comparision_values_for_llm['indicator_2']['trend']='Moderate Downward'
  else:
    if slope2>0:
      comparision_values_for_llm['indicator_2']['trend']='Strong Upward'
    else:
      comparision_values_for_llm['indicator_2']['trend']='Strong Downward'

  # To identify volatility for first indicator
  if len(timeseries_data_1)<3:
    comparision_values_for_llm['indicator_1']['volatility']='Insufficient'
  else:
    if (cv1<0.05) & (peak_ratio1<0.10):
      comparision_values_for_llm['indicator_1']['volatility']='Low Volatility'
    elif (cv1<0.10) & (peak_ratio1<0.20):
      comparision_values_for_llm['indicator_1']['volatility']='Moderate Volatility'
    else:
      comparision_values_for_llm['indicator_1']['volatility']='High Volatility'


  # To identify volatility for second indicator
  if len(timeseries_data_2)<3:
    comparision_values_for_llm['indicator_2']['volatility']='Insufficient'
  else:
    if (cv2<0.05) & (peak_ratio2<0.10):
      comparision_values_for_llm['indicator_2']['volatility']='Low Volatility'
    elif (cv2<0.10) & (peak_ratio2<0.20):
      comparision_values_for_llm['indicator_2']['volatility']='Moderate Volatility'
    else:
      comparision_values_for_llm['indicator_2']['volatility']='High Volatility'

  # To identify higher of average values of two indicators for comparision metrix
  if timeseries_data_1['rate_value_%'].mean()>timeseries_data_2['rate_value_%'].mean():
    comparision_values_for_llm['indicator_higher_rate']=indicator_label[0]
    comparision_values_for_llm['difference_of_indicators_average']=round(float((timeseries_data_1['rate_value_%'].mean())-(timeseries_data_2['rate_value_%'].mean())),2)
  else:
    comparision_values_for_llm['indicator_higher_rate']=indicator_label[1]
    comparision_values_for_llm['difference_of_indicators_average']=round(float((timeseries_data_2['rate_value_%'].mean())-(timeseries_data_1['rate_value_%'].mean())),2)

  return comparision_values_for_llm,timeseries_data_1,timeseries_data_2




# %%
#### Retrieval function 4 : Numeric and multi duration Comparision query type

## Designing of function for each query intent type.
## Below function is to be used for numeric type of query and with intent of comparision between two period range query.

def numeric_compare_multiperiod_query_retreival(llm_query_parse_dict):

  # Map the parser indicators to actual indicator labels
  indicator_map = {
        "gdp_rate": "GDP Growth Rate",
        "fed_fund_rate": "Federal Funds Rate",
        "employment_rate": "Employment Rate",
        "unemployment_rate": "Un-employment Rate",
        "inflation_5year": "Inflation Rate 5Year",
        "inflation_10year": "Inflation Rate 10Year"}

  start_dt = pd.to_datetime(llm_query_parse_dict['start_dt'])
  end_dt   = pd.to_datetime(llm_query_parse_dict['end_dt'])

  indicator_label=[indicator_map[i] for i in llm_query_parse_dict['indicator']]

  # To get the list of timeseries data of both time period separately for comparision
  timeseries_data_1=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt[0])\
                & (fred_master_df['Date']<=end_dt[0])][['rate_value_%','Date']]
  timeseries_data_2=fred_master_df[(fred_master_df['economic_indicator_desc'].isin(indicator_label)) & \
                (fred_master_df['Date']>=start_dt[1])\
                & (fred_master_df['Date']<=end_dt[1])][['rate_value_%','Date']]
  # To get the slop of both duration data separately for comparision
  x1 = np.arange(len(timeseries_data_1))
  slope1 = round(np.polyfit(x1, timeseries_data_1['rate_value_%'], 1)[0],3)
  x2= np.arange(len(timeseries_data_2))
  slope2 = round(np.polyfit(x2, timeseries_data_2['rate_value_%'], 1)[0],3)
  # to get total peaks(lows and ups) to identoify volatility.
  lows=0
  ups = 0
  for i in range(1,len(timeseries_data_1)-1):
    if (timeseries_data_1['rate_value_%'].iloc[i]>timeseries_data_1['rate_value_%'].iloc[i-1]) & \
      (timeseries_data_1['rate_value_%'].iloc[i]>timeseries_data_1['rate_value_%'].iloc[i+1]):
      ups = ups+1
    if (timeseries_data_1['rate_value_%'].iloc[i]<timeseries_data_1['rate_value_%'].iloc[i-1]) & \
      (timeseries_data_1['rate_value_%'].iloc[i]<timeseries_data_1['rate_value_%'].iloc[i+1]):
      lows = lows+1
  peaks1=lows+ups
  peak_ratio1 = round(peaks1/(len(timeseries_data_1)-2),3)
  lows=0
  ups = 0
  for i in range(1,len(timeseries_data_2)-1):
    if (timeseries_data_2['rate_value_%'].iloc[i]>timeseries_data_2['rate_value_%'].iloc[i-1]) & \
      (timeseries_data_2['rate_value_%'].iloc[i]>timeseries_data_2['rate_value_%'].iloc[i+1]):
      ups = ups+1
    if (timeseries_data_2['rate_value_%'].iloc[i]<timeseries_data_2['rate_value_%'].iloc[i-1]) & \
      (timeseries_data_2['rate_value_%'].iloc[i]<timeseries_data_2['rate_value_%'].iloc[i+1]):
      lows = lows+1
  peaks2=lows+ups
  peak_ratio2 = round(peaks2/(len(timeseries_data_2)-2),3)
  # to get coefficient variation to identify volatility.
  cv1 = np.std(timeseries_data_1['rate_value_%'])/abs(np.mean(timeseries_data_1['rate_value_%']))
  cv2 = np.std(timeseries_data_2['rate_value_%'])/abs(np.mean(timeseries_data_2['rate_value_%']))
  # to prepare timeseries data output in dictionary format for llm
  comparision_values_for_llm={}

  # Add other required information for llm to generate fact data.

  comparision_values_for_llm['type_of_comparision']='compare_multiple_time_period'
  comparision_values_for_llm['period_1']={'indicator_label':indicator_label,\
                                             'high_value':timeseries_data_1['rate_value_%'].max(),\
                                             'low_value':timeseries_data_1['rate_value_%'].min(),\
                                             'average_value':round(float(timeseries_data_1['rate_value_%'].mean()),2),\
                                             'first_value':float(timeseries_data_1['rate_value_%'].iloc[0]),\
                                             'last_value':float(timeseries_data_1['rate_value_%'].iloc[-1]),\
                                             'slope_value':float(slope1),\
                                             'period1_range':str(start_dt[0].date())+' to '+str(end_dt[0].date())
                                            }
  comparision_values_for_llm['period_2']={'indicator_label':indicator_label,\
                                             'high_value':timeseries_data_2['rate_value_%'].max(),\
                                             'low_value':timeseries_data_2['rate_value_%'].min(),\
                                             'average_value':round(float(timeseries_data_2['rate_value_%'].mean()),2),\
                                             'first_value':float(timeseries_data_2['rate_value_%'].iloc[0]),\
                                             'last_value':float(timeseries_data_2['rate_value_%'].iloc[-1]),\
                                             'slope_value':float(slope2),\
                                             'period2_range':str(start_dt[1].date())+' to '+str(end_dt[1].date())
                                            }
  # Chart visulaisation is required for facts.
  comparision_values_for_llm['chart_required']='yes'

  slop_rel1=slope1/abs(np.mean(timeseries_data_1['rate_value_%']))
  slop_rel2=slope1/abs(np.mean(timeseries_data_2['rate_value_%']))

  # To identify the slope trend for first indicator
  if  abs(slop_rel1)<0.01:
    comparision_values_for_llm['period_1']['trend']='Stable'
  elif abs(slop_rel1)<0.05:
    if slope1>0:
      comparision_values_for_llm['period_1']['trend']='Moderate Upward'
    else:
      comparision_values_for_llm['period_1']['trend']='Moderate Downward'
  else:
    if slope1>0:
      comparision_values_for_llm['period_1']['trend']='Strong Upward'
    else:
      comparision_values_for_llm['period_1']['trend']='Strong Downward'

    # To identify the slope trend for second indicator
  if  abs(slop_rel2)<0.01:
    comparision_values_for_llm['period_2']['trend']='Stable'
  elif abs(slop_rel2)<0.05:
    if slope2>0:
      comparision_values_for_llm['period_2']['trend']='Moderate Upward'
    else:
      comparision_values_for_llm['period_2']['trend']='Moderate Downward'
  else:
    if slope2>0:
      comparision_values_for_llm['period_2']['trend']='Strong Upward'
    else:
      comparision_values_for_llm['period_2']['trend']='Strong Downward'

  # To identify volatility for first indicator
  if len(timeseries_data_1)<3:
    comparision_values_for_llm['period_1']['volatility']='Insufficient'
  else:
    if (cv1<0.05) & (peak_ratio1<0.10):
      comparision_values_for_llm['period_1']['volatility']='Low Volatility'
    elif (cv1<0.10) & (peak_ratio1<0.20):
      comparision_values_for_llm['period_1']['volatility']='Moderate Volatility'
    else:
      comparision_values_for_llm['period_1']['volatility']='High Volatility'


  # To identify volatility for second indicator
  if len(timeseries_data_2)<3:
    comparision_values_for_llm['period_2']['volatility']='Insufficient'
  else:
    if (cv2<0.05) & (peak_ratio2<0.10):
      comparision_values_for_llm['period_2']['volatility']='Low Volatility'
    elif (cv2<0.10) & (peak_ratio2<0.20):
      comparision_values_for_llm['period_2']['volatility']='Moderate Volatility'
    else:
      comparision_values_for_llm['period_2']['volatility']='High Volatility'

  # To identify higher of average values of two indicators for comparision metrix
  if timeseries_data_1['rate_value_%'].mean()>timeseries_data_2['rate_value_%'].mean():
    comparision_values_for_llm['period_having_higher_rate']=str(start_dt[0].date())+' to '+str(end_dt[0].date())
    comparision_values_for_llm['difference_of_rate_average']=round(float((timeseries_data_1['rate_value_%'].mean())-(timeseries_data_2['rate_value_%'].mean())),2)
  else:
    comparision_values_for_llm['period_having_higher_rate']=str(start_dt[1].date())+' to '+str(end_dt[1].date())
    comparision_values_for_llm['difference_of_rate_average']=round(float((timeseries_data_2['rate_value_%'].mean())-(timeseries_data_1['rate_value_%'].mean())),2)

  # To add inidicator values in both period range for llm
  comparision_values_for_llm['period_1']['indicator_label']=indicator_label
  comparision_values_for_llm['period_2']['indicator_label']=indicator_label

  return comparision_values_for_llm,timeseries_data_1,timeseries_data_2



# %%
#### Retrieval function 5 : Text and document summary query type

## Designing of function for each query intent type.
## Below function is to be used for text type of query and with intent of document summarisation query.

def text_document_summary_query_retreival(llm_query_parse_dict):

  # Map the parser indicators to actual indicator labels
  indicator_map = {
        "gdp_rate": "GDP Growth Rate",
        "fed_fund_rate": "Federal Funds Rate",
        "employment_rate": "Employment Rate",
        "unemployment_rate": "Un-employment Rate",
        "inflation_5year": "Inflation Rate 5Year",
        "inflation_10year": "Inflation Rate 10Year"}

  start_dt = pd.to_datetime(llm_query_parse_dict['start_dt'])
  end_dt   = pd.to_datetime(llm_query_parse_dict['end_dt'])
  text_summary_doc_for_llm={}
  text_summary_doc_for_llm['chart_required']='no'
  indicator_label=[indicator_map[i] for i in llm_query_parse_dict['indicator']]
  if llm_query_parse_dict['query_task_type']=='summary_fomc_minute':
    summary_doc_df = fomc_chunk_new_df[(fomc_chunk_new_df.doc_type=='Minute') & (fomc_chunk_new_df.meeting_date>=start_dt) & \
                                                                              (fomc_chunk_new_df.meeting_date<=end_dt)]
    retreievd_doc_with_timestamp = pd.DataFrame(summary_doc_df.groupby('meeting_date')['chunked_text'].apply(lambda x: "\n\n".join(x)))
    retreievd_doc_with_timestamp.reset_index(inplace=True)
    retreievd_doc_with_timestamp['meeting_date']=pd.to_datetime(retreievd_doc_with_timestamp['meeting_date']).dt.strftime("%Y-%m-%d")
    retreievd_doc_with_timestamp=retreievd_doc_with_timestamp.to_dict(orient="records")
    text_summary_doc_for_llm['document_retreived_for_sumamry']=retreievd_doc_with_timestamp
    text_summary_doc_for_llm['query_type']='document_summary'
    text_summary_doc_for_llm['document_type']='FOMC_minute'
    text_summary_doc_for_llm['start_date']=str(start_dt.date())
    text_summary_doc_for_llm['end_date']=str(end_dt.date())
    return text_summary_doc_for_llm
  elif llm_query_parse_dict['query_task_type']=='summary_fomc_statement':
    summary_doc_df = fomc_chunk_new_df[(fomc_chunk_new_df.doc_type=='Statement') & (fomc_chunk_new_df.meeting_date>=start_dt) & \
                                                                                  (fomc_chunk_new_df.meeting_date<=end_dt)]
    retreievd_doc_with_timestamp = pd.DataFrame(summary_doc_df.groupby('meeting_date')['chunked_text'].apply(lambda x: "\n\n".join(x)))
    retreievd_doc_with_timestamp.reset_index(inplace=True)
    retreievd_doc_with_timestamp['meeting_date']=pd.to_datetime(retreievd_doc_with_timestamp['meeting_date']).dt.strftime("%Y-%m-%d")
    retreievd_doc_with_timestamp=retreievd_doc_with_timestamp.to_dict(orient="records")
    text_summary_doc_for_llm['document_retreived_for_sumamry']=retreievd_doc_with_timestamp
    text_summary_doc_for_llm['query_type']='document_summary'
    text_summary_doc_for_llm['document_type']='FOMC_statement'
    text_summary_doc_for_llm['start_date']=str(start_dt.date())
    text_summary_doc_for_llm['end_date']=str(end_dt.date())
    return text_summary_doc_for_llm


# %%
### Function to rewrite user query for effective semantic retrieval through llm.
## Function to rewrite user query using claude sonnet llm for generating efficient query.

def query_rewriter_llm(user_query):
  ## To get Anthropic key.
  with open("claude_anthropic_api_key.txt", "r") as f:
    anthropic_api_key = ' '.join(f.readlines())
  
  ## Call claude conet model using api key to rewrite text type of query.
  client = anthropic.Anthropic(api_key=anthropic_api_key)

  original_query=user_query
  response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=150,
    system="""You are an expert in Federal Reserve communications and monetary policy domain and excel at rewriting original query to generate optimized output query.\
    You should rewrite the user's original query for effective searching in FOMC meeting minutes and policy statements.\
    Your primary aim is to expand relevant financial terminology, provide synonyms, and make the intent explicit.\
     You must return only the rewritten query in the output and do not add any labels, do not provide an explanation, and do not add the original query.""",
    messages=[
        {"role": "user", "content": f"Rewrite this query for better retrieval: {original_query}"}
    ])
  llm_output={"rewritten_query":response.content[0].text, "original_query":original_query}
  return llm_output

# %%
#### Function to call chroma database collection for semantic retrieval.
## First upload chroma db collection database that was generated from 'FOMC_and_FRED_RAG.ipynb' python code.

def fomc_collection_load():
    # Path of stored collection database.
    coll_path = os.path.join(os.getcwd(), "FOMC_ChromaDB_Data_v2")
    collection_name = "FOMC_Chroma_Client1"
    model_name = "mukaj/fin-mpnet-base"

    embedd_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="mukaj/fin-mpnet-base")

    client = chromadb.PersistentClient(path=coll_path)

    chroma_collection = client.get_collection(name=collection_name,embedding_function=embedd_function)

    return chroma_collection


# %%
#### Function for query search in chroma db collection and retrieval of top 15 semantic chunks.

## Defining Function for query search in chroma vector DB.

def query_engine(user_query,chroma_collection,start_dt,end_dt):
    start_year = pd.to_datetime(start_dt).year
    end_year = pd.to_datetime(end_dt).year

    if start_dt is not None and end_dt is not None:
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


# %%
## Re-ranking function to rerank top 15 chunks retrieved at first instance, based on similarity scores for further refinement.
## Re-ranking function using crossencoder model for retrieval of more semantic similar documents from top 15 docuemnts retreievd from chromadb.

def rerank_top15_document(user_query,top_15_similar_document):

  ## Creating pair of document(top 10 documents obtianed from engiene query) and query list for re-ranking score.
    cross_encoder_sentence_query_pair = [[user_query,x] for x in top_15_similar_document.Document]

  ## Call of cross encoder model
    reranker = CrossEncoder("cross-encoder/stsb-roberta-large")
    score=reranker.predict(cross_encoder_sentence_query_pair)

  ## Normalising score obtained above in range[0-1] by applying sigmoid function.
    score_norm = [round(float(1/(1+np.exp(-scr))),4) for scr in score]

  ## Updating score and normalised score in dataframe.
    top_15_similar_document['Encoder_Norm_score']=score_norm
    top_15_similar_document['Encoder_score']=score

  ## return sorted table by sorting Norm Score in Descending order.
    return(top_15_similar_document.sort_values(by='Encoder_Norm_score',ascending=False))


# %%
#### Retrieval function 6 : Text and topic wise summary query type

## Designing of function for each query intent type.
## Below function is to be used for text type of query and with intent of topic wise summarisation query.

def text_topic_summary_query_retreival(llm_query_parse_dict,user_query):

  # Map the parser indicators to actual indicator labels
  indicator_map = {
        "gdp_rate": "GDP Growth Rate",
        "fed_fund_rate": "Federal Funds Rate",
        "employment_rate": "Employment Rate",
        "unemployment_rate": "Un-employment Rate",
        "inflation_5year": "Inflation Rate 5Year",
        "inflation_10year": "Inflation Rate 10Year"}

  start_dt = pd.to_datetime(llm_query_parse_dict['start_dt'])
  end_dt   = pd.to_datetime(llm_query_parse_dict['end_dt'])
  indicator_label=[indicator_map[i] for i in llm_query_parse_dict['indicator']]
  text_summary_item_for_llm={}
  text_summary_item_for_llm['chart_required']='no'
  ## Call query rewriter function to generate efficient query.
  llm_output=query_rewriter_llm(user_query)
  ## Call collection load function for semantic retrieval of text.
  chroma_collection=fomc_collection_load()
  ## Call query engine function to retrieve top 15 semantic chunks from embed collection.
  top_15_similar_document=query_engine(llm_output['rewritten_query'],chroma_collection,start_dt,end_dt)
  ## Calling re-rank function to get normalised re-ranked score of top 15 documents for refined search result.
  top_15_doc_reranked_score_df= rerank_top15_document(llm_output['rewritten_query'],top_15_similar_document)
  ## Return top 10 documents based on Encoder score for llm to provide summary of chunks based on user topic.
  document_retreived_for_sumamry = top_15_doc_reranked_score_df.sort_values(by='Encoder_Norm_score',ascending=False)[0:10]
  text_summary_item_for_llm['query_type']='topic_summary'
  text_summary_item_for_llm['original_user_query']=user_query
  text_summary_item_for_llm['rewritten_user_query']=llm_output['rewritten_query']

  metadatas_list = document_retreived_for_sumamry['Metadatas'].tolist()
  documents_list = document_retreived_for_sumamry['Document'].tolist()

  combined_retrieval_list = []

  for meta, doc in zip(metadatas_list, documents_list):
    combined_retrieval_list.append({**meta,"document_text": doc})

  text_summary_item_for_llm['documents_retrieved_for_summary'] = combined_retrieval_list
  return text_summary_item_for_llm


# %%
#### Retrieval function 7 : Text and question answer query type

## Designing of function for each query intent type.
## Below function is to be used for text type of query and with intent of qyestion answer type query.

def text_question_answer_query_retreival(llm_query_parse_dict,user_query):

  # Map the parser indicators to actual indicator labels
  indicator_map = {
        "gdp_rate": "GDP Growth Rate",
        "fed_fund_rate": "Federal Funds Rate",
        "employment_rate": "Employment Rate",
        "unemployment_rate": "Un-employment Rate",
        "inflation_5year": "Inflation Rate 5Year",
        "inflation_10year": "Inflation Rate 10Year"}

  start_dt = pd.to_datetime(llm_query_parse_dict['start_dt'])
  end_dt   = pd.to_datetime(llm_query_parse_dict['end_dt'])
  indicator_label=[indicator_map[i] for i in llm_query_parse_dict['indicator']]
  text_question_answer_for_llm={}
  text_question_answer_for_llm['chart_required']='no'
  ## Call query rewriter function to generate efficient query.
  llm_output=query_rewriter_llm(user_query)
  ## Call collection load function for semantic retrieval of text.
  chroma_collection=fomc_collection_load()
  ## Call query engine function to retrieve top 15 semantic chunks from embed collection.
  top_15_similar_document=query_engine(llm_output['rewritten_query'],chroma_collection,start_dt,end_dt)
  ## Calling re-rank function to get normalised re-ranked score of top 15 documents for refined search result.
  top_15_doc_reranked_score_df= rerank_top15_document(llm_output['rewritten_query'],top_15_similar_document)
  ## Return top 10 documents based on Encoder score for llm to provide summary of chunks based on user topic.
  document_retreived_for_query = top_15_doc_reranked_score_df.sort_values(by='Encoder_Norm_score',ascending=False)[0:5]
  text_question_answer_for_llm['query_type']=llm_query_parse_dict['query_task_type']
  text_question_answer_for_llm['original_user_query']=user_query
  text_question_answer_for_llm['rewritten_user_query']=llm_output['rewritten_query']

  metadatas_list = document_retreived_for_query['Metadatas'].tolist()
  documents_list = document_retreived_for_query['Document'].tolist()

  combined_retrieval_list = []

  for meta, doc in zip(metadatas_list, documents_list):
    combined_retrieval_list.append({**meta,"document_text": doc})

  text_question_answer_for_llm['documents_retrieved_for_question_answer'] = combined_retrieval_list
  return text_question_answer_for_llm

# %%
### Function defined for orchestartion of query to required document retrival functions.

def query_orchestrator_for_retreival(llm_query_parse_dict,user_query):
  ## Route numeric query type to correct functions for retrieval.
  if llm_query_parse_dict['query_type']=='numeric':   
    if llm_query_parse_dict['query_task_type']=='single':
      qry_res=numeric_single_query_retreival(llm_query_parse_dict,user_query)
      return qry_res
    elif llm_query_parse_dict['query_task_type']=='timeseries':
      timeseries_values_for_llm,timeseries_data=numeric_timeseries_query_retreival(llm_query_parse_dict,user_query)
      return timeseries_values_for_llm,timeseries_data
    elif llm_query_parse_dict['query_task_type']=='compare_indicator':
      comparision_values_for_llm,timeseries_data_1,timeseries_data_2=numeric_compare_multiindicator_query_retreival(llm_query_parse_dict)
      return comparision_values_for_llm,timeseries_data_1,timeseries_data_2
    elif llm_query_parse_dict['query_task_type']=='compare_period':
      comparision_values_for_llm,timeseries_data_1,timeseries_data_2=numeric_compare_multiperiod_query_retreival(llm_query_parse_dict)
      return comparision_values_for_llm,timeseries_data_1,timeseries_data_2
  ## Route text query type to correct functions for retrieval
  elif llm_query_parse_dict['query_type']=='text':
      if (llm_query_parse_dict['query_task_type']=='summary_fomc_minute') or (llm_query_parse_dict['query_task_type']=='summary_fomc_statement'):
          documents_for_llm_summary=text_document_summary_query_retreival(llm_query_parse_dict)
          return documents_for_llm_summary
      elif llm_query_parse_dict['query_task_type']=='summary_topic':
        documents_for_llm_summary=text_topic_summary_query_retreival(llm_query_parse_dict,user_query)
        return documents_for_llm_summary
      elif llm_query_parse_dict['query_task_type']=='question_answer':
        documents_for_llm_summary=text_question_answer_query_retreival(llm_query_parse_dict,user_query)
        return documents_for_llm_summary


# %%
#### LLM Prompt 1: For generation of response to numeric and single type of query.

## Creating system message and user message for prompt.

def numeric_single_query_response_llm(output_json,api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic analysis domain especially with FRED economic indcator and FOMC to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC and FRED-based macroeconomic queries.
  You will be provided with structured query details and the computed numeric result in JSON format.
  Your task is to generate a clear, human understandable, natural language response to the user's question using only from the provided inputs.

  You must follow below guidelines before generating the response-
  1. Read and understand the user question from <<user_query>> parameter.
  2. Understand the user question context from below parameter values:
	  a. <<query_task_type>>
	  b. <<start_date>>
	  c. <<end_date>>
	  d. <<indicator>>
	  e. <<aggregation_method>>
  3. Read the final computed answer from <<numeric_query_result>>.
  4. Generate a concise and natural response that directly answers the user question. Also, provide a small and simple explanation about the output result in numeric_query_result and explain in very brief what does the output value indicate whether it is moderate or high or low. 
  5. Do not repeat same sentence.
  6. Do not calculate or generate any value on your own.
  7. Use only the result values present in <<numeric_query_result>>.
  8. For better response, you must mention the indicator name and unit in the response.
  9. When required mention the <<aggregation_method>> naturally in the response.
  10. If the numeric result is empty, say that no data was found for the requested query period.
  11. Replace the word percentage with '%' when generating direct response to user query after a single space.

 Note that query and response details is provided in {output_json} and you must return a response.
  """
  # Initialize client with API key
  client = OpenAI(api_key=api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content


# %%
#### LLM Prompt 2: For generation of response to numeric and timeseries type of query.

## Creating system message and user message for prompt.

def numeric_timeseries_query_response_llm(output_json,api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic analysis domain especially with FRED economic indcator and FOMC to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC and FRED-based macroeconomic queries.
  You will be provided with structured query details and the computed numeric result in JSON format.
  Your task is to generate a clear, human understandable, natural language response to the user's question using only from the provided inputs.

  You must follow below guidelines before generating the response-
  1. Read and understand the user question from <<user_query>> parameter.
  2. Understand the user question context from below parameter values:
	  a. <<query_task_type>>
	  b. <<start_date>>
	  c. <<end_date>>
	  d. <<indicator>>
	  e. <<high_value>>
	  f. <<low_value>>
    g. <<average_value>>
    h. <<first_value>>
	  i. <<last_value>>
	  j. <<slope_value>>
	  k. <<trend>>
	  l. <<volatility>>
	  m. <<chart_required>>

  3. Generate a concise and natural response that answers the user question directly and naturally.
  4. Do not calculate or generate any value on your own.
  5. You must generate natural language answers that explains the trend of the data in simple language.
  6. When required mention the <<trend>> and <<volatility>> naturally in the response.
  7. If the result is empty or missing, say that no data was found for the requested query period.
  8. Return plain text only and do not format in bold font. Do not use Markdown formatting like **bold** or bullet points.
  9. A chart will be required to visualise the trend and pattern of both indicators in given time range and mention that a visual comparison supported by separate charts are provided.

  Note that query and response details is provided in {output_json} and you must return a response.
  """
  # Initialize client with API key
  client = OpenAI(api_key=api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content


# %%
#### LLM Prompt 3: For generation of response to numeric and indicator comparision type of query.

## Creating system message and user message for prompt.

def numeric_indicator_compare_query_response_llm(output_json,api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic analysis domain especially with FRED economic indcator and FOMC to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC and FRED-based macroeconomic queries.
  You will be provided with structured query details and the computed numeric result in JSON format.
  Your task is to perform comparison analysis between two indicators and generate a clear, human understandable, natural language comparative analysis response to the user's question using only from the provided structured input data.

  You must follow below guidelines before generating the response-
  1. Read and understand the user question from <<user_query>> parameter.
  2. Understand the user question context from <<type_of_comparision>> parameter.
  3. Read and perform comparison analysis between indicators from below parameter values provided for both the indicators:
    a. <<indicator_label>>
    b. <<high_value>>
    c. <<low_value>>
    d. <<average_value>>
    e. <<first_value>>
    f. <<last_value>>
    g. <<slope_value>>
    h. <<start_date>>
    i. <<end_date>>
    j. <<trend>>
    k. <<volatility>>
  4. Read the value of parameter <<'indicator_higher_rate'>> to understand which indicator have higher rate value from the two use this information when generating a comparison summary.
  5. Parameter <<difference_of_indicators_average>> will tell you the value of rate differences between two indicators and use this information when generating a comparison summary.
  6. You must compare the indicators based on their trend, volatility, average value and slope.
  7. If indicator have significantly increased or decreased, then mention the same by highlighting during the response generation.
  8. A chart will be required to visualise the trend and pattern of both indicators in given time range and mention that a visual comparison supported by separate charts are provided.
  9. Generate a natural language response for explaining the comparison analytically understandable by financial users as well.
  10. Whenever you see any change or movement of values, provide an interpretation of the change in economic terms.
  11. If the result is empty or missing, say that no data was found for the requested query period.
  12. Return plain text only and do not format in bold font. Do not use Markdown formatting like **bold** or bullet points.

  Note that query and response details is provided in {output_json} and you must return a response.

  """
  # Initialize client with API key
  client = OpenAI(api_key=api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content


# %%
#### LLM Prompt 4: For generation of response to numeric and period wise comparision type of query.

## Creating system message and user message for prompt.

def numeric_period_compare_query_response_llm(output_json,api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic analysis domain especially with FRED economic indcator and FOMC to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC and FRED-based macroeconomic queries.
  You will be provided with structured query details and the computed numeric result in JSON format.
  Your task is to perform comparison analysis between two period range data of same indicator and generate a clear, human understandable, natural language comparative analysis response to the user's question using only from the provided structured input data.

  You must follow below guidelines before generating the response-
  1. Read and understand the user question from <<user_query>> parameter.
  2. Understand the user question context from <<type_of_comparision>> parameter.
  3. Read and perform comparison analysis of one indicators between different time range from below parameter values provided for both the period:
    a. <<indicator_label>>
    b. <<high_value>>
    c. <<low_value>>
    d. <<average_value>>
    e. <<first_value>>
    f. <<last_value>>
    g. <<slope_value>>
    h. <<period1_range>>
    i. <<trend>>
    j. <<volatility>>
  4. Read the value of parameter <<'period_having_higher_rate'>> to identify the duration when the indicator have higher rate value from the two diffferent time range period and use this information when generating a comparison summary.
  5. Parameter <<difference_of_rate_average>> will tell you the value of rate differences between two different time period of same indicator and use this information when generating a comparison summary.
  6. You must compare the rate value changes across two different time range of same indicator based on their trend, volatility, average value and slope.
  7. If for a specific time period the rate value have significantly changed, then mention the same by highlighting during the response generation.
  8. A chart will be required to visualise the trend and pattern of both indicators in given time range and mention that a visual comparison supported by separate charts are provided.
  9. Generate a natural language response for explaining the comparison analytically understandable by financial users as well.
  10. Whenever you see any change or movement of values, provide an interpretation of the change in economic terms.
  11. If the result is empty or missing, say that no data was found for the requested query period.
  12. Return plain text only and do not format in bold font. Do not use Markdown formatting like **bold** or bullet points.
  13. You need to give more importance in explaining how the indicator evolved/changed between the two time periods rather than repeating raw numbers.
  Note that query and response details is provided in {output_json} and you must return a response.

  """
  # Initialize client with API key
  client = OpenAI(api_key=api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content


# %%
#### LLM Prompt 5: For generation of response to summary type query based on topic/indicator.

## Creating system message and user message for prompt.

def text_summary_topic_query_response_llm(output_json,api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic and policy analysis of FOMC textual data to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC statements and minutes document textual query.
  You will be provided with structured FOMC text related query details in <<User_query>> and the optimized re-written query in <<Rewritten_query>> to obtain context of the query for generating summary based on indicator.
  You will be provided list of relevant chunks of meeting documents that are most relevant to user query in <<Retrieved_documents>>.
  Your task is to generate a comprehensive topic-based summary from retrieved Federal Open Market Committee (FOMC) meeting documents provided in <<Retrieved_documents>>.

  <<User_query>> : <<original_user_query>> parameter from <<output_json>>

  <<Rewritten_query>> : <<rewritten_user_query>> parameter from <<output_json>>

  <<Retrieved_documents>> : <<documents_retrieved_for_summary>> from <<output_json>>


  You must follow below instructions for generating the response:
  1. Identify the main topic requested by the user by analysing the user query in <<User_query>>.
  2. Review all the ten retrieved document excerpts from <<Retrieved_documents>> and identify statements most relevant to the topic.
  3. Extract below key and important economic insights to be included as part of summary.
    a. risk assessments
    b. meeting discussions
    c. economic indicators
    d. committee member participant's views and opinion
    e. policy stance
  4. If any overlapping insights or ideas across multiple documents then combine the information to produce a coherent explanation of the topic.
  5. Use only information provided in the retrieved documents and do not add any information of your own.
  6. You must not discuss indicators not related to topic or policy themes unless they are necessary to explain or provide context or support the explanation of requested topic.
  7. You must write as a professional and economic analyst.
  8. At the end of summary , you must provide citation of the documents that you used for generating summary.
  9. Follow below format for generating summary of the topic
  10. You must use '%' when refering to percentages and instead of writing the word 'percent' provide '%'.
      Example: 2% instead of 2 percent.

  Note that query and response details is provided in {output_json} and you must return a response.
  Output Format:

  Summary of topic:

  Write a concise analytical summary of 7–9 sentences explaining the topic discussed in FOMC documents.

  Key Insights:

  • Insight 1
  • Insight 2
  • Insight 3
  • Insight 4

  Citation Evidence:

  Mention the meeting dates referenced in the retrieved documents in format yyyy-mm-dd along with the source type as Minutes (M) or Statement(S) based on value provided in 'doc_type' parameter of <<documents_retrieved_for_summary>>.
  You must also provide citation link for each document.
  You must ensure that for FOMC statement documents, the url link is 'https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm'
  and for FOMC minute document url link is 'https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm'
  You must use document type value based on 'doc_type' patrameter of <<documents_retrieved_for_summary>>.
  List all document citations using the STRICT format below:
  Document Date: YYYY-MM-DD, Source: Minute/Statement, https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm or https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm based on document type.
  Document Date: YYYY-MM-DD, Source: Minute/Statement, https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm or https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm based on document type.

  Replace YYYYMMDD in the url link  with values present in document timestamp of document in <<documents_retrieved_for_question_answer>>.

  """
  # Initialize client with API key
  client = OpenAI(api_key=api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content


# %%
#### LLM Prompt 6: For generation of response to summary type query based on FOMC Statement document type.

## Creating system message and user message for prompt.

def text_summary_fomc_statement_query_response_llm(output_json,api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic and policy analysis of FOMC textual data to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC statements only document textual query.
  You will be provided list of relevant chunks of FOMC statement documents for a specific period for generating comprehensive summary.
  Your task is to generate a comprehensive FOMC statement document summary from retrieved Federal Open Market Committee (FOMC) meeting documents provided in <<document_retreived_for_sumamry>>.

  <<document_retreived_for_sumamry>> from <<output_json>>

  You must follow below instructions for generating the summary response:
  1. Read and understand all the relevant documents excerpts from <<document_retreived_for_sumamry>>.
  2. Extract below key and important economic insights to be included as part of summary.
    a. risk assessments
    b. meeting discussions
    c. impact of economic indicators on policy
    d. committee member participant's views and opinion
    e. policy stance
  3. If any overlapping insights or ideas across multiple documents then combine the information to produce a coherent explanation of the topic.
  4. Use only information provided in the retrieved documents and do not add any information of your own.
  5. You must write as a professional and economic analyst.
  6. At the end of summary , you must provide citation of the documents that you used for generating summary.
  7. You must use '%' when refering to percentages and instead of writing the word 'percent' provide '%'.
      Example: 2% instead of 2 percent.

  Note that query type, document type, start date, emd date and relevant documents details are provided in {output_json} and you must return a response.
  <<start_date>> parameter value indicates the start date of FOMC statement.
  <<end_date>> parameter value indicates the end date of FOMC statement.

  Follow below format for generating summary of the topic
  Output Format:

  Summary of FOMC statement from <<start_date>> to <<end_date>>:

  Write a concise analytical summary of 7–9 sentences explaining the topic discussed in FOMC documents but do not exceed more than 200 words. Try to fit the first layer of summary in 200 words.

  Key Insights:

  • Insight 1
  • Insight 2
  • Insight 3
  • Insight 4

  Citation Evidence:

  Mention the meeting dates referenced in the retrieved documents in format yyyy-mm-dd and also provide citation link for each document.
  List all document citations using the STRICT format below:
  Document Date: YYYY-MM-DD , https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm
  Document Date: YYYY-MM-DD , https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm

  Replace YYYYMMDD in the link 'https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm' with values present in document timestamp of document in <<document_retreived_for_sumamry>>.
  """
  # Initialize client with API key
  client = OpenAI(api_key=api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content


# %%
#### LLM Prompt 7: For generation of response to summary type query based on FOMC Minute document type.

## Creating system message and user message for prompt.

def text_summary_fomc_minute_query_response_llm(output_json,api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic and policy analysis of FOMC textual data to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC minutes only document textual query.
  You will be provided list of relevant chunks of FOMC minute documents for a specific period for generating comprehensive summary.
  Your task is to generate a comprehensive FOMC minute document summary from retrieved Federal Open Market Committee (FOMC) meeting documents provided in <<document_retreived_for_sumamry>>.

  <<document_retreived_for_sumamry>> from <<output_json>>

  You must follow below instructions for generating the summary response:
  1. Read and understand all the relevant documents excerpts from <<document_retreived_for_sumamry>>.
  2. Extract below key and important economic insights to be included as part of summary.
    a. risk assessments
    b. meeting discussions
    c. impact of economic indicators on policy
    d. committee member participant's views and opinion
    e. policy stance
  3. If any overlapping insights or ideas across multiple documents then combine the information to produce a coherent explanation of the topic.
  4. Use only information provided in the retrieved documents and do not add any information of your own.
  5. You must write as a professional and economic analyst.
  6. At the end of summary , you must provide citation of the documents that you used for generating summary.
  7. You must use '%' when refering to percentages and instead of writing the word 'percent' provide '%'.
      Example:3% instead of 3 percent.

  Note that query type, document type, start date, emd date and relevant documents details are provided in {output_json} and you must return a response.
  <<start_date>> parameter value indicates the start date of FOMC minute.
  <<end_date>> parameter value indicates the end date of FOMC minute.

  Follow below format for generating summary of the topic
  Output Format:

  Summary of FOMC minute from <<start_date>> to <<end_date>>:

  Write a concise analytical summary of 7 to 9 sentences explaining the topic discussed in FOMC minute documents but do not exceed more than 200 words. Try to fit the first layer of summary in 200 words.

  Key Insights:

  • Insight 1
  • Insight 2
  • Insight 3
  • Insight 4

  Citation Evidence:

  Mention the meeting dates referenced in the retrieved documents in format yyyy-mm-dd and also provide citation link for each document.
  List all document citations using the STRICT format below:
  Document Date: YYYY-MM-DD , https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm
  Document Date: YYYY-MM-DD , https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm

  Replace YYYYMMDD in the link 'https://www.federalreserve.gov/monetarypolicy/fomcminutesYYYYMMDD.htm' with values present in document timestamp of document in <<document_retreived_for_sumamry>>.
  """
  # Initialize client with API key
  client = OpenAI(api_key=api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content


# %%
#### LLM Prompt 8: For generation of response to question answer textual type query from FOMC documents.

## Creating system message and user message for prompt.

def text_question_answer_query_response_llm(output_json,api_key):
  system_message = """ You are an expert financial analysis assistant who is expertise in Macroeconomic and policy analysis of FOMC textual data to generate human understandable response from user query. """

  user_message = f"""
  You are an expert financial analysis assistant for FOMC minutes and statements document textual query.
  You will be provided list of relevant chunks of FOMC documents including both statement and minute type for a specific period.
  Your task is to generate answer to original user query related to FOMC communication documents from retrieved Federal Open Market Committee (FOMC) documents provided in <<documents_retrieved_for_question_answer>>.

  <<documents_retrieved_for_question_answer>> from <<output_json>>

  You must follow below instructions for generating the summary response:
  1. Read and understand all the relevant documents excerpts from <<documents_retrieved_for_question_answer>>.
  2. Read and understand the context of user original query from <<original_user_query>> parameter from <<output_json>>
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

  Note that query type, original user query, meeting date and relevant documents details are provided in {output_json} and you must return a response.

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
  client = OpenAI(api_key=api_key)

  ## Call LLM model GPT-5.4 to generate query parser in JSON response. We arere using chat gpt model GPT-4o Realtime(gpt-4.5-preview).

  llm_response_generation = client.chat.completions.create( model="gpt-5.4",
                  messages=[{"role":"system", "content":system_message},
                             {"role":"user","content":user_message}], temperature=0
                  )
  return llm_response_generation.choices[0].message.content


# %%
### Chart Visualisation Agent :  To return plot of chart when user query demands for the plot of an indicator.

############################ Chart plot for trend of an indicator for timeseries type of query  #########################################

def chart_plot_of_trend_for_timeseries(data,summ):
  ## Set Date column as index.
  data.set_index('Date',inplace=True)
  ## To add trend in graph as liner straight line to understand the rise/fall in rate from 2000.
  x=np.arange(len(data.index))
  y=data['rate_value_%'].values
  coef=np.polyfit(x,y,1)    ## polyfit() function is to find best value of m(slope) and c(intercept) given value of x and y.
  trend=[]
  for i in x:
    trend.append(float(round(coef[0]*i+coef[1],2)))   ## first element in 'coef' is value of m(slope of line) and /
                                                    ##second element in 'coef' is value of c(intercept of linbe).

  ## Plot of chart for diaply on streamlit UI
  fig, ax = plt.subplots(figsize=(5, 3),dpi=80)
  ax.plot(data.index, y, marker='o', linewidth=2.5, label='Actual')
  ax.plot(data.index,trend, linestyle='--', linewidth=2, label='Linear Trend')
  ax.set_title(f"{summ['indicator'][0]} trend/pattern in year {summ['start_date'][:4]}")
  ax.set_xlabel("Date (yyyy-mm-dd)")
  ax.set_ylabel(f"{summ['indicator'][0]} in %")
  if len(data) > 12:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
  else:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
  plt.xticks(rotation=45)
  ax.grid(alpha=0.5)
  ax.legend()
  plt.tight_layout()
  return(fig)

########################### Chart plot for trend of an indicator for comparision type of query ##########################################

def chart_plot_of_trend_for_comparision(data1,data2,summ):
  ## Plot for indicator_1
  ## Set Date column as index.
  data1.set_index('Date',inplace=True)
  ## To add trend in graph as liner straight line to understand the rise/fall in rate from 2000.
  x1=np.arange(len(data1.index))
  y1=data1['rate_value_%'].values
  coef1=np.polyfit(x1,y1,1)    ## polyfit() function is to find best value of m(slope) and c(intercept) given value of x and y.
  trend1=[]
  for i in x1:
    trend1.append(float(round(coef1[0]*i+coef1[1],2)))   ## first element in 'coef' is value of m(slope of line) and /
                                                        ##second element in 'coef' is value of c(intercept of linbe).

  ## Plot of chart for diaply on streamlit UI
  fig1, ax1 = plt.subplots(figsize=(5, 3),dpi=80)
  ax1.plot(data1.index, y1, marker='o', linewidth=2.5, label='Actual')
  ax1.plot(data1.index,trend1, linestyle='--', linewidth=2, label='Linear Trend')
  ax1.set_title(f"{summ['indicator_1']['indicator_label']} trend/pattern in year {summ['indicator_1']['start_date'][:4]}")
  ax1.set_xlabel("Date (yyyy-mm-dd)")
  ax1.set_ylabel(f"{summ['indicator_1']['indicator_label']} in %")
  if len(data1) > 12:
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
  else:
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
  plt.xticks(rotation=45)
  ax1.grid(alpha=0.5)
  ax1.legend()
  plt.tight_layout()

  ## Plot for indicator_2
  ## Set Date column as index.
  data2.set_index('Date',inplace=True)
  ## To add trend in graph as liner straight line to understand the rise/fall in rate from 2000.
  x2=np.arange(len(data2.index))
  y2=data2['rate_value_%'].values
  coef2=np.polyfit(x2,y2,1)    ## polyfit() function is to find best value of m(slope) and c(intercept) given value of x and y.
  trend2=[]
  for i in x2:
    trend2.append(float(round(coef2[0]*i+coef2[1],2)))   ## first element in 'coef' is value of m(slope of line) and /
                                                    ##second element in 'coef' is value of c(intercept of linbe).
  ## Plot of chart for diaply on streamlit UI
  fig2, ax2 = plt.subplots(figsize=(5, 3),dpi=80)
  ax2.plot(data2.index, y2, marker='o', linewidth=2.5, label='Actual')
  ax2.plot(data2.index,trend2, linestyle='--', linewidth=2, label='Linear Trend')
  ax2.set_title(f"{summ['indicator_2']['indicator_label']} trend/pattern in year {summ['indicator_2']['start_date'][:4]}")
  ax2.set_xlabel("Date (yyyy-mm-dd)")
  ax2.set_ylabel(f"{summ['indicator_2']['indicator_label']} in %")
  if len(data2) > 12:
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
  else:
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
  plt.xticks(rotation=45)
  ax2.grid(alpha=0.5)
  ax2.legend()
  plt.tight_layout()
  
  return fig1, fig2

########################### Chart plot for trend of an indicator for comparision between two period ##########################################

def chart_plot_of_trend_for_period_comparision(data1,data2,summ):

  ## Setting font size of title text for all charts.
  plt.rcParams.update({
    "font.size": 8,         
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8 })
  
  ## Plot for period_1
  ## Set Date column as index.
  data1.set_index('Date',inplace=True)
  ## To add trend in graph as liner straight line to understand the rise/fall in rate from 2000.
  x1=np.arange(len(data1.index))
  y1=data1['rate_value_%'].values
  coef1=np.polyfit(x1,y1,1)    ## polyfit() function is to find best value of m(slope) and c(intercept) given value of x and y.
  trend1=[]
  for i in x1:
    trend1.append(float(round(coef1[0]*i+coef1[1],2)))   ## first element in 'coef' is value of m(slope of line) and /
                                                    ##second element in 'coef' is value of c(intercept of linbe).
  
  ## Plot of chart for diaply on streamlit UI
  fig1, ax1 = plt.subplots(figsize=(5, 3),dpi=80)
  ax1.plot(data1.index, y1, marker='o', linewidth=2.5, label='Actual')
  ax1.plot(data1.index,trend1, linestyle='--', linewidth=2, label='Linear Trend')
  ax1.set_title(f"{summ['period_1']['indicator_label'][0]} trend/pattern from {summ['period_1']['period1_range']}")
  ax1.set_xlabel("Date (yyyy-mm-dd)")
  ax1.set_ylabel(f"{summ['period_1']['indicator_label'][0]} in %")
  if len(data2) > 12:
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
  else:
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
  plt.xticks(rotation=45)
  ax1.grid(alpha=0.5)
  ax1.legend()
  plt.tight_layout()
 
  # Plot for period 2
  ## Set Date column as index.
  data2.set_index('Date',inplace=True)
  ## To add trend in graph as liner straight line to understand the rise/fall in rate from 2000.
  x2=np.arange(len(data2.index))
  y2=data2['rate_value_%'].values
  coef2=np.polyfit(x2,y2,1)    ## polyfit() function is to find best value of m(slope) and c(intercept) given value of x and y.
  trend2=[]
  for i in x2:
    trend2.append(float(round(coef2[0]*i+coef2[1],2)))   ## first element in 'coef' is value of m(slope of line) and /
                                                    ##second element in 'coef' is value of c(intercept of linbe).
  
  ## Plot of chart for diaply on streamlit UI
  fig2, ax2 = plt.subplots(figsize=(5, 3),dpi=80)
  ax2.plot(data2.index, y2, marker='o', linewidth=2.5, label='Actual')
  ax2.plot(data2.index,trend2, linestyle='--', linewidth=2, label='Linear Trend')
  ax2.set_title(f"{summ['period_2']['indicator_label'][0]} trend/pattern from {summ['period_2']['period2_range']}")
  ax2.set_xlabel("Date (yyyy-mm-dd)")
  ax2.set_ylabel(f"{summ['period_2']['indicator_label'][0]} in %")
  if len(data2) > 12:
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
  else:
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
  plt.xticks(rotation=45)
  ax2.grid(alpha=0.5)
  ax2.legend()
  plt.tight_layout()

  return fig1, fig2

# %%
## User Query response generator function

def chat_query_response(user_query):
 
  ## Load personal key files from the local path
  with open("OpenAI_API_Key.txt", "r") as f:
    openai_api_key = ' '.join(f.readlines())

  query_parse=query_intent_parser(user_query)
  strt=time.time()
  
  # Initializing return variables
  final_response = None
  fig = None
  fig1 = None
  fig2 = None
  retrieve_generate_time=0

  if query_parse['query_type']=='numeric' and query_parse['query_task_type']=='single':
    summ=query_orchestrator_for_retreival(query_parse,user_query)
    output_json=json.dumps(summ, indent=1)
    final_response=numeric_single_query_response_llm(output_json,openai_api_key)

  elif query_parse['query_type']=='numeric' and query_parse['query_task_type']=='timeseries':
      summ,data=query_orchestrator_for_retreival(query_parse,user_query)
      output_json=json.dumps(summ, indent=1)
      final_response=numeric_timeseries_query_response_llm(output_json,openai_api_key)
      fig=chart_plot_of_trend_for_timeseries(data,summ)

  elif query_parse['query_type']=='numeric' and query_parse['query_task_type']=='compare_indicator':
      summ,data1,data2=query_orchestrator_for_retreival(query_parse,user_query)
      output_json=json.dumps(summ, indent=1)
      final_response=numeric_indicator_compare_query_response_llm(output_json,openai_api_key)
      fig1,fig2=chart_plot_of_trend_for_comparision(data1,data2,summ)

  elif query_parse['query_type']=='numeric' and query_parse['query_task_type']=='compare_period':
      summ,data1,data2=query_orchestrator_for_retreival(query_parse,user_query)
      output_json=json.dumps(summ, indent=1)
      final_response=numeric_period_compare_query_response_llm(output_json,openai_api_key)
      fig1,fig2=chart_plot_of_trend_for_period_comparision(data1,data2,summ)

  elif query_parse['query_type']=='text' and query_parse['query_task_type']=='summary_topic':
      text_data=query_orchestrator_for_retreival(query_parse,user_query)
      output_json=json.dumps(text_data, indent=1)
      final_response=text_summary_topic_query_response_llm(output_json,openai_api_key)

  elif query_parse['query_type']=='text' and query_parse['query_task_type']=='summary_fomc_statement':
      text_data=query_orchestrator_for_retreival(query_parse,user_query)
      output_json=json.dumps(text_data, indent=1)
      final_response=text_summary_fomc_statement_query_response_llm(output_json,openai_api_key)

  elif query_parse['query_type']=='text' and query_parse['query_task_type']=='summary_fomc_minute':
      text_data=query_orchestrator_for_retreival(query_parse,user_query)
      output_json=json.dumps(text_data, indent=1)
      final_response=text_summary_fomc_minute_query_response_llm(output_json,openai_api_key)

  elif query_parse['query_type']=='text' and query_parse['query_task_type']=='question_answer':
      text_data=query_orchestrator_for_retreival(query_parse,user_query)
      output_json=json.dumps(text_data, indent=1)
      final_response=text_question_answer_query_response_llm(output_json,openai_api_key)

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
