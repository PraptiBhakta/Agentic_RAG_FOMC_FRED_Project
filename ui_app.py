## import python code of agentic rag pipeline.

import streamlit as st
from agentic_rag_pipeline_vscode import chat_query_response
# from traditional_rag_pipeline_vsccode import chat_query_response

# Page Layout and Format

st.markdown("""
<style>

/* Main page wrapper */
.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* Sub-heading style */
.sub-heading {
    font-size: 18px;
    color: #4f5b66;
    margin-bottom: 2rem;
}

/* Format of User message box */
.user-box {
    background-color: #f7f9fc;
    border-left: 5px solid #2c3e50;
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 14px;
    color: #111111;
    font-size: 17px;
    line-height: 1.6;
}

/* Format of Assistant message box */
.assistant-box {
    background-color: #f4fbf6;
    border-left: 5px solid #1b8a5a;
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 18px;
    color: #1a1a1a;
    font-size: 17px;
    line-height: 1.7;
}

/* Format of User role type box */
.role-user {
    font-weight: 700;
    color: #1f2d3d;
    margin-bottom: 6px;
    font-size: 18px;
}

/* Format of Assistant role type box */
.role-assistant {
    font-weight: 700;
    color: #1b8a5a;
    margin-bottom: 6px;
    font-size: 18px;
}

/* Format for citation area */
.citation-box {
    background-color: #fffdf5;
    border-left: 4px solid #d4a017;
    padding: 10px 14px;
    border-radius: 8px;
    margin-top: 10px;
    font-size: 15px;
    color: #333333;
}

/* Input user query box spacing */
.stChatInput {
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Format of User and AI Assistant message response and charts.

def render_message(message):
    role = message["role"]
    content = message["content"]
    if role == "user":
        st.markdown(f"""
        <div class="user-box">
            <div class="role-user">User</div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-box">
            <div class="role-assistant">AI Assistant</div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)

        # Obtain charts from history
        if message.get("chart") is not None:
            st.pyplot(message["chart"],width="content")

        if message.get("chart1") is not None:
            st.pyplot(message["chart1"],width="content")

        if message.get("chart2") is not None:
            st.pyplot(message["chart2"],width='content')

        if message.get("time_taken"):
            st.markdown(f"**Total time taken:** {message['time_taken']}")

# Main heading of the page

st.markdown("""
<h1 style='font-size:30px; font-weight:700; color:#1f2d3d; margin-bottom:10px;'>
FOMC Assistant Chatbot
</h1>
""", unsafe_allow_html=True)

# Sub heading of the page
st.markdown(
    '<div class="sub-heading">An interactive AI chatbot for assisting in analysis of FOMC policy communications and FRED macroeconomic indicators.</div>',
    unsafe_allow_html=True
)

#  Session State and default AI assistant message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
        "role": "assistant",
        "content": "Hello. Please enter your question related to FOMC policy or FRED macroeconomic indicators.",
        "chart": None,
        "chart1": None,
        "chart2": None,
        "time_taken": None
        }
    ]

# Chat History details
for message in st.session_state.messages:
    render_message(message)

# Initialize state of pending query to none.
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Input Chat box
user_query = st.chat_input("Enter your question here...")

# Save user query and then rerun.
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.pending_query = user_query
    st.rerun()

# Process the pending query after rerun.
if st.session_state.pending_query is not None:

    query_for_processing = query_to_process = st.session_state.pending_query

    # Call main agentic rag pipeline to get response of user query.
    with st.spinner("Generating response..."):
        response = chat_query_response(query_for_processing)

    # Extract values from response obtained from pipeline
    response_text = response.get("response", "")
    fig_data = response.get("chart", None)
    fig_data1 = response.get("chart1", None)
    fig_data2 = response.get("chart2", None)
    total_time=response.get("time_taken","")

    # Store history of response and charts as well.
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "chart": fig_data,
        "chart1": fig_data1,
        "chart2": fig_data2,
        "time_taken": total_time
    })

    ## Set pendiung query back to null
    st.session_state.pending_query = None
    st.rerun()

