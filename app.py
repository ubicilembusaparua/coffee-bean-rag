import streamlit as st
from assistant import create_assistant

# Page Configuration
st.set_page_config(
    page_title="Coffee Knowledge Assistant",
    page_icon="☕",
    layout="centered"
)

# Initialize Backend
if "assistant" not in st.session_state:
    st.session_state.assistant = create_assistant()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and Layout
st.title("☕ Coffee Knowledge Assistant")
st.caption("A RAG-powered assistant for coffee beans, brewing, and techniques.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input via Chat Box
if prompt := st.chat_input("Ask a question about coffee..."):
    # Render user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                response = st.session_state.assistant.rag(
                    prompt,
                    history=st.session_state.messages
                ).output_text
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)