import streamlit as st
from assistant_vector import create_assistant

# 1. Page Configuration
st.set_page_config(
    page_title="BeanRAG",
    page_icon="☕",
    layout="centered"
)

# 2. Cache Assistant Initialization
@st.cache_resource
def load_assistant():
    return create_assistant()

assistant = load_assistant()

# 3. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.title("☕ BeanRAG Settings")
    st.markdown("Your expert RAG assistant for coffee beans, brewing ratios, and techniques.")
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Title Header
st.title("☕ Coffee Bean Knowledge Base")

# 4. Render Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If sources were saved in history, render them in an expander
        if "sources" in message and message["sources"]:
            with st.expander("View Retrieved Context"):
                st.write(message["sources"])

# 5. Process User Input
if prompt := st.chat_input("Ask about coffee recipes, grind sizes, or bean origins..."):
    
    # Display & Save User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                # Call RAG pipeline
                rag_result = assistant.rag(prompt)
                
                # Extract text and optional sources/retrieved docs
                response_text = rag_result.output_text
                retrieved_sources = getattr(rag_result, "sources", None) or getattr(rag_result, "context", None)

                # Render Response
                st.markdown(response_text)
                
                # Render Sources if available in the output object
                if retrieved_sources:
                    with st.expander("View Retrieved Context"):
                        st.write(retrieved_sources)

                # Save Assistant Message to State
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": retrieved_sources
                })

            except Exception as e:
                st.error(f"Failed to generate response: {str(e)}")