import streamlit as st
from assistant_vector import create_assistant


assistant = create_assistant()

st.title("Barista AI - Coffee Knowledge Assistant")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Processing..."):
        answer = assistant.rag(user_input).output_text
        st.success("Completed!")
        st.write(answer)