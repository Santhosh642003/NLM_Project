import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
model.eval()  

def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        answer_tokens = all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1]
        answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    return answer

def render_qa_page():
    st.title("Question Answering App")
    st.write("Enter context and ask your questions:")
    context = st.text_area("Context:", height=100)
    question = st.text_input("Enter question:", key="question")

    initial_answer = get_answer("", context)


    answer = ""  
    if context and question:
        answer = get_answer(question, context)
    st.write(f"Answer: {answer}")