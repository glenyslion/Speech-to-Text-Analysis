import streamlit as st
import tempfile
import os
from processing import (
    transcribe_audio,
    clean_text,
    extract_named_entities,
    clean_summary,
    get_summarizer,
    summarize_with_lora,
    summarize_with_base,
    summarize_with_agent_icl
)

st.set_page_config(page_title="Audio Summarizer", layout="centered")
st.title("Audio Analysis")
st.write("Upload an audio file to get its transcript, named entities, and a summary.")

# initialize session state variables
for key in ["transcript", "entities", "summary_result", "tmp_path"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Upload block
uploaded_file = st.file_uploader("Upload Audio File (.mp3, .wav, .flac)", type=["mp3", "wav", "flac"])

if uploaded_file:
    # Save uploaded file to a persistent temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Transcribe"):
        try:
            transcript = transcribe_audio(st.session_state.tmp_path)
            st.session_state.transcript = transcript
            st.session_state.entities = extract_named_entities(transcript)
            st.success("Transcription completed successfully.")
        except Exception as e:
            st.error(f"Transcription error: {e}")

# transcript and NER
if st.session_state.transcript:
    st.subheader("Transcript")
    st.write(st.session_state.transcript)

    st.subheader("Named Entities")
    if st.session_state.entities:
        for ent_label, ent_texts in st.session_state.entities.items():
            ent_list = ent_texts.split(", ")
            st.markdown(f"**{ent_label}**:")
            for ent in ent_list:
                st.markdown(f"- {ent}")

    else:
        st.info("No named entities found.")

    # summary generation
    st.subheader("Summary")
    with st.form("summary_form"):
       
        model_choice = st.selectbox("Choose summarization model:", ["T5 w/o LoRA finetune", "T5 with LoRA finetune", "Llama 3"])
        summary_type = st.selectbox("Choose summary length:", ["tiny", "short", "long"])
        submit_summary = st.form_submit_button("Generate Summary")

    if submit_summary:
        try:
            cleaned = clean_summary(st.session_state.transcript)
            if model_choice == "T5 with LoRA finetune":
                generated_summary = summarize_with_lora(cleaned, style=summary_type)
            elif model_choice == "T5 w/o LoRA finetune":
                generated_summary = summarize_with_base(cleaned, style=summary_type)
            else:
                generated_summary = summarize_with_agent_icl(st.session_state.transcript, style=summary_type)
            st.session_state.summary_result = generated_summary
        except Exception as e:
            st.warning(f"Summary generation failed: {e}")
            st.session_state.summary_result = None

# Summary output
if st.session_state.summary_result:
    st.markdown("### Summary")
    st.write(st.session_state.summary_result)