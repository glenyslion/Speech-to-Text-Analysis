# Speech to Text Analysis

This project turns speech audio into clean and summarized text using modern NLP models.  
We focused mostly on **summarization** and improved the base T5 model through fine-tuning and agent-based methods.

---

## What We Did

### 1. Speech to Text  
- Used **Whisper** to transcribe audio clips from the **LibriSpeech** dataset  
- Whisper gives high-quality transcriptions that match well with the original text

### 2. EDA (Exploratory Data Analysis)  
- Checked audio durations and token lengths  
- Most clips are short (2–7 seconds), which works well for summarization  
- Whisper transcriptions are clean and consistent

### 3. NER (Named Entity Recognition)  
- Applied Spacy’s `en_core_web_trf` model  
- Found entities in almost all transcriptions (over 98% coverage)

### 4. Summarization

### T5 Base Model (Pretrained)
- Tried the T5-large model first  
- Output summaries were repetitive and not very informative

### T5 Fine-Tuned with LoRA
- Used **LLaMA** to generate better summaries
- Fine-tuned **T5-small** using LoRA on those summaries
- Trained 3 models for Tiny, Small, and Large summaries
- Got over **50% improvement** on ROUGE and BERTScore

### LLaMA Multi-Agent System
- Built a 3-agent system: **Summarizer → Evaluator → Regenerator**
- Used prompt engineering and in-context learning
- Best performance across fluency, coverage, coherence, and faithfulness
- Evaluated using both BERTScore and LLM-as-a-judge

---

## Web App

We created a Streamlit web app where users can:
- Upload audio
- View transcription
- Run NER
- Generate different types of summaries (T5 / LLaMA-based)

> **Note:** The web app can only be run on the **Deep Dish server** because it uses multi-agent summarization with **LLaMA-3**, which requires significant computational resources and a locally hosted model.

## How to Run Web App (on Deep Dish Server)

1. **Open Deep Dish server**

2. **Download the required files:**
   - Download both Python files from the `web_app/` folder
   - Download the `LoRA_Weights/` folder (needed for PEFT)

3. **Navigate to the folder**  
   ```bash
   cd web_app
   ```

4. Run the Streamlit web app (This only works on Deep Dish):
    ```bash
    streamlit run webapp.py
    ```

---


## Project Structure
```text
├── data/                  # Audio transcripts and intermediate CSV files
├── LoRA_Weights/          # Fine-tuned model weights using LoRA
├── T5_Fine_Tuning_Code/   # Code for LoRA-based fine-tuning of T5
├── web_app/               # Streamlit web app code
├── README.md
└── Text Analytics Project Workbook.ipynb   # End-to-end notebook (Whisper transcription, EDA, 
                                            NER, Summarization, Evaluation)
```

---

## Team Members

- Fuqian Zou  
- Glenys Charity Lion
- Iris Lee  
- Kavya Bhat  
- Liana Bergman-Turnbull

---

## Notes

- Dataset: [LibriSpeech dev-clean](http://www.openslr.org/12)  
- Models used: Whisper, T5, LLaMA  
- Fine-tuning with LoRA (Parameter-Efficient)

---