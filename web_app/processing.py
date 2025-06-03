import os
import re
import whisper
import spacy
import librosa
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel, PeftConfig
import torch
from huggingface_hub import login
import pandas as pd
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv

#------PREPROCESSING-------
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("tiny", device=device)
nlp = spacy.load("en_core_web_sm")
model_base = pipeline("summarization", model="t5-small",tokenizer="t5-small")

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_named_entities(text):
    doc = nlp(text)
    entities_by_label = {}
    for ent in doc.ents:
        label = ent.label_
        name = ent.text
        entities_by_label.setdefault(label, set()).add(name)
    return {
        label: ', '.join(sorted(names))
        for label, names in entities_by_label.items()
    }


#------FINE TUNED T5 MODEL-------

def clean_summary(text, add_paragraphs=False):
    if not text:
        return ""
    
    # ensure proper ending punctuation
    if not text.endswith(('.', '!', '?')):
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        last_comma = text.rfind(',')
        
        if last_period > 0 and last_period > last_comma:
            text = text[:last_period+1]
        elif last_comma > 0:
            text = text[:last_comma] + "."
        else:
            text = text.rstrip() + "."
    
    # add pargraph breaks for long summary
    # since the data used for fine-tuning model is mostly 3 sentences for each paragraph, so we break the paragraph for every 3 sentences.
    if add_paragraphs:
        abbreviations = [r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\."]
        protected = text
        for abbr in abbreviations:
            protected = re.sub(abbr, lambda m: m.group(0).replace(".", "<DOT>"),protected)

        parts = re.split(r'(?<=[.!?])\s+', protected)

        sentences = [p.replace("<DOT>", ".") for p in parts]
        
        if len(sentences) >= 4:  
            paragraphs = []
            current_para = []
            
            for i, sentence in enumerate(sentences):
                current_para.append(sentence)
                
                if (i + 1) % 3 == 0 and i < len(sentences) - 1:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            
            if current_para:
                paragraphs.append(' '.join(current_para))
            
            text = '\n\n'.join(paragraphs)
    
    return text

# load the fine-tuned weights
def get_summarizer(style):
    length_map = {
        "tiny": (10, 40),
        "short": (50, 150),
        "long": (150, 400)
    }

    min_len, max_len = length_map[style]

    adapter_path = os.path.join("LoRA_Weights", f"lora_adapters_{style}")

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)

    summarizer = pipeline("summarization", model = model_with_lora, tokenizer = tokenizer)
    return summarizer

def summarize_with_lora(text, style = "short"):
    summarizer = get_summarizer(style)
    prompt = ("summarize: " + text.strip())[:1024]
    raw = summarizer(prompt)[0]["summary_text"]

    add_paragraphs = (style == "long")
    summary = clean_summary(raw, add_paragraphs)

    return summary


#------BASE T5 MODEL-------

def summarize_with_base(text, style):
    style_map = {
        "tiny": {
            "max_len": 50, "min_len": 30,
            "do_sample": False, "temperature": 0.4, "top_k": 50, "top_p": 0.95
        },
        "short": {
            "max_len": 100, "min_len": 50,
            "do_sample": False, "temperature": 0.4, "top_k": 50, "top_p": 0.95
        },
        "long": {
            "max_len": 200, "min_len": 100,
            "do_sample": False, "temperature": 0.4, "top_k": 50, "top_p": 0.95
        },
    }

    params = style_map.get(style, style_map["short"])

    try:
        prompt = "summarize: " + text.strip()

        summary = model_base(
            prompt,
            max_length=params["max_len"],
            min_length=params["min_len"],
            do_sample=params["do_sample"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
        )

        return summary[0]['summary_text']

    except Exception as e:
        print(f"[ERROR] Base summarization failed: {e}")
        return "Error during summarization."


#------LLAMA 3 AGENTS WITH ICL-------

# load llama 3
load_dotenv('../.env')
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  
tokenizer_llama = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=torch.float16, 
                                             device_map="auto")

# summarizer prompt
def get_long_summary_prompt(text):
    return f"""<|system|>
You are a helpful AI assistant specialized in text summarization. 
<|user|>
I need you to summarize the following text into three to five paragraphs (150–400 words). Return only the summary. Do not include any introductory or concluding sentences, explanations, or conversational statements. Your output must be the summary only.

Example text: long ago they lived in merchant who had three daughters every year at a certain day of a certain month he went away to a distant city to collect money when an account how do you know astor father i am older and wiser than you are and i know that there are many evils which might come upon you when it was evening he led his band into a nearby street and in his disguise approached the merchants house he knocked at the door have pity upon a poor unfortunate one he called out let me enter i pray you to pass the night under your roof its surely a terrible storm outside so the merchants eldest daughter as the wind rattled the tiles of the roof and the rain beat in torrents against the doors and windows he is old as well as poor she said if we decide to show mercy to this poor beggar it is not for you to oppose it but we should not forget our promise to our father cried the youngest daughter however in spite of all she could say the elder sisters opened the door and admitted the bigger it is a fearful night to send away a bigger said the eldest sister while they were eating while they were talking the beggar had taken the apples which the girls were to eat for dessert and had sprinkled a sleeping powder over them the two eldest ate their apples but the youngest could not eat that night she threw the apple away she did not stir and he knew that the sleeping powder had thoroughly done its work then she hurt him go down the stairway and unbolt the heavy doors which led into the store it was the youngest one who deceived me cried the robert chieftain perhaps we can outlet her yet cry another the merchants daughter at first did not answer but as he kept on calling to her she finally asked him what it was that he wanted i promise you i will do you no harm you shall not come into my fathers house pass the charm out to me then said the robber when she returned his hand was sticking through the hole in the door the cries and curses of the robbers filled the air they tried and vane to break down the great doors all my worries about you were foolish

<|assistant|>
A merchant with three daughters traveled each year to a distant city to collect money. One evening during a terrible storm, while the merchant was away, a beggar came to their house asking for shelter. The eldest daughters, moved by his plea, insisted on letting him in despite their youngest sister’s warning that they should honor their father’s instructions.

During dinner, the beggar secretly sprinkled sleeping powder on the apples meant for dessert. The two elder sisters ate the apples and fell asleep, but the youngest, suspicious, did not eat hers. Awake and alert, she heard the beggar—revealed to be a robber—go downstairs to unbolt the doors leading to the merchant’s store. Realizing her plan was foiled, the robber chief became furious, and his gang tried in vain to break into the house.

Despite their efforts and curses, the youngest daughter’s caution and bravery protected the household from harm. Her wisdom proved greater than her sisters’ kindness, showing that mercy should be tempered with vigilance, especially when danger is near.

<|user|>
Now please summarize this text into two to four paragraphs:

{text}

<|assistant|>
"""

def get_short_summary_prompt(text):
    return f"""<|system|>
You are a helpful AI assistant specialized in text summarization.
<|user|>
I need you to summarize the following text into a single concise paragraph (100–150 words). Return only the summary. Do not include any introductory or concluding sentences, explanations, or conversational statements. Your output must be the summary only.

Example text: long ago they lived in merchant who had three daughters every year at a certain day of a certain month he went away to a distant city to collect money when an account how do you know astor father i am older and wiser than you are and i know that there are many evils which might come upon you when it was evening he led his band into a nearby street and in his disguise approached the merchants house he knocked at the door have pity upon a poor unfortunate one he called out let me enter i pray you to pass the night under your roof its surely a terrible storm outside so the merchants eldest daughter as the wind rattled the tiles of the roof and the rain beat in torrents against the doors and windows he is old as well as poor she said if we decide to show mercy to this poor beggar it is not for you to oppose it but we should not forget our promise to our father cried the youngest daughter however in spite of all she could say the elder sisters opened the door and admitted the bigger it is a fearful night to send away a bigger said the eldest sister while they were eating while they were talking the beggar had taken the apples which the girls were to eat for dessert and had sprinkled a sleeping powder over them the two eldest ate their apples but the youngest could not eat that night she threw the apple away she did not stir and he knew that the sleeping powder had thoroughly done its work then she hurt him go down the stairway and unbolt the heavy doors which led into the store it was the youngest one who deceived me cried the robert chieftain perhaps we can outlet her yet cry another the merchants daughter at first did not answer but as he kept on calling to her she finally asked him what it was that he wanted i promise you i will do you no harm you shall not come into my fathers house pass the charm out to me then said the robber when she returned his hand was sticking through the hole in the door the cries and curses of the robbers filled the air they tried and vane to break down the great doors all my worries about you were foolish

<|assistant|>
While their merchant father was away, three sisters sheltered a beggar during a fierce storm, despite the youngest’s warning to keep their promise to their father. The beggar, actually a robber in disguise, drugged the elder sisters with sleeping powder, but the cautious youngest sister avoided the trap. Staying awake, she overheard the robber’s plan to break into the house and foiled his attempt, saving her family through her vigilance and quick thinking.

<|user|>
Now please summarize this text into a single concise paragraph:

{text}

<|assistant|>
"""

def get_tiny_summary_prompt(text):
    return f"""<|system|>
You are a helpful AI assistant specialized in text summarization.
<|user|>
I need you to summarize the following text into a single sentence (15–40 words). Return only the summary. Do not include any introductory or concluding sentences, explanations, or conversational statements. Your output must be the summary only.

Example text: long ago they lived in merchant who had three daughters every year at a certain day of a certain month he went away to a distant city to collect money when an account how do you know astor father i am older and wiser than you are and i know that there are many evils which might come upon you when it was evening he led his band into a nearby street and in his disguise approached the merchants house he knocked at the door have pity upon a poor unfortunate one he called out let me enter i pray you to pass the night under your roof its surely a terrible storm outside so the merchants eldest daughter as the wind rattled the tiles of the roof and the rain beat in torrents against the doors and windows he is old as well as poor she said if we decide to show mercy to this poor beggar it is not for you to oppose it but we should not forget our promise to our father cried the youngest daughter however in spite of all she could say the elder sisters opened the door and admitted the bigger it is a fearful night to send away a bigger said the eldest sister while they were eating while they were talking the beggar had taken the apples which the girls were to eat for dessert and had sprinkled a sleeping powder over them the two eldest ate their apples but the youngest could not eat that night she threw the apple away she did not stir and he knew that the sleeping powder had thoroughly done its work then she hurt him go down the stairway and unbolt the heavy doors which led into the store it was the youngest one who deceived me cried the robert chieftain perhaps we can outlet her yet cry another the merchants daughter at first did not answer but as he kept on calling to her she finally asked him what it was that he wanted i promise you i will do you no harm you shall not come into my fathers house pass the charm out to me then said the robber when she returned his hand was sticking through the hole in the door the cries and curses of the robbers filled the air they tried and vane to break down the great doors all my worries about you were foolish

<|assistant|>
When a beggar-turned-robber tricks two sisters into eating sleeping powder, the youngest sister’s caution thwarts his plan and protects her family.

<|user|>
Now please summarize this text into a single sentence:

{text}

<|assistant|>
"""

# evaluator prompt
def build_evaluator_prompt(original_text, summary, summary_type):
    return f"""<|system|>
You are a critical reviewer of summaries. You read an original text, a summary, and decide if the summary is good or not, following the provided examples. Output your decision and feedback using the format shown in the examples.

<|user|>
Example 1:
Original text:
The cat sat on the mat. The dog barked at the mailman.

Summary:
A cat sat on a mat while a dog barked at a mailman.

Evaluation:
- Result: Pass
- Feedback: The summary covers all key details accurately and concisely.

Example 2:
Original text:
Alice goes to Wonderland and meets many strange creatures.

Summary:
Alice meets some people.

Evaluation:
- Result: Fail
- Feedback: The summary is too vague and omits many important plot points.

Now, evaluate the following summary:

Original text:
{original_text}

Summary:
{summary}

Return your response in this format:
- Result: <Pass or Fail>
- Feedback: <Your feedback>
<|assistant|>
"""

# regenerator prompt
def get_regenerator_prompt(text, last_summary, feedback, summary_type):
    if summary_type == "long":
        prompt = f"""<|system|>
You are a helpful AI assistant specialized in text summarization.
<|user|>
Please summarize the original text into three to five paragraphs (150–400 words), addressing the evaluator's feedback of the last summary below. Return only the summary. Do not include any introductory or concluding sentences, explanations, or conversational statements. Your output must be the summary only.

Original text:
{text}

Last summary:
{last_summary}

Evaluator feedback of last summary:
{feedback}
<|assistant|>
"""
    elif summary_type == "short":
        prompt = f"""<|system|>
You are a helpful AI assistant specialized in text summarization.
<|user|>
Please summarize the original text in a single paragraph (100–150 words), addressing the evaluator's feedback of the last summary below. Return only the summary. Do not include any introductory or concluding sentences, explanations, or conversational statements. Your output must be the summary only.

Original text:
{text}

Last summary:
{last_summary}

Evaluator feedback of last summary:
{feedback}
<|assistant|>
"""
    elif summary_type == "tiny":
        prompt = f"""<|system|>
You are a helpful AI assistant specialized in text summarization.
<|user|>
Please summarize the original text in a single sentence (15–40 words), addressing the evaluator's feedback of the last summary below. Return only the summary. Do not include any introductory or concluding sentences, explanations, or conversational statements. Your output must be the summary only.

Original text:
{text}

Last summary:
{last_summary}

Evaluator feedback of last summary:
{feedback}
<|assistant|>
"""
    else:
        raise ValueError("Invalid summary_type.")
    return prompt


def llm(prompt, max_new_tokens=512, temperature=0.4):
    inputs = tokenizer_llama(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer_llama.eos_token_id
        )
    decoded = tokenizer_llama.decode(outputs[0], skip_special_tokens=True)
    if prompt in decoded:
        return decoded[len(prompt):].strip()
    return decoded.strip()

# define agents
def summarizer_node(state):
    summary_type = state["summary_type"]
    text = state["text"]
    if summary_type == "long":
        prompt = get_long_summary_prompt(text)
    elif summary_type == "short":
        prompt = get_short_summary_prompt(text)
    elif summary_type == "tiny":
        prompt = get_tiny_summary_prompt(text)
    else:
        raise ValueError("Invalid summary_type.")
    state["summary"] = llm(prompt)
    return state

def evaluator_node(state):
    eval_prompt = build_evaluator_prompt(state["text"], state["summary"], state["summary_type"])
    eval_result = llm(eval_prompt)
    state["eval_result"] = eval_result
    state["pass"] = "pass" in eval_result.lower()
    return state

def regenerator_node(state):
    state["attempts"] = state.get("attempts", 0) + 1
    feedback = state["eval_result"]
    summary_type = state["summary_type"]
    text = state["text"]
    last_summary = state["summary"]

    prompt = get_regenerator_prompt(text, last_summary, feedback, summary_type)
    state["summary"] = llm(prompt)
    return state

def router(state):
    if state.get("pass", False) or state["attempts"] >= 3:
        return END
    else:
        print('Finished trial ' + str(state["attempts"]+1))
        return "regenerator"
    
# create graph
class SummaryState(TypedDict, total=False):
    text: str
    summary_type: str
    summary: str
    eval_result: str
    pass_: bool
    attempts: int

graph = StateGraph(state_schema=SummaryState)
graph.add_node("summarizer", summarizer_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("regenerator", regenerator_node)
graph.add_edge("summarizer", "evaluator")
graph.add_conditional_edges("evaluator", router)
graph.add_edge("regenerator", "evaluator")
graph.set_entry_point("summarizer")
langgraph_pipeline = graph.compile()

# clean generated text
def clean_summary2(text):
    lines = text.strip().split('\n')
    cleaned_lines = []
    stop_markers = re.compile(r'^\s*(<\|system\|>|<\|user\|>|<\|assistant\|>|system|user|assistant|Generated text)', re.IGNORECASE)
    skip_markers = re.compile(r'^\s*(here|improved|revised|rewritten|summary)[^:]*:', re.IGNORECASE)

    for line in lines:
        if stop_markers.match(line):
            break  # stop collecting lines
        if skip_markers.match(line):
            continue  # skip this line
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
    return '\n\n'.join(cleaned_lines)


def summarize_with_agent_icl(text, style):
    state = {
        "text": text,
        "summary_type": style,
        "attempts": 0
    }
    result = langgraph_pipeline.invoke(state)
    if not result.get("pass", False) and result["attempts"] >= 3:
        return "Can't generate summary. Please try again."
    else:
        return clean_summary2(result["summary"])
