#####################################################################################################
"""
Project: CLiTSSA
inference_utils.py: Script to implement several methods to be used in inference main file.
"""
######################################################################################################

## imports
import sys,os
import pandas as pd
import json
import tqdm
import csv
import numpy as np
import ast
import torch
import argparse
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# Defining Prompt Templates for Low-Resource Languages
def get_cross_lingual_prompt_template(task,language):
    prompt_head =""
    example_prompt =""
    query_prompt =""
    prompt_tail =""

    if task == "L1":
        example_prompt = f"Question: [QUESTION] Answer both the month and year.\nAnswer: The correct answer in month,year is [ANSWER]\n\n"
    elif task == "L2":
        example_prompt = f"Question: [QUESTION]\nAnswer: The correct answer is "
    elif task == "L3":
        example_prompt = f"Question: [QUESTION]\nAnswer: The correct answer is [ANSWER]\n\n"
    else:
        print("select a a valid temporal Task: L1, L2 or L3")

    if language == "French":
        prompt_head = "Considérez les exemples anglais suivants: \n"
        if task == "L1":
            query_prompt = f"Question: [QUESTION] Répondez au mois et à l'année.\nRéponse: La bonne réponse en mois,année est "
        else:
            query_prompt = f"Question: [QUESTION]\nRéponse: La bonne réponse est"
        prompt_tail = f"Maintenant, veuillez répondre à la question suivante en français. La réponse doit être en français:\n\n"

    elif language == "German":
        prompt_head = f"Betrachten Sie die folgenden englischen Beispiele: \n"
        if task == "L1":
            query_prompt = f"Frage: [QUESTION] Antworte Monat und Jahr.\nAntwort: Die richtige Antwort in Monat, Jahr ist "
        else:
            query_prompt = f"Frage: [QUESTION]\nAntwort: Die richtige Antwort ist"
        prompt_tail = f"Beantworten Sie nun bitte die nachfolgende Anfrage auf Deutsch. Die Antwort muss auf Deutsch erfolgen:\n\n"

    elif language == "Romanian":
        prompt_head = f"Luați în considerare următoarele exemple în limba engleză: \n"
        if task == "L1":
            query_prompt = f"Întrebare: [QUESTION] Răspuns luna și an.\nRăspuns: Răspunsul corect în lună, an este "
        else:
            query_prompt = f"Întrebare: [QUESTION]\nRăspuns: Răspunsul corect este"
        prompt_tail = f"Acum, vă rugăm să răspundeți la întrebarea ulterioară în limba română. Răspunsul trebuie să fie în limba română:\n\n"

    else:
        print("select a a valid low-resource language: Romanian, German or French")

    return  prompt_head, example_prompt,query_prompt,prompt_tail

# Method to generate final prompt
def build_prompt(examples_questions, examples_answers, shots, query,exampleids,task,prompt_head, example_prompt,query_prompt,prompt_tail):
    
    prompt = prompt_head
    examples_idx = exampleids
    
    for idx in examples_idx:
        if task == "L1":
            example_question = examples_questions[idx]
            if example_question[-1] != '?':
                example_question += '?'
                prompt += example_prompt.replace('[QUESTION]',example_question).replace('[ANSWER]',examples_answers[idx])
        elif task == "L2":
            prompt += example_prompt.replace('[QUESTION]',examples_questions[idx])
            answer_entities = examples_answers[idx]
            for answer_id,answer_entity in enumerate(answer_entities):
                prompt += answer_entity
                if (answer_id + 1) != len(answer_entities):
                    if (answer_id + 2) != len(answer_entities):
                        prompt += ", "
                    else:
                        prompt += " et "
            prompt += "\n\n"
        elif task == "L3":
            prompt += example_prompt.replace('[QUESTION]',examples_questions[idx]).replace('[ANSWER]',examples_answers[idx])

    prompt += prompt_tail

    if task == "L1":
        if query[-1] != '?':
            query += '?'

    prompt += query_prompt.replace('[QUESTION]',query)
    return prompt

# Method to load mTEMPREASON dataset
def load_datafile(data_file_path):
    h = open(data_file_path, 'r')
    queries = []
    responses = []
    for line in h:
        row = json.loads(line)
        queries.append(row['question'].rstrip())
        responses.append(row['text_answers']['text'][0].rstrip())
    h.close()
    return queries,responses

# loading semantic indexes if experiment choice is 'semantic'
def load_indexfile(semantic_index_file, exp):
    ex_idx = {}
    if exp =="semantic":
        with open(semantic_index_file) as idxfile:
            idxreader = csv.reader(idxfile, delimiter=',')
            next(idxreader)
            for row in idxreader:
                ex_idx[int(row[0])]= ast.literal_eval(row[1])
    return ex_idx

def get_generation_config(task):
    if task=="L3":
        num_beams = 2
    else:
        num_beams = 4
    # Generation Config for LLM
    gen_config = GenerationConfig(
        num_beams=num_beams,
        max_new_tokens=30,
        #use it for sensitivity testing
        #top_p=0.6
    )
    return gen_config

# Loading LLM model and tokenizer
def load_llm_model(llm_model_path):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path,padding_side = 'left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_model_path,torch_dtype=torch.float16,device_map = 'auto')
    return tokenizer, model

def get_few_shot_indexes(shots,task,exp):
    if exp =="semantic":
        exampleids = ex_idx[i][:shots]
    elif exp =="random":
        if task=="L1" or task=="L3":
            exampleids = np.random.choice(len(training_queries),size = shots, replace = False)
        if task == "L2":
            exampleids = np.random.choice(len(training_questions),size = shots, replace = False)
            bad_example = True
            while bad_example:
                bad_example = False
                exampleids = np.random.choice(len(training_questions),size = shots, replace = False)
                for idx in exampleids:
                    if '-' in training_answers[idx]:
                        bad_example = True
    return exampleids