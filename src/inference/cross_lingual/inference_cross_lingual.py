#####################################################################################################
"""
Project: CLiTSSA
inference_cross_lingual.py: Script to generate responses from a specified LLM for a set language-
                            and a temporal task in cross-lingual setting using mTEMPREASON dataset
"""
######################################################################################################

## imports
import sys,os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ['CURL_CA_BUNDLE'] = ''
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
from inference_utils import get_cross_lingual_prompt_template,build_prompt,load_datafile,load_indexfile,get_generation_config,load_llm_model,get_few_shot_indexes

parser = argparse.ArgumentParser(prog='CLiTSA',
                    description='Script to fine-tune CLiTSSA retiever',
                    epilog='')
parser.add_argument("--llm_model_path", type=str ,help='path to to an LLM model. i.e., Llama-3-8b-hf/')
parser.add_argument("--test_data_file", type=str ,help='path to the test data file i.e., test_l1_french.json')
parser.add_argument("--example_data_file", type=str ,help='path to the example data file from where the examples to be taken i.e., train_l1.json')
parser.add_argument("--task", type=str ,help='tasks for which CLiTSSA is being fine-tuned - L1, L2 and L3')
parser.add_argument("--language", type=str ,help='language for which CLiTSSA is being fine-tuned - Romanian, German, French')
parser.add_argument("--experiment", type=str ,help='choices: random, semantic')
parser.add_argument("--semantic_index_file", type=str ,help='if experiment is semantic, a file which provides the similar indexes from examples datasets for a query in test data')
parser.add_argument("--shots", type=int ,help='few-shots to run. i.e., 3')
parser.add_argument("--output_file", type=int ,help='a path for output csv file')

args = parser.parse_args()

llm_model_path = args.llm_model_path
test_data_file = args.test_data_file
example_data_file = args.example_data_file
task = args.task
language = args.language
experiment = args.experiment
semantic_index_file = args.semantic_index_file
shots = args.shots
output_file = args.output_file
batch_size = 10

# Defining Prompt Templates for Low-Resource Languages
prompt_head,example_prompt,query_prompt,prompt_tail = get_cross_lingual_prompt_template(task,language)

# Get Generation Config for LLM
gen_config = get_generation_config(task)

# Loading LLM model and tokenizer
tokenizer, model = load_llm_model(llm_model_path)

# loading test and example dataset
test_queries, test_answers = load_datafile(test_data_file)
examples_queries, examples_answers = load_datafile(example_data_file)

# loading semantic indexes if experiment choice is 'semantic'
ex_idx = load_indexfile(semantic_index_file,experiment)

# creating an output file
outputfile = open(output_file, 'w')
outputwriter = csv.writer(outputfile)
outputwriter.writerow(["query", 'expected_answer', 'generated_answer'])

batch_answers = []
batch_prompts = []
batch_queries = []

# iterating over each query in test data
for i,query in enumerate(tqdm.tqdm(test_queries)):

    expected_answer = test_answers[i]
    # getting the semantically or random aligned few_shot example indexes based on experiment setting
    exampleids = get_few_shot_indexes(shots,task,experiment)
    
    batch_queries.append(query)
    batch_answers.append(expected_answer)
    # build final prompt for a query along with examples
    prompt = build_prompt(examples_queries, examples_answers, shots, query, exampleids, prompt_head, example_prompt,query_prompt,prompt_tail)
    batch_prompts.append(prompt)

    if (len(batch_prompts)< batch_size) and (i + 1 != len(test_queries)):
        continue
    
    input_ids = tokenizer(batch_prompts, return_tensors="pt",padding=True)
    input_ids = input_ids.to(model.device)

    # calling LLM model to generate next tokens
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            generation_config=gen_config,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )

    # decode LLM's response
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_answers = []
    
    # post process and save the response
    for prompt,response in zip(batch_prompts,responses):
        try:
            generated_answers.append((response.split(prompt)[1]).strip().replace("\n"," "))
        except:
            generated_answers.append(response.strip().replace("\n"," "))
    for _query,_expected_answer,_generated_answer in zip(batch_queries,batch_answers,predicted_answers):
        outputwriter.writerow([_query,_expected_answer,_generated_answer])
    
    #clear the buckets for next batch
    batch_queries.clear()
    batch_prompts.clear()
    batch_answers.clear()

