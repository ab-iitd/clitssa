#####################################################################################################
"""
Project: CLiTSSA
inference_cross_lingual.py: Script to generate metrics output for F1 and EM score from a response file
                            generated by an LLM
"""
######################################################################################################

## imports
import argparse
import pandas as pd
import sys
from eval_utils import get_pre_process_macros,get_processed_final_answer,get_processed_expected_answer,calculate_metrics

parser = argparse.ArgumentParser(prog='CLiTSA',
                    description='Script to fine-tune CLiTSSA retiever',
                    epilog='')
parser.add_argument("--llm_response_file", type=str ,help='LLM response file containing groundtruths and generated texts columns')
parser.add_argument("--task", type=str ,help='tasks for which CLiTSSA is being fine-tuned - L1, L2 and L3')
parser.add_argument("--language", type=str ,help='language for which CLiTSSA is being fine-tuned - Romanian, German, French')
parser.add_argument("--setup", type=str ,help='choices: few_shot or zero_shot')

args = parser.parse_args()

llm_response_file = args.llm_response_file
lang = args.language
task = args.task
setup = args.setup

# read llm response file
response_data = pd.read_csv(llm_response_file)

processed = 0
em_match_count = 0
cumulative_f1_score = 0
initials = initials2 = []
months_short = months_num = months_num2 = months_eng_rom = {}

# get some macros to clean generated dataset
initials, months_short, months_num, months_num2, months_eng_rom  = get_pre_process_macros(lang, task, setup)

# iterate over all samples
for _, sample in response_data.iterrows():
    final_gnerated_answer = ""
    final_expected_answer = ""
    final_expected_answer2 = ""
    # get expected and generated llm response for an instance
    try:
        if task == "L3":
            expected_answer = sample['correct_answer']
        else:
            expected_answer = sample['correct_answer'].strip('][').split(', ')
    except:
        continue
    generated_answer = sample['predicted_answer']
    # if generated reposne is just a numeric float value/ garbage then continue
    if type(generated_answer) == float:
        processed += 1
        continue
    processed += 1

    # preprocess expected answer 
    final_expected_answer,final_expected_answer2 = get_processed_expected_answer(lang,expected_answer,months_short,months_eng_rom, task)
    # preprocess generated answer 
    final_gnerated_answer = get_processed_final_answer(lang,setup,generated_answer,initials, months_short, months_num, months_num2, months_eng_rom, task, final_expected_answer)

    # empty clean response from LLM then continue
    if final_gnerated_answer == '':
        continue
    # get metrics outputs and store it
    em_match_value, f1 = calculate_metrics(final_gnerated_answer,final_expected_answer,final_expected_answer2,task)
    em_match_count += em_match_value
    cumulative_f1_score += f1

# Avergae EM and F1 scores of all samples
final_f1_score = cumulative_f1_score / processed
final_em_score = em_match_count/processed

print("F1 Score: "+str(final_f1_score))
print("EM Score: "+str(final_em_score))