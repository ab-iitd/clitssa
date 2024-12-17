#####################################################################################################
"""
Project: CLiTSSA
eval_utils.py: Script to implement summplement methods to calculate metrics F1 and EM
"""
######################################################################################################

## imports
import argparse
import pandas as pd

# pre-processing generated data
def get_processed_final_answer(lang,setup,generated_answer,initials, months_short, months_num, months_num2, months_eng_rom, task, expected_answer):
    
    if task == "L1":
        return get_processed_final_answer_L1(lang,setup,generated_answer,initials, months_short, months_num, months_num2, months_eng_rom)
    if task == "L2":
        return get_processed_final_answer_L2(lang,setup,generated_answer,initials,expected_answer)
    if task == "L3":
        return get_processed_final_answer_L3(lang,setup,generated_answer,initials,expected_answer)

# pre-processing expected data
def get_processed_expected_answer(lang,expected_answer,months_short,months_eng_rom,task):
    if task == "L1":
        return get_processed_expected_answer_L1(lang,expected_answer,months_short,months_eng_rom)
    if task == "L2":    
        return get_processed_expected_answer_L2(lang,expected_answer)
    if task == "L3":    
        return get_processed_expected_answer_L3(lang,expected_answer)

# pre-processing generated data for L1 task
def get_processed_final_answer_L1(lang,setup,generated_answer,initials, months_short, months_num, months_num2, months_eng_rom):

    generated_answer = generated_answer.lower()
    final_answer =""

    # split the response to retirve relevant text
    split_word1 = "question"
    split_word2 = ""
    split_word3 = ""
    split_word4 = " de "

    if lang == "English":
        split_word2 = 'question'
        split_word3 = ' is '
        split_word4 = ' from '
    elif lang == "French":
        split_word2 = 'question'
        split_word3 = ' est '
    elif lang == "German":
        split_word2 = 'frage'
    elif lang == "Romanian":
        split_word2 = 'întrebare'

    if setup == 'few_shot':

        if split_word2 in generated_answer:
            final_answer = ''.join(generated_answer.split(split_word2)[0]).strip()
        else:
            final_answer = generated_answer
        
        if split_word1 in final_answer:
            final_answer = ''.join(final_answer.split(split_word1)[0]).strip()
        
        if split_word3 != "":
            if split_word3 in final_answer:
                final_answer = ''.join(final_answer.split(split_word3.strip())[1]).strip()
                final_answer = ' '.join(final_answer.split()[:2]).strip()
            else:
                final_answer = ' '.join(final_answer.split()[-2:]).strip()

    elif setup == 'zero_shot':
        if split_word2 in generated_answer:
            final_answer = ''.join(generated_answer.split(split_word2)[0]).strip()

        if split_word4 in final_answer:
            final_answer = ''.join(final_answer.split(split_word4.strip())[0]).strip()

        if split_word3 != "":
            if split_word3 in final_answer:
                final_answer = ''.join(predicted_answer.split(split_word3.strip())[1]).strip()
            else:
                final_answer = predicted_answer

        final_answer = ' '.join(final_answer.split()[:2]).strip()

    # remove some extra initials
    for extra in initials:
        if extra in final_answer:
            final_answer = ''.join(final_answer.split(extra)[1]).strip()
    if final_answer == '':
        return final_answer

    if lang == 'French' and "l'année" in final_answer:
        final_answer = ''.join(final_answer.split("l'année")[-1]).strip()
    if lang == 'German' and "datum" in final_answer:
        final_answer = ''.join(final_answer.split("datum")[-1]).strip()
    if final_answer == '':
        return final_answer
    
    # remove some extra delimiters in text 
    if final_answer[-1] == '.' or final_answer[-1] == '?':
        final_answer = final_answer[:-1]
    if ', ' in final_answer:
        final_answer = ' '.join(final_answer.split(', '))
    if ',' in final_answer:
        final_answer = ' '.join(final_answer.split(','))
    if '-' in final_answer:
        final_answer = ' '.join(final_answer.split('-'))
    if '/' in final_answer:
        final_answer = ' '.join(final_answer.split('/'))
    answers = final_answer.split()
    final_answer = ' '.join(answers).strip()
    
    # standradize the month names
    if len(answers) == 2:
        if answers[0] in months_num.keys():
            final_answer = months_num[answers[0]] + ' ' + answers[1]
        elif answers[1] in months_num.keys():
            final_answer = answers[0] + ' ' + months_num[answers[1]]
        elif answers[0] in months_num2.keys():
            final_answer = months_num2[answers[0]] + ' ' + answers[1]
        elif answers[1] in months_num2.keys():
            final_answer = answers[0] + ' ' + months_num2[answers[1]]
    if 'date' in final_answer:
        final_answer = ''.join(final_answer.split('date')[1]).strip()
    for month, full in months_short.items():
        if month in final_answer:
            final_answer = final_answer.replace(month, full)
            break

    return final_answer

# pre-processing generated data for L2 task
def get_processed_final_answer_L2(lang,setup,generated_answer,initials,expected_answer):
    
    generated_answer = generated_answer.lower()
    final_answer = ""
    # split the response to retirve relevant text
    split_word1 = "question"
    split_word2 = ""
    split_word3 = "because"
    split_word4 = "parce que"
    split_word5 = ""
    
    if lang == "English":
        split_word2 = "question"
        split_word4 = "because"
    elif lang == "French":
        split_word2 = "question"
    elif lang == "German":
        split_word2 = "frage"
        split_word5 = "bitte"
    elif lang == "Romanian":
        split_word2 = 'întrebare'
        split_word5 = "acum"

    if split_word2 in generated_answer:
        final_answer = ''.join(generated_answer.split(split_word2)[0]).strip()
    else:
        final_answer = generated_answer
    if split_word1 in final_answer:
        final_answer = ''.join(final_answer.split(split_word1)[0]).strip()
    if split_word4 in final_answer:
        final_answer = ''.join(final_answer.split(split_word4)[0]).strip()
    if split_word3 in final_answer:
        final_answer = ''.join(final_answer.split(split_word3)[0]).strip()
    if lang == "German" or lang =="Romanian":
        if split_word5 in final_answer:
            final_answer = ''.join(final_answer.split(split_word5)[0]).strip()

    # remove some extra initials
    for extra in initials:
        if extra in final_answer:
            final_answer = ''.join(final_answer.split(extra)[1]).strip()
    if final_answer == '':
        return final_answer

    if final_answer[-1] == '.' or final_answer[-1] == ',':
        final_answer = final_answer[:-1]

    if setup == 'zero_shot' and not(final_answer == ''):
        if final_answer[0]==":":
            final_answer = final_answer[1:].strip()
        if len(final_answer.split(" ")) > len(expected_answer.split(" ")):
            final_answer = " ".join(final_answer.split(" ")[:len(expected_answer.split())+1])
    
    return final_answer

# pre-processing generated data for L3 task
def get_processed_final_answer_L3(lang,setup,generated_answer,initials,expected_answer):
    
    generated_answer = generated_answer.lower()
    final_answer = ""
    # split the response to retirve relevant text

    split_word1 = "question"
    split_word2 = "because"
    split_word3 = " is "
    split_word4 = ""
    split_word5 = ""
    split_word6 = ""
    split_word7 = ""

    # clean French and English Response
    if lang == "French" or lang == "English":
        split_word4 = " est "
        split_word5 = " for "
        split_word6 = " car si "
        split_word7 = "parce que"

        if split_word1 in generated_answer:
            final_answer = ''.join(generated_answer.split(split_word1)[0]).strip()
        else:
            final_answer = generated_answer
        if split_word2 in final_answer:
            final_answer = ''.join(final_answer.split(split_word2)[0]).strip()
        if split_word3 in final_answer:
            final_answer = ''.join(final_answer.split(split_word3)[1]).strip()
        if split_word4 in final_answer:
            final_answer = ''.join(final_answer.split(split_word4)[1]).strip()
        if split_word5 in final_answer:
            final_answer = ''.join(final_answer.split(split_word5)[1]).strip()
        if split_word6 in final_answer:
            final_answer = ''.join(final_answer.split(split_word6.strip())[0]).strip()
        if split_word7 in final_answer:
            final_answer = ''.join(final_answer.split(split_word6)[0]).strip()

    # clean German Response
    elif lang == "German":
        split_word4 = "frage"
        split_word5 = "bitte"
        split_word6 = " ist "

        if split_word4 in generated_answer:
            final_answer = ''.join(generated_answer.split(split_word4)[0]).strip()
        else:
            final_answer = generated_answer
        if split_word1 in final_answer:
            final_answer = ''.join(final_answer.split(split_word1)[0]).strip()
        if split_word6 in final_answer:
            final_answer = ''.join(final_answer.split(split_word6)[1]).strip()
        if split_word2 in final_answer:
            final_answer = ''.join(final_answer.split(split_word2)[0]).strip()
        if split_word5 in final_answer:
            final_answer = ''.join(final_answer.split(split_word5)[0]).strip()
        if split_word3 in final_answer:
            final_answer = ''.join(final_answer.split(split_word3)[1]).strip()

    #clean Romanian Response
    elif lang == "Romanian":
        split_word4 = 'întrebare'
        split_word5 = "acum"
        split_word6 = " este "

        if split_word1 in generated_answer:
            final_answer = ''.join(generated_answer.split(split_word1)[0]).strip()
        else:
            final_answer = generated_answer

        if split_word4 in final_answer:
                final_answer = ''.join(final_answer.split(split_word4)[0]).strip()
        if split_word5 in final_answer:
                final_answer = ''.join(final_answer.split(split_word5)[0]).strip()
        if split_word6 in final_answer:
            final_answer = ''.join(final_answer.split(split_word6)[1]).strip()
        #if ' ist ' in answer:
        #    answer = ''.join(answer.split(' ist ')[1]).strip()
        if split_word2 in final_answer:
            final_answer = ''.join(final_answer.split(split_word2)[0]).strip()

    # remove some extra initials
    for extra in initials:
        if extra in final_answer:
            final_answer = ''.join(final_answer.split(extra)[1]).strip()
    if final_answer == '':
        return final_answer

    if final_answer[-1] == '.' or final_answer[-1] == ',':
        final_answer = final_answer[:-1]

    if setup == 'zero_shot' and not(final_answer == ''):
        if final_answer[0]==":":
            final_answer = final_answer[1:].strip()
        if len(final_answer.split(" ")) > len(expected_answer.split(" ")):
            final_answer = " ".join(final_answer.split(" ")[:len(expected_answer.split())+1])

    return final_answer


# pre-processing expected data for L1 task
def get_processed_expected_answer_L1(lang,expected_answer,months_short,months_eng_rom):

    expected_answer = [element.strip("'").lower() for element in expected_answer]
    expected_answer = ' '.join(expected_answer).lower()

    # standradize the month names
    for month, full in months_short.items():
        if month in expected_answer:
            expected_answer = expected_answer.replace(month, full)
            break
    if lang == "Romanian":
        for month, full in months_eng_rom.items():
                if month in expected_answer:
                    expected_answer = expected_answer.replace(month, full)
                    break
    temp_answer = expected_answer.split()
    temp_answer.reverse()
    expected_answer2 = ' '.join(temp_answer).strip()

    return expected_answer,expected_answer2

# pre-processing expected data for L2 task
def get_processed_expected_answer_L2(lang,expected_answer):
    
    expected_answer = [element.strip("'").lower() for element in expected_answer]
    expected_answer = ', '.join(expected_answer).lower()
    return expected_answer, ""

# pre-processing expected data for L3 task
def get_processed_expected_answer_L3(lang,expected_answer):
    
    expected_answer = expected_answer.lower()
    return expected_answer, ""

# method to calculate exact match and f1 scores for a pair of text
def calculate_metrics(final_gnerated_answer,final_expected_answer,final_expected_answer2, task):
    em_match = 0
    f1 = 0
    # calculate exact match score
    if task =="L1":
        if final_gnerated_answer != "" and (final_gnerated_answer == final_expected_answer or final_gnerated_answer == final_expected_answer2):
            em_match = 1
        else:
            em_match = 0
    if task =="L2" or task == "L3":
        if final_gnerated_answer != "" and final_gnerated_answer == final_expected_answer:
            em_match = 1
        else:
            em_match = 0

    # calculate F1
    precision = 0
    recall = 0
    pred_tokens = final_gnerated_answer.split()
    try:
        truth_tokens = final_expected_answer.split()
    except AttributeError:
        return em_match, f1
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        f1 = int(pred_tokens == truth_tokens)
        return em_match, f1
        
    common_tokens = set(pred_tokens) & set(truth_tokens)
        
    if len(common_tokens) == 0:
        return em_match, f1

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return em_match, f1

# method to return macros to clean pre-process data
def get_pre_process_macros(lang, task, setup):
    
    if task == "L1":
        extras = ['1)', '1.', '2)', '2.', '3)', '3.']

        months_eng_rom = {}
    
        if lang == "French":

            months_short = {
            'jan ' : 'janvier ',
            'feb ' : 'février ' ,
            'mar ' : 'mars ',
            'apr ' : 'avril ',
            'may ' : 'mai ',
            'jun ' : 'juin ',
            'jul ' : 'juillet ',
            'aug ' : 'aout ',
            'sep ' : 'septembre ',
            'oct ' : 'octobre ',
            'nov ' : 'novembre ',
            'dec ' : 'décembre '}

            months_num = {
            '1' : 'janvier',
            '2' : 'février' ,
            '3' : 'mars',
            '4' : 'avril',
            '5' : 'mai',
            '6' : 'juin',
            '7' : 'juillet',
            '8' : 'aout',
            '9' : 'septembre',
            '10' : 'octobre',
            '11' : 'novembre',
            '12' : 'décembre'}

            months_num2 = {
            '01' : 'janvier',
            '02' : 'février' ,
            '03' : 'mars',
            '04' : 'avril',
            '05' : 'mai',
            '06' : 'juin',
            '07' : 'juillet',
            '08' : 'aout',
            '09' : 'septembre',
            '10' : 'octobre',
            '11' : 'novembre',
            '12' : 'décembre'}

        elif lang == "German":

            months_short = {
            'jan ' : 'januar ',
            'feb ' : 'februar ' ,
            'mar ' : 'märz ',
            'apr ' : 'april ',
            'may ' : 'mai ',
            'jun ' : 'juni ',
            'jul ' : 'juli ',
            'aug ' : 'august ',
            'sep ' : 'september ',
            'oct ' : 'oktober ',
            'nov ' : 'november ',
            'dec ' : 'dezember '}

            months_num = {
            '1' : 'januar',
            '2' : 'februar' ,
            '3' : 'märz',
            '4' : 'april',
            '5' : 'mai',
            '6' : 'juni',
            '7' : 'juli',
            '8' : 'august',
            '9' : 'september',
            '10' : 'oktober',
            '11' : 'november',
            '12' : 'dezember'}

            months_num2 = {
            '01' : 'januar',
            '02' : 'februar' ,
            '03' : 'märz',
            '04' : 'april',
            '05' : 'mai',
            '06' : 'juni',
            '07' : 'juli',
            '08' : 'august',
            '09' : 'september',
            '10' : 'oktober',
            '11' : 'november',
            '12' : 'dezember'}


        elif lang == "Romanian":

            months_short = {
            'jan ' : 'ianuarie ',
            'feb ' : 'februarie ' ,
            'mar ' : 'martie ',
            'apr ' : 'aprilie ',
            'may ' : 'mai ',
            'jun ' : 'iunie ',
            'jul ' : 'iulie ',
            'aug ' : 'august ',
            'sep ' : 'septembrie ',
            'oct ' : 'octombrie ',
            'nov ' : 'noiembrie ',
            'dec ' : 'decembrie '}

            months_eng_rom = {
            'january': 'ianuarie',
            'february' : 'februarie',
            'march': 'martie',
            'april': 'aprilie',
            'may': 'mai',
            'june': 'iunie',
            'july': 'iulie',
            'august': 'august',
            'september': 'septembrie',
            'october': 'octombrie',
            'november': 'noiembrie',
            'december': 'decembrie'}

            months_num = {
            '1' : 'ianuarie',
            '2' : 'februarie' ,
            '3' : 'martie',
            '4' : 'aprilie',
            '5' : 'mai',
            '6' : 'iunie',
            '7' : 'iulie',
            '8' : 'august',
            '9' : 'septembrie',
            '10' : 'octombrie',
            '11' : 'noiembrie',
            '12' : 'decembrie'}

            months_num2 = {
            '01' : 'ianuarie',
            '02' : 'februarie' ,
            '03' : 'martie',
            '04' : 'aprilie',
            '05' : 'mai',
            '06' : 'iunie',
            '07' : 'iulie',
            '08' : 'august',
            '09' : 'septembrie',
            '10' : 'octombrie',
            '11' : 'noiembrie',
            '12' : 'decembrie'}

        else:
            print("please provide valid language paramter - French, German, Romanian")
    
        return extras, months_short, months_num, months_num2, months_eng_rom

    if task == "L2":
        extras = ['1)', '1.', '2)', '2.', '3)', '3.','4.']
        extras2 = ['(A).', 'A.', '(B).', 'B.','(C).', 'C.','(D).', 'D.']

        if setup == "zero_shot":
            extras += extras2

        return extras, {}, {}, {}, {}