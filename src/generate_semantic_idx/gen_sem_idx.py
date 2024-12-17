############################################################################################
"""
Project: CLiTSSA
gen_sem_idx.py: Script to generate semantically aligned indexes for few-shot through- 
                a retiever model
"""
############################################################################################

## imports
import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
import argparse
from sentence_transformers import SentenceTransformer, util
import torch

#function to get top-p semantically similar examples indexes
def genSimilarIndexes(retriever_model_path,test_data_file,examples_data_file,top_p):

    test_set = test_data_file
    train_set = examples_data_file

    #load retirever model
    retriever_model = SentenceTransformer(retriever_model_path)

    #load test and examples dataset
    df_train = pd.read_json(train_set, lines=True)
    df_test = pd.read_json(test_set, lines=True)

    test_final = df_test.to_dict(orient='records')
    example_corpus=df_train['question'].to_list()

    # generate embeddings for example dataset
    examples_embeddings = retriever_model.encode(example_corpus, convert_to_tensor=True)

    df_out = []
    i = 0
    for instance in test_final:
        # print intermediate status 
        if i%500==0:
            print(i + " instances completed")
        query=instance['question']
        # get query embedding
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # get cos similarity scores between a query embedding anf examples embeddings
        cos_scores = util.cos_sim(query_embedding, examples_embeddings)[0]
        # select top-p indexes and scores
        top_p_results = torch.topk(cos_scores, k=top_p)

        top_p_idx = []
        for score, idx in zip(top_p_results[0], top_p_results[1]):
            top_p_idx.append(idx.item())
        df_out.append({'sem_indexes' : top_p_idx})
        i += 1

    return df_out
    

if __name__ =='__main__':

    parser = argparse.ArgumentParser(prog='CLiTSA',
                    description='Script to generate semantically aligned example indexes for few-shots',
                    epilog='')
    parser.add_argument("--retriever_model_path", type=str ,help='path to a retriever model i.e., distiluse-base-multilingual-cased-v1 or CLiTSSA fine-tuned')
    parser.add_argument("--test_data_file", type=str ,help='path to a test data file for a language and task')
    parser.add_argument("--examples_data_file", type=str ,
                help='path to a example data file to get indexes of semantically similar examples. i.e, train file for a task and language, for cross lingual its a English train set')
    parser.add_argument("--top_p", type=int ,help='how many indexes for a query to be generated.')
    parser.add_argument("--task", type=str ,help='tasks for which CLiTSSA is being fine-tuned - L1, L2 and L3')
    parser.add_argument("--language", type=str ,help='language for which CLiTSSA is being fine-tuned')
    parser.add_argument("--output_dir", type=str ,help='output directory location where indexes to be stored')
    
    args = parser.parse_args()

    retriever_model_path = args.retriever_model_path
    test_data_file = args.test_data_file
    examples_data_file = args.examples_data_file
    top_p = args.top_p
    language = args.language
    task = args.task
    output_dir = args.output_dir

    # output file where indexes will be saved
    output_file_path = f"{output_directory}/{task}_{language}_semantic_indexes.csv"

    # get top-p indexes
    output_df = genSimilarIndexes(retriever_model_path,test_data_file,examples_data_file,top_p)
    
    # save the output file
    pd.DataFrame(output_df).to_csv(output_file_path)