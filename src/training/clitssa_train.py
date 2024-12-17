##########################################################################################
"""
Project: CLiTSSA
clitssa_train.py: To fiunetune CLiTSSA retriever from paraller corpus
"""
##########################################################################################

## imports
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd
import sys
import numpy as np
import argparse
np.random.seed(1)


parser = argparse.ArgumentParser(prog='CLiTSA',
                    description='Script to fine-tune CLiTSSA retiever',
                    epilog='')
parser.add_argument("--base_model_path", type=str ,help='path to base retriever model i.e., distiluse-base-multilingual-cased-v1')
parser.add_argument("--train_data_file", type=str ,help='path to paraller corpus csv file')
parser.add_argument("--task", type=str ,help='tasks for which CLiTSSA is being fine-tuned - L1, L2 and L3')
parser.add_argument("--language", type=str ,help='language for which CLiTSSA is being fine-tuned')
parser.add_argument("--epoch", type=int ,help='epochs to run')

args = parser.parse_args()

train_data_path = args.train_data_file
language = args.language
task = args.task
ep = args.epoch
base_model_path = args.base_model_path

output_directory = "finetuned_models"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

 
finetuned_output_model_path = f"{output_directory}/{task}_{language}"

#Loading base model
model = SentenceTransformer(base_model_path,device='cuda')

#Loading training data
df = pd.read_csv(train_data_path)
text1 = df['text1'].tolist()
text2 = df['text2'].tolist()
scores = df['score'].tolist()

evaluator_sim =[]
train_examples =[]

#Train-test split 
test_idx= np.random.choice(list(range(len(text1))), int(len(text1)*0.1),replace=False)

# Transforming and pre-processing data for SentencesDataset() fuction
for k in range(len(text1)):
    if type(text1[k]) == type(""):
        if k in test_idx:
            evaluator_sim.append(InputExample(texts=[text1[k], text2[k]], label=scores[k]))
        else:
            train_examples.append(InputExample(texts=[text1[k], text2[k]], label=scores[k]))

train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)

print("dataset loaded")

# Define the evaluator
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(evaluator_sim, name='my_evaluator')

# Define the loss function
train_loss = losses.CoSENTLoss(model=model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=ep, warmup_steps=100, evaluator=evaluator, evaluation_steps=500)

# Save the model
model.save(finetuned_output_model_path)
