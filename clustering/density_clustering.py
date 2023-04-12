import utility
import config
from question import Question
from sklearn.metrics.pairwise import cosine_similarity as cs
from sentence_transformers import SentenceTransformer,util
import sys
import time
import subprocess
import json
import html
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import hdbscan
import os
join = os.path.join

from transformers import AutoTokenizer, AutoModel
import torch

# Argument list
argumentList = sys.argv[1:]

# Setting Config Parameters
config_params = config.getConfig(argumentList)
if not bool(config_params):
    print("Configuration not found...")
    sys.exit()
locals().update(config_params)

# Getting Question From MySQL Database
# Change this part if you want to import questions from file or other database

# username and password credential for connecting to MySQL database
username = "root"
password = "user@123"

result, row_headers = utility.getInputQuestionSQL(username,password,tag_list,number_of_samples)

q_arr = list()
c_arr = list()

# Extracting questions from the result
for x in result:
    q_x= Question(x,row_headers)
    q, c = q_x.getText()

    # Format the code to remove html entitites
    c = html.unescape(c)
    # Remove newlines
    c = c.replace('\n', ' ')

    q_arr.append(q)
    c_arr.append(c)

print('Number of questions: ', len(q_arr),len(c_arr))

# code summarizer from Code-T5 (This summarizer works for both Javascript and Python) 
code_tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
code_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

for i,(q,c) in enumerate(zip(q_arr,c_arr)):
    if c == '':
        continue
    input_ids = code_tokenizer(c,truncation=True,max_length=512, return_tensors="pt").input_ids

    generated_ids = code_model.generate(input_ids, max_length=20)
    q_arr[i] = q + ' ' + code_tokenizer.decode(generated_ids[0], skip_special_tokens=True)


print(f"Number of questions after code summarization: {len(q_arr)}")
# Vectorization of question
start = time.time()
model = SentenceTransformer(transformer_model)
embedding = model.encode(q_arr)
end = time.time()
print("Model + Embedding Time : ",end-start)

embedding = torch.tensor(embedding)
# Printing Information regarding Clustering
utility.printInitialInfo(len(embedding),threshold,tag_list,transformer_model,len(embedding[0]))

# Pairwise Cosine Similarity Index Matrix
# start = time.time()
# print(f"Embedding Shape: {embedding.shape}")
# similarity_index = cs(embedding)
# print(f"Similarity index: {similarity_index}")
# end = time.time()
# print("Cosine Similarity Index Matrix Calculation Time : ",end-start)

# Cluster Generation
start = time.time()
clusters,umap_embeddings = utility.generate_clusters(embedding,len(embedding)//4,2,2,42) # for density based clustering 
end = time.time()

labels = clusters.labels_

c = [[] for _ in range(len(set(labels)))]
for i, l in enumerate(labels):
    c[l].append(i)

clusters = c
# visualizing umap embeddings
hdbscanner = hdbscan.HDBSCAN()
hdbscan_labels = hdbscanner.fit_predict(umap_embeddings)
plt.scatter(umap_embeddings[:,0], umap_embeddings[:,1], c=labels) 
plt.savefig(join(os.getcwd(),"dense_clusters.png"))

print("Cluster Generation Time : ",end-start)
print("--------------------------------------------------------------------------------------------------")

# Printing Cluster Information
utility.printCluster(clusters,q_arr,details=enable_question_print,print_min_size=print_min_size)

# Printing Evaluation Score 
utility.printScores(embedding,labels)
