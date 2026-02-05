LLMs with retrieval augmentation excel at knowledge-intensive tasks but often fail to distinguish semantically similar yet critically different inputs, a core obstacle to reliable reasoning. We present {Vul-RAG+}, a retrieval-augmented reasoning framework that improves grounding and reduces overconfidence through calibrated prompts and a hybrid retriever combining embeddings, lexical similarity, and a learned re-ranker. To assess its quality, we introduce {TruePairVul}, a benchmark ensuring each query has a relevant entry. Experiments show Vul-RAG+ significantly improves pairwise discrimination, grounding reliability, and calibration over strong baselines. We validate our approach on the industry-standard BigVul dataset and a Java code segment, demonstrating applicability beyond Linux kernel vulnerabilities and C/C++ code. In addition, we evaluate multiple model variants to assess robustness across different architectures. Beyond code analysis, our framework and evaluation provide broader lessons for retrieval-augmented NLP in fine-grained reasoning tasks.


How to recreate results:

1. Install Elasticsearch
* Utilized Version 9.2.3
* Ensure connection through the ES_CONFIG in common/config.py


2. Build virtual environment
* Utilize the provided requirements.txt


3. Load API Keys
* Create folder common/api_keys
* Load file with provided common/load_keys.py file 


4. Load Elasticsearch with data using ChatGPTExtraction.py

* Run divide_data.py in partial/{benchmark}/1_raw_data

* Run *python ChatGPTExtraction.py* with the following parameters

* *--model_name str* (We used the default model of gpt-3.5-turbo, but you can set a different model if desired)

* *--store_knowledge* (signals the system to put knowledge in elasticsearch)

* *--extract_knowledge* (signals system to extract to a json file)

* *--benchmark str* (Specifies the benchmark that will be run, and subsequently the CWEs)

* SPECIAL CASE: BigVul is a testset only. Please run ChatGPTExtraction on PairVul, and copy PairVul's 2_elasticsearch_ready directory into BigVul's 2_elasticsearch_ready directory.


5. Enrich test data

* Run *python v2testing.py* with the following parameters

* *-action str* (should be "enrich_test")

* *--benchmark str* (Specifies the benchmark directory and cwes we are running on)

* *--model str* (Specifies the model to use for testset preprocessing. Our testing utilized gpt-3.5-turbo which is the default)


6. Run search

* Run *python v2testing.py* with the following parameters

* *-action str* (should be "search")

* *--benchmark str* (Specifies the benchmark directory and cwes we are running on)

* *--top_K int* (Controls the length of lists returned from searching. This is different from the amount of information provided to the decision making step.)

* *--action_type int* (Specifies what type of search to perform)
    0: Search only using BM25 as used in Vul-RAG original
    1: Uses BM25 and embeddings with no formatting applied
    2: Searches only embeddings
    3: Searches using BM25 and embeddings and formats for input to the learned reranker as used in Vul-RAG+
    4: Searches using BM25 and embeddings and formats for input into the lazy learned reranker ie. no backfilling

* *--new_directory str* (Specifies a new subdirectory to place results into)

* *--all* (Runs on all possible action types)


7. Run reranking

* Run *python v2testing.py* with the following parameters

* *-action str* (should be "rerank")

* *--benchmark str* (Specifies the benchmark directory and cwes we are running on)

* *--action_type* (Specifies what type of reranking to perform)
    1: RRF with 50-50 weighting (BM25 - Embed) 
    2: RRF with 75-25 weighting (BM25 - Embed)
    3: RRF with 0-100 weighting (BM25 - Embed)
    4: RRF with 100-0 weighting (BM25 - Embed) used in Vul-RAG original
    5: RRF with only signals from Code
    6: RRF with only signals from Function
    7: RRF with only signals from Purpose
    8: Learned Reranker with lazy reranking
    9: Full Learned Reranker used in Vul-RAG+

* *--top_N int* (Controls the amount of knowledge returned from this step. Directly controls the amount of knowldge the decision step recieves.)

* *--new_directory str* (Specifies a new subdirectory to place results into)

* *--all* (Runs on all possible rerank types)

* *--input_dir str* (Specifies the input_directory. Make sure this matches the format you are trying to rerank. Possible formats are RRF format or learned format)


8. Run decision making

* Run *python v2testing.py* with the following parameters

* *-action str* (should be "decision")

* *--benchmark str* (Specifies the benchmark directory and cwes we are running on)

* *--input_dir str* (Specifies the input_directory.)

* *--desc str* (Give a description to your output directory.)

* *--model str* (Specify the model that runs the decision making process)

* *--resume* (Allows resuming functionality if the process is interrupted)

* *--prompt* (Specifies the prompt type to use. Prompt types can be edited in common/common_prompt.py)
    0: Original prompt used in Vul_RAG original
    1: +Reordering
    2: +Positional Emphasis
    3: +Role Neutralization. This prompt is used in Vul-RAG+

* NOTE! This file is set to take in a reranked list with a maximum knowledge list length of 10. This function will run with N=10 then cut down to N = 9, 8, 7, ..., 1. If you want to change what sub-length files are created please see the top of cut_down() within v2testing.py. 

Metrics will run on all length files. Metrics are two-fold. A {cwe}_metrics .json file will be created for each CWE as well as for Overall metrics. This file holds the metrics run a single time and holds all gathered information. A bootstrap.json file will also be created which holds some metrics run on bootstrap over pairs 10,000 times. This will contain mean and 95% confidence metrics for the main metrics. If you want to add metrics to bootstrap.json please check the 