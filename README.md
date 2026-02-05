LLMs with retrieval augmentation excel at knowledge-intensive tasks but often fail to distinguish semantically similar yet critically different inputs, a core obstacle to reliable reasoning. We present {Vul-RAG+}, a retrieval-augmented reasoning framework that improves grounding and reduces overconfidence through calibrated prompts and a hybrid retriever combining embeddings, lexical similarity, and a learned re-ranker. To assess its quality, we introduce {TruePairVul}, a benchmark ensuring each query has a relevant entry. Experiments show Vul-RAG+ significantly improves pairwise discrimination, grounding reliability, and calibration over strong baselines. We validate our approach on the industry-standard BigVul dataset and a Java code segment, demonstrating applicability beyond Linux kernel vulnerabilities and C/C++ code. In addition, we evaluate multiple model variants to assess robustness across different architectures. Beyond code analysis, our framework and evaluation provide broader lessons for retrieval-augmented NLP in fine-grained reasoning tasks.


How to recreate results:

1. Install Elasticsearch
2. Build virtual environment
3. Load Elasitcsearch with data using ChatGPTExtraction.py
4. Enrich test data
5. Run search
6. Run reranking
7. Run decision making