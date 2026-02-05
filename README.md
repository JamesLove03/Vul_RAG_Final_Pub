# Vul-RAG+

LLMs with retrieval augmentation excel at knowledge-intensive tasks, but they often struggle to distinguish between semantically similar yet critically different inputs—an important obstacle to reliable reasoning.

**Vul-RAG+** is a retrieval-augmented reasoning framework designed to improve grounding and reduce overconfidence through:
- Calibrated prompts
- A hybrid retriever combining embeddings, lexical similarity, and a learned re-ranker

To support rigorous evaluation, we introduce **TruePairVul**, a benchmark that ensures every query has at least one relevant entry. Experiments demonstrate that Vul-RAG+ significantly improves pairwise discrimination, grounding reliability, and calibration over strong baselines.

We validate our approach on the industry-standard **BigVul** dataset as well as a **Java code segment**, showing applicability beyond Linux kernel vulnerabilities and C/C++ code. We also evaluate multiple model variants to assess robustness across architectures. Beyond vulnerability analysis, this framework provides broader insights for retrieval-augmented NLP in fine-grained reasoning tasks.

---

## Reproducing Results

### 1. Install Elasticsearch
- Tested with **Elasticsearch 9.2.3**
- Ensure the connection is configured via `ES_CONFIG` in `common/config.py`

---

### 2. Build the Virtual Environment
- Create and activate a virtual environment
- Install dependencies using the provided `requirements.txt`

---

### 3. Load API Keys
- Create the directory:
  ```
  common/api_keys/
  ```
- Add your API key files
- Load them using:
  ```
  python common/load_keys.py
  ```

---

### 4. Load Elasticsearch with Data
Run data extraction using `ChatGPTExtraction.py`.

1. Divide raw data:
   ```
   python divide_data.py
   ```
   from:
   ```
   partial/{benchmark}/1_raw_data
   ```

2. Run extraction:
   ```
   python ChatGPTExtraction.py --benchmark <benchmark> [options]
   ```

**Options**
- `--model_name str`  
  Default: `gpt-3.5-turbo`
- `--store_knowledge`  
  Store extracted knowledge in Elasticsearch
- `--extract_knowledge`  
  Save extracted knowledge to JSON

**Special Case: BigVul**
- BigVul is a *test-only* dataset
- Run `ChatGPTExtraction.py` on **PairVul**
- Copy:
  ```
  PairVul/2_elasticsearch_ready
  ```
  into:
  ```
  BigVul/2_elasticsearch_ready
  ```

---

### 5. Enrich Test Data
Run:
```
python v2testing.py -action enrich_test --benchmark <benchmark> --model <model>
```

- Default model: `gpt-3.5-turbo`

---

### 6. Run Search
```
python v2testing.py -action search --benchmark <benchmark> [options]
```

**Options**
- `--top_K int`  
  Number of candidates returned from search
- `--action_type int`
  - `0`: BM25 only (Vul-RAG original)
  - `1`: BM25 + embeddings (no formatting)
  - `2`: Embeddings only
  - `3`: BM25 + embeddings (formatted for learned reranker, Vul-RAG+)
  - `4`: BM25 + embeddings (lazy learned reranker)
- `--new_directory str`  
  Output subdirectory
- `--all`  
  Run all search types

---

### 7. Run Reranking
```
python v2testing.py -action rerank --benchmark <benchmark> [options]
```

**Options**
- `--action_type`
  1. RRF (50/50 BM25–Embedding)
  2. RRF (75/25)
  3. RRF (0/100)
  4. RRF (100/0, Vul-RAG original)
  5. Code-only signals
  6. Function-only signals
  7. Purpose-only signals
  8. Lazy learned reranker
  9. Full learned reranker (Vul-RAG+)
- `--top_N int`  
  Number of entries passed to decision-making
- `--input_dir str`  
  Input directory format (RRF or learned)
- `--new_directory str`
- `--all`

---

### 8. Run Decision Making
```
python v2testing.py -action decision --benchmark <benchmark> --input_dir <dir> [options]
```

**Options**
- `--desc str`  
  Description for output directory
- `--model str`  
  Model used for decision making
- `--resume`  
  Resume interrupted runs
- `--prompt`
  - `0`: Original Vul-RAG prompt
  - `1`: + Reordering
  - `2`: + Positional Emphasis
  - `3`: + Role Neutralization (Vul-RAG+)

**Note**
- Decision making assumes a maximum knowledge list length of **10**
- The system evaluates sublists of size 10 → 1
- To modify this behavior, edit `cut_down()` in `v2testing.py`

---

## Metrics
Metrics are computed for:
- Each individual CWE
- Overall performance

Outputs:
- `{cwe}_metrics.json`  
  Single-run metrics
- `bootstrap.json`  
  Bootstrap results over 10,000 samples, including means and 95% confidence intervals

To add additional bootstrap metrics, refer to the relevant section in the codebase.
