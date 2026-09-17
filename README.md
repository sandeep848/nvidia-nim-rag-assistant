# NVIDIA NIM Knowledge Assistant

A document-question-answering system built around **hybrid retrieval and auditable citations**. It uses NVIDIA NIM embeddings and generation, but keeps ranking logic explicit and testable.

## Distinct use case

This repository is a focused organizational knowledge-base assistant. It does not search the web or behave as a general tool-using agent.

Its distinguishing logic is:

1. split uploaded PDFs into page-aware passages
2. create dense NVIDIA embeddings
3. compute a separate lexical ranking
4. combine both rankings with reciprocal-rank fusion
5. answer only from the fused evidence
6. display page-level sources beside the answer

## Architecture

~~~mermaid
flowchart LR
    A["PDF knowledge base"] --> B["Page-aware chunks"]
    B --> C["NVIDIA dense retrieval"]
    B --> D["Lexical retrieval"]
    C --> E["Reciprocal-rank fusion"]
    D --> E
    E --> F["NIM answer with citations"]
~~~

## Run

~~~bash
git clone https://github.com/sandeep848/nvidia-nim-rag-assistant.git
cd nvidia-nim-rag-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run final.py
~~~

## Test

~~~bash
pytest
~~~

`rag_core.py` contains the provider-independent lexical ranker and fusion algorithm, allowing retrieval behavior to be tested without API calls.

## Limitations

- Uploaded files are indexed in the current Streamlit session.
- Citation presence does not guarantee that the cited passage entails every claim.
- Production use requires persistent indexes, access controls and evaluation on a domain-specific question set.
