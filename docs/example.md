# Graph RAG Example

Haystack orchestrates the indexing and query pipelines. Neo4j stores chunks, embeddings, entities, and relations. Gemma 4 31B Instruct is called through a HuggingFace Dedicated Endpoint for entity extraction and answer generation.

The system combines vector retrieval with graph traversal. Vector retrieval finds semantically similar chunks. Graph traversal expands from mentioned entities to connected chunks and relation context.
