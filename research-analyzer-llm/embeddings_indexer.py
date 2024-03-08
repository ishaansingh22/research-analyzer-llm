import numpy as np
import faiss
from document_parser import DocumentParser
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch


class EmbeddingsIndexer:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dimension = self.model.config.hidden_size
        self.faiss_index = faiss.IndexFlatL2(self.dimension)

    def encode(self, texts, batch_size=32):
        """Encodes a list of texts into embeddings."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # cls token embedding
            embeddings_batch = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embeddings_batch)
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

    def add_documents(self, texts):
        """Adds documents to the FAISS index after encoding."""
        embeddings = self.encode(texts)
        self.faiss_index.add(embeddings)

    def search(self, query, k=5):
        """Searches the index for the k most similar documents to a query."""
        query_embedding = self.encode(
            [query])[0]
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]), k)
        return indices[0], distances[0]

# testing
if __name__ == "__main__":
    indexer = EmbeddingsIndexer()
    parsed_text = DocumentParser.extract_text_from_pdf("/content/sample.pdf")
    structured_doc = DocumentParser.structured_document(parsed_text)
    texts_to_index = []
    for section in structured_doc["sections"].values():
        section_text = " ".join(section)
        texts_to_index.append(section_text)
    indexer.add_documents(texts_to_index)
    query = "Summarize this research paper please"
    indices, distances = indexer.search(query)
    print("Indices of top documents:", indices)
    print("Distances:", distances)
