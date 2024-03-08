import json
import replicate  # Assuming replicate or similar library is used for LLM interaction
from embeddings_indexer import EmbeddingsIndexer


class QuestionAnsweringSystem:
    def __init__(self, indexer, llm_model):
        """
        Initializes the QA system.
        :param indexer: An instance of EmbeddingsIndexer or similar that can search for relevant document sections.
        :param llm_model: The model identifier for the LLM.
        """
        self.indexer = indexer
        self.llm_model = llm_model

    def generate_prompt(self, query, context):
        """
        Generates a prompt for the LLM based on the title and context.
        :param title: The title or the main query.
        :param context: The context or document sections relevant to the query.
        """
        prompt_template = """
        Please give me an insightful answer for this question: "{query}". The following information may be useful to you:
        {doc_texts}
        Generate your answer in concise and write a summarization paragraph at the end if the length of the response is very long\n\n"""
        return prompt_template.format(query=query, doc_texts="\n\n".join(context))

    def structure_input(self, prompt, max_tokens, temp, top_p):
      """
      Generates the structured input.
      """
      input= {"prompt": prompt, "max_new_tokens": max_tokens, "temperature": temp, "top_p": top_p}
      return input

    def query_llm(self, prompt, max_len=800, temp=.74, top_p=.9):
        """
        Queries the LLM with the given prompt and parameters.
        :param prompt: The prompt for the LLM.
        """
        input = self.structure_input(prompt, max_len, temp, top_p)
        response = replicate.run(self.llm_model, input=input)
        return response

    def answer_query(self, query):
        """
        Generates an answer to the query using the embeddings indexer and LLM.
        :param query: The query to answer.
        """
        indices, distances = self.indexer.search(query, k=3)  # fetch top 3 relevant documents/sections
        relevant_texts = self.fetch_texts_by_indices(indices)
        prompt = self.generate_prompt(query, relevant_texts)
        llm_response = self.query_llm(prompt)
        return llm_response

    def fetch_texts_by_indices(self, indices):
        """
        Fetches the texts corresponding to the given indices.
        Placeholder method - implement according to how your documents are stored.
        :param indices: The indices of documents to fetch.
        """
        relevant_texts = ["Document text for index {}".format(index) for index in indices]
        return relevant_texts

#testing
llm_model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
qa_system = QuestionAnsweringSystem(EmbeddingsIndexer(), llm_model)
query = "Can you summarize the sample paper?"
response = qa_system.answer_query(query)
response = "".join(response)
print(response)