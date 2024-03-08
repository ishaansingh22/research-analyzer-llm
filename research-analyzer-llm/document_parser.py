import re
from pdfminer.high_level import extract_text
import spacy

#set up
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import tools
from langchain.llms import Replicate
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
nlp = spacy.load("en_core_web_sm")

class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """Extracts text from a PDF file using pdfminer.six."""
        text = extract_text(pdf_path)
        return text

    @staticmethod
    def structured_document(text):
        """Structures the document text into sections and identifies table captions."""
        section_headers = DocumentParser.extract_headers_with_nlp(text)
        section_pattern = "|".join([re.escape(header)
                                   for header in section_headers])
        table_pattern = r"Table \d+:"

        lines = text.split('\n')
        structured_doc = {"headers": [], "sections": {}, "tables": []}

        current_section = None
        for line in lines:
            if re.match(section_pattern, line, re.IGNORECASE):
                current_section = line.strip()
                structured_doc["sections"][current_section] = []
                structured_doc["headers"].append(current_section)
            elif re.search(table_pattern, line):
                structured_doc["tables"].append(line.strip())
            elif current_section:
                structured_doc["sections"][current_section].append(
                    line.strip())
        return structured_doc

    @staticmethod
    def extract_headers_with_nlp(text):
        """
        Attempts to extract section headers from a research paper using NLP.
        """
        potential_headers = []
        doc = nlp(text)

        for sent in doc.sents:
            if sent.text.strip()[0].isupper() and len(sent.text.strip()) < 100:
                if any(chunk.label_ == 'NP' for chunk in sent.noun_chunks):
                    potential_headers.append(sent.text.strip())

        return potential_headers
