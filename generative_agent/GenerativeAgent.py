
# implementation of Generative Agents: Interactive Simulacra of Human Behavior
# references: https://arxiv.org/pdf/2304.03442.pdf
# Based on 
# https://github.com/hwchase17/langchain/blob/master/langchain/experimental/generative_agents/generative_agent.py

import logging
import math
import faiss
import re
import numpy as np
from datetime import datetime
from typing import List

from langchain import LLMChain
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from .TimeWeightedVectorStoreRetrieverModified import TimeWeightedVectorStoreRetrieverModified
from .prompt import *

logger = logging.getLogger(__name__)

def score_normalizer(val: float) -> float:
    return 1 - 1 / (1 + np.exp(val))

def get_text_from_docs(documents, include_time = False):
    ret = ""
    for i, document in enumerate(documents):
        time = document.metadata['created_at'].strftime('%A %B %d, %Y, %H:%M') + ": " if include_time else ""
        ret += '- ' if i == 0 else '\n- '
        ret += time + document.page_content
    
    return ret

def merge_documents(documents1, documents2):
    """"""
    buffer_indexes = []
    merged_documents = []
    for document in documents1:
        buffer_indexes.append(document.metadata['buffer_idx'])
        merged_documents.append(document)
    for document in documents2:
        if document.metadata['buffer_idx'] not in buffer_indexes:
            merged_documents.append(document)
    return merged_documents

def parse_as_list(text: str) -> List[str]:
    """Parse a newline-separated string into a list of strings"""
    lines = re.split(r"\n", text.strip())
    lines = [line for line in lines if line.strip()]
    return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

class GenerativeAgent():
    def __init__(self,
                 name: str, 
                 age: str, 
                 description: str,
                 traits: str,
                 embeddings_model,
                 current_time=None,
                 llm=None,
                 guidance=None
                 ):

        self.llm = llm
        self.guidance=guidance
        self.name = name
        self.age = age
        self.description = description.split(';')
        self.summary = traits
        self.plan = []
        self.status = None
        embed_size = 384
        index = faiss.IndexFlatL2(embed_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=score_normalizer)
        self.retriever = TimeWeightedVectorStoreRetrieverModified(vectorstore=vectorstore, other_score_keys=['importance'], k=15)
        self.current_time = current_time
        if self.current_time is None:
            self.last_refreshed = datetime.now()
        else:
            self.last_refreshed = current_time
        self.summary_refresh_seconds = 3600
        self.aggregate_importance = 0
        self.reflecting = False
        self.reflection_threshold = 25
        self.dialogue_list = []
        self.verbose = False

        self.add_memories(self.description)

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def set_current_time(self, time):
        self.current_time = time

    def get_current_time(self):
        return self.current_time if self.current_time is not None else datetime.now()

    def add_memories(self, memories):
        """Add memories to retriever and reflecting if aggregated importance is higher than threshold"""
        for memory in memories:
            description_, time_ = None, None
            if isinstance(memory, dict):
                description_, time_ = memory
            else:
                description_ = memory
                time_ = self.get_current_time()
            
            memory_importance_score = self._score_memory_importance(description_)

            self.retriever.add_documents([
                Document(
                    page_content=description_, 
                    metadata={
                    "importance": memory_importance_score,
                    "created_at": time_
                    }),
                ],
                current_time=time_
            )
            self.aggregate_importance += memory_importance_score

        if (
            not self.reflecting
            and self.aggregate_importance > self.reflection_threshold
        ):
            self.reflecting = True
            self._reflection()
            self.aggregate_importance = 0.0
            self.reflecting = False
            
    def _score_memory_importance(self, memory_content):
        """Score the absolute importance of the given memory"""
        prompt = self.guidance(PROMPT_ADDMEM, silent=self.verbose)
        result = prompt(memory_content=memory_content)
        score = int(result['rate'])
        return score

    def _reflection(self):
        list_salient = self._get_salient()
        documents = []
        for salient in list_salient:
            relevant_documents = self.retriever.get_relevant_documents(salient, self.get_current_time())
            documents = merge_documents(documents, relevant_documents)
        insights = self._get_insights(documents)
        self.add_memories(insights)

    def _get_salient(self):
        number_of_recent_memories = 20
        recent_memories = self.retriever.memory_stream[-number_of_recent_memories:]
        recent_memories_as_text = get_text_from_docs(recent_memories, include_time = True)
        prompt = self.guidance(PROMPT_SALIENT, silent=self.verbose)
        result = prompt(recent_memories=recent_memories)
        return result['items']
    
    def _get_insights(self, documents):
        documents_ = documents
        statements = get_text_from_docs(documents_, include_time = False)
        prompt = self.guidance(PROMPT_INSIGHTS, silent=self.verbose)
        result = prompt(statements=statements)
        return result['items']

    def make_plan(self):
        now = self.get_current_time().strftime('%H:%M')
        prompt = self.guidance(PROMPT_PLAN, silent=self.verbose)
        result = prompt(summary=self.summary,
                        name=self.name,
                        now=now,
                        current_time=self.get_current_time().strftime('%A %B %d, %Y, %H:%M')
                        )

        current_time = self.get_current_time()
        tasks = 


