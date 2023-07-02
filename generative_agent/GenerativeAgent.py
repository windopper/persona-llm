# implementation of Generative Agents: Interactive Simulacra of Human Behavior
# references: https://arxiv.org/pdf/2304.03442.pdf
# Based on
# https://github.com/hwchase17/langchain/blob/master/langchain/experimental/generative_agents/generative_agent.py
import logging
import math
import faiss
import re
import numpy as np
from datetime import datetime, timedelta
from typing import List
from generative_agent.TimeWeightedVectorStoreRetrieverModified import (
    TimeWeightedVectorStoreRetrieverModified,
)
from generative_agent.prompt import *

from langchain import LLMChain
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


def score_normalizer(val: float) -> float:
    return 1 - 1 / (1 + np.exp(val))


def get_text_from_docs(documents, include_time=False):
    ret = ""
    for i, document in enumerate(documents):
        time = (
            document.metadata["created_at"].strftime("%A %B %d, %Y, %H:%M") + ": "
            if include_time
            else ""
        )
        ret += "- " if i == 0 else "\n- "
        ret += time + document.page_content

    return ret


def merge_documents(documents1, documents2):
    """"""
    buffer_indexes = []
    merged_documents = []
    for document in documents1:
        buffer_indexes.append(document.metadata["buffer_idx"])
        merged_documents.append(document)
    for document in documents2:
        if document.metadata["buffer_idx"] not in buffer_indexes:
            merged_documents.append(document)
    return merged_documents


class GenerativeAgent:
    def __init__(
        self,
        name: str,
        age: str,
        description: str,
        traits: str,
        embeddings_model,
        current_time=None,
        llm=None,
        guidance=None,
        verbose=False,
    ):
        self.llm = llm

        self.guidance = guidance
        self.name = name
        self.age = age
        self.description = description.split(";")
        self.summary = traits
        self.plan = []
        self.status = None
        embed_size = 1536
        index = faiss.IndexFlatL2(embed_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=score_normalizer,
        )
        self.retriever = TimeWeightedVectorStoreRetrieverModified(
            vectorstore=vectorstore,
            other_score_keys=["importance"],
            k=10,
            decay_rate=0.01,
        )
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
        self.verbose = verbose

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
            if isinstance(memory, dict):
                description_, time_ = memory
            else:
                description_ = memory
                time_ = self.get_current_time()

            memory_importance_score = self._score_memory_importance(description_)

            self.retriever.add_documents(
                [
                    Document(
                        page_content=description_,
                        metadata={
                            "importance": memory_importance_score,
                            "created_at": time_,
                        },
                    ),
                ],
                current_time=time_,
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
        chain = self.guidance(
            PROMPT_ADDMEM,
            silent=self.verbose,
        )
        result = chain(memory_content=memory_content)

        score = int(result["rate"])
        return score

    def _reflection(self):
        list_salient = self._get_salient()
        documents = []
        for salient in list_salient:
            relevant_documents = self.retriever.get_relevant_documents(
                salient, self.get_current_time()
            )
            documents = merge_documents(documents, relevant_documents)
        insights = self._get_insights(documents)
        self.add_memories(insights)

    def _get_salient(self):
        number_of_recent_memories = 20
        recent_memories = self.retriever.memory_stream[-number_of_recent_memories:]
        recent_memories_as_text = get_text_from_docs(recent_memories, include_time=True)
        prompt = self.guidance(PROMPT_SALIENT, silent=self.verbose)
        result = prompt(recent_memories=recent_memories_as_text)
        return result["items"]

    def _get_insights(self, documents):
        documents_ = documents
        statements = get_text_from_docs(documents_, include_time=False)
        prompt = self.guidance(PROMPT_INSIGHTS, silent=self.verbose)
        result = prompt(statements=statements)
        return result["items"]

    def update_status(self):
        current_time = self.get_current_time()
        need_replan = True
        for task in self.plan:
            task_to = task["to"]
            if task_to > current_time:
                self.status = task["task"]
                need_replan = False

        if need_replan:
            new_plan = self.make_plan()
            self.status = new_plan[0]["task"]

        return self.status

    def make_plan(self):
        now = self.get_current_time().strftime("%H:%M")
        prompt = self.guidance(PROMPT_PLAN, silent=self.verbose)
        result = prompt(
            summary=self.summary,
            name=self.name,
            now=now,
            current_time=self.get_current_time().strftime("%A %B %d, %Y, %H:%M"),
        )

        current_time = self.get_current_time()

        tasks = result["vs"]
        tasks.insert(
            0,
            {
                "from": now,
                "to": re.findall(r"[0-9]+:[0-9][0-9]", result["to"])[0],
                "task": result["task"],
            },
        )
        tasks_time = []

        for i, task in enumerate(tasks):
            print(task)
            task["from"] = re.findall(r"[0-9]+:[0-9][0-9]", task["from"])[0]
            task_from = datetime.strptime(task["from"], "%H:%M")
            task_from = current_time.replace(
                hour=task_from.hour, minute=task_from.minute
            )
            task["to"] = re.findall(r"[0-9]+:[0-9][0-9]", task["to"])[0]
            task_to = datetime.strptime(task["to"], "%H:%M")
            task_to = current_time.replace(hour=task_to.hour, minute=task_to.minute)
            delta_time = task_to - task_from
            if delta_time.total_seconds() < 0:
                task_to += timedelta(days=1)
            tasks_time.append({"from": task_from, "to": task_to, "task": task["task"]})

        self.plan = tasks_time
        return tasks_time

    def react(self, observation, observed_entity, entity_status):
        if isinstance(observed_entity, str):
            observed_entity_name = observed_entity
        else:
            observed_entity_name = observed_entity.name

        should_react, reaction, context = self._check_reaction(
            observation, observed_entity, entity_status
        )

        if should_react == "Yes":
            if isinstance(observed_entity, GenerativeAgent):
                self._start_dialogue(
                    observation, observed_entity_name, entity_status, context, reaction
                )
            new_plan = self._replan(observation, reaction)
            self.plan = new_plan
            self.update_status()

        return should_react, reaction, context

    def _check_reaction(self, observation, observed_entity, entity_status):
        context = self._get_relevant_context(observed_entity, entity_status)
        prompt = self.guidance(PROMPT_REACT, silent=self.verbose)
        result = prompt(
            summary=self.summary,
            name=self.name,
            status=self.status,
            observation=observation,
            observed_entity=observed_entity,
            context=context,
            current_time=self.get_current_time().strftime("%A %B %d, %Y, %H:%M"),
        )
        return result["reaction"], result["result"], context

    def _get_relevant_context(self, observed_entity, entity_status):
        document_1 = self.retriever.get_relevant_documents(
            f"What is {self.name}'s relationship with {observed_entity}?",
            self.get_current_time(),
        )
        document_2 = self.retriever.get_relevant_documents(
            entity_status, self.get_current_time()
        )

        merged_documents = merge_documents(document_1, document_2)
        statements = get_text_from_docs(merged_documents, include_time=False)

        prompt = self.guidance(PROMPT_CONTEXT, silent=self.verbose)
        result = prompt(
            statements=statements,
            name=self.name,
            observed_entity=observed_entity,
            entity_status=entity_status,
        )
        return result["context"]

    def _start_dialogue(
        self, observation, observed_entity_name, entity_status, context, reaction
    ):
        prompt = self.guidance(PROMPT_DIALOGUE, silent=self.verbose)

        result = prompt(
            summary=self.summary,
            status=self.status,
            observation=observation,
            reaction=reaction,
            observed_entity=observed_entity_name,
            context=context,
            current_time=self.get_current_time().strftime("%A %B %d, %Y, %H:%M"),
        )

        self.dialogue_list.append(
            f"{self.get_current_time().strftime('%A %B %d, %Y, %H:%M')}\n{result['dialogue']}"
        )
        return result["dialogue"]

    def _replan(self, observation, reaction):
        now = self.get_current_time().strftime("%H:%M")
        prompt = self.guidance(PROMPT_REPLAN, silent=self.verbose)
        result = prompt(
            summary=self.summary,
            name=self.name,
            status=self.status,
            observation=observation,
            reaction=reaction,
            now=now,
            current_time=self.get_current_time().strftime("%A %B %d, %Y, %H:%M"),
        )
        tasks = result["items"]
        tasks.insert(0, {"from": now, "to": result["to"], "task": result["task"]})

        current_time = self.get_current_time()
        tasks_time = []

        for i, task in enumerate(tasks):
            task_from = datetime.strptime(task["from"], "%H:%M")
            task_from = current_time.replace(
                hour=task_from.hour, minute=task_from.minute
            )
            task_to = datetime.strptime(task["to"], "%H:%M")
            task_to = current_time.replace(hour=task_to.hour, minute=task_to.minute)
            delta_time = task_to - task_from
            if delta_time.total_seconds() < 0:
                task_to += timedelta(days=1)
            tasks_time.append({"from": task_from, "to": task_to, "task": task["task"]})

        return tasks_time
