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
        self.traits = traits
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

    def get_summary(self, force_refresh=False, now=None):
        current_time = self.get_current_time() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds

        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time

        age = self.age if self.age is not None else "N/A"
        return (
            f"Name: {self.name} (age: {age})"
            + f"\nInnate traits: {self.traits}"
            + f"\n{self.summary}"
        )

    # TODO: summary 내용 향상을 위한 프롬프트 변경 필요
    def _compute_agent_summary(self):
        prompt = self.guidance(PROMPT_SUMMARIZE, silent=self.verbose)
        documents = self.retriever.get_relevant_documents(
            self.name + "'s core characteristics", self.get_current_time()
        )
        statements = get_text_from_docs(documents, include_time = False)
        result = prompt(
            name=self.name,
            relevant_memories=statements,
        )

        return result["summary"]

    def add_memories(self, memories):
        """Add memories to retriever and reflecting if aggregated importance is higher than threshold"""

        memory_importance_scores = self._score_memories_importance(memories)

        for memory, score in zip(memories, memory_importance_scores):
            if isinstance(memory, dict):
                description_, time_ = memory
            else:
                description_ = memory
                time_ = self.get_current_time()

            self.retriever.add_documents(
                [
                    Document(
                        page_content=description_,
                        metadata={"importance": score, "created_at": time_},
                    ),
                ],
                current_time=time_,
            )
            self.aggregate_importance += score

        # for memory in memories:
        #     if isinstance(memory, dict):
        #         description_, time_ = memory
        #     else:
        #         description_ = memory
        #         time_ = self.get_current_time()

        #     memory_importance_score = self._score_memory_importance(description_)

        #     self.retriever.add_documents(
        #         [
        #             Document(
        #                 page_content=description_,
        #                 metadata={
        #                     "importance": memory_importance_score,
        #                     "created_at": time_,
        #                 },
        #             ),
        #         ],
        #         current_time=time_,
        #     )
        #     self.aggregate_importance += memory_importance_score

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

        if self.verbose:
            print("Scoring memory importance...")

        result = chain(memory_content=memory_content)

        score = int(result["rate"])
        return score

    def _score_memories_importance(self, memory_content):
        chain = self.guidance(
            PROMPT_ADDMEMS,
            silent=self.verbose,
        )
        result = chain(memory_content=memory_content)
        rates = [int(x) for x in result["rate"].strip().split(";")]

        return rates

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
                break

        if need_replan:
            new_plan = self.make_plan()
            self.status = new_plan[0]['task']

        return self.status

    def make_plan(self):
        now = self.get_current_time().strftime("%H:%M")
        prompt = self.guidance(PROMPT_PLAN, silent=self.verbose)
        current_time = self.get_current_time()

        if self.verbose:
            print("Generating Plans...")

        result = prompt(
            name=self.name,
            traits=self.traits,
            summary=self.summary,
            now=now,
            current_time=self.get_current_time().strftime("%A %B %d, %Y, %H:%M"),
        )

        # current_plan = result['current_plan']
        plans = result["plans"]

        prompt = self.guidance(PROMPT_RECURSIVELY_DECOMPOSED, silent=self.verbose)
        result = prompt(summary=self.summary, name=self.name, plans=plans)

        plans = result["plans"]
        plan_splitted = [plan for plan in plans.split("\n")]
        # print(plan_splitted)
        tasks_time = []

        # tasks = [f"{self.name} plans " + plan for plan in plans.split("\n")]
        # self.add_memories(tasks)

        for plan in plan_splitted:
            splitted = plan.split(")")
            time_ = splitted[0]
            task_ = splitted[1].strip()
            # print(time_, task_)
            from_, to_ = (_.strip() for _ in time_.split("-"))

            from_ = re.findall(r"[0-9]+:[0-9][0-9]", from_)[0]
            from_ = datetime.strptime(from_, "%H:%M")
            from_ = current_time.replace(hour=from_.hour, minute=from_.minute)

            to_ = re.findall(r"[0-9]+:[0-9][0-9]", to_)[0]
            to_ = datetime.strptime(to_, "%H:%M")
            to_ = current_time.replace(hour=to_.hour, minute=to_.minute)

            delta_time = to_ - from_

            if delta_time.total_seconds() < 0:
                to_ += timedelta(days=1)
            tasks_time.append({"from": from_, "to": to_, "task": task_})

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
        current_time = self.get_current_time()
        prompt = self.guidance(PROMPT_REPLAN, silent=self.verbose)
        result = prompt(
            summary=self.summary,
            name=self.name,
            status=self.status,
            observation=observation,
            reaction=reaction,
            current_time=self.get_current_time().strftime("%A %B %d, %Y, %H:%M"),
        )

        plans = result["plans"]
        plan_splitted = [plan for plan in plans.split("\n")]
        print(plan_splitted)
        tasks_time = []

        # tasks = [f"{self.name} plans " + plan for plan in plans.split("\n")]
        # self.add_memories(tasks)

        for plan in plan_splitted:
            splitted = plan.split(")")
            time_ = splitted[0]
            task_ = splitted[1].strip()
            # print(time_, task_)
            from_, to_ = (_.strip() for _ in time_.split("-"))

            from_ = re.findall(r"[0-9]+:[0-9][0-9]", from_)[0]
            from_ = datetime.strptime(from_, "%H:%M")
            from_ = current_time.replace(hour=from_.hour, minute=from_.minute)

            to_ = re.findall(r"[0-9]+:[0-9][0-9]", to_)[0]
            to_ = datetime.strptime(to_, "%H:%M")
            to_ = current_time.replace(hour=to_.hour, minute=to_.minute)

            delta_time = to_ - from_

            if delta_time.total_seconds() < 0:
                to_ += timedelta(days=1)
            tasks_time.append({"from": from_, "to": to_, "task": task_})

        return tasks_time

    def interview(self, user, question):
        documents = self.retriever.get_relevant_documents(question, self.get_current_time())
        context = get_text_from_docs(documents, include_time = False)

        prompt = self.guidance(PROMPT_INTERVIEW, silent=self.verbose)
        result = prompt(
            summary=self.summary,
            name=self.name,
            status=self.status,
            user=user,
            context=context,
            question=question,
            current_time = self.get_current_time().strftime('%A %B %d, %Y, %H:%M')
        )

        return result['response']
