
import sys, os
import re
from raven.embedder.embedders import VLMEmbeddings
from raven.agents.agent import Agent, AgentOutput
from raven.memory.faiss_memory_vlm import FAISSVLMMemory
from raven.agents.agent import Agent, AgentOutput

class EmbedderOnlyAgent(Agent):
    def __init__(self, embedder:VLMEmbeddings=None):
        self.embedder = embedder


    def set_memory(self, memory: FAISSVLMMemory):
        self.memory = memory


    def query(self, question: str) -> AgentOutput:
        response = self.memory.search_by_text(question)
        try:
            match_path = re.search(r'\[IMG\](.*?)\[/IMG\]', response).group(1)
            frame_num = int(re.search(r'(\d+)(?=\.(jpg|png|jpeg|bmp|webp)$)', match_path, re.IGNORECASE).group(1))
            return frame_num
        except Exception as e:
            return None