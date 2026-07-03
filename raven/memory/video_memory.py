from dataclasses import dataclass
from PIL import Image
from raven.memory.memory import Memory, MemoryItem
from .memory import VLMMemoryItem

FIXED_SUBTRACT=1721761000 # this is just a large value that brings us closed to 1970


@dataclass
class ImageMemoryItem(MemoryItem):
    time: float
    position: list
    theta: float
    image: Image.Image


class VideoMemory(Memory):

    def __init__(self, fps=1, start_time=0):
        self.memory = []
        self.last_memory_time = 0
        self.fps = fps
        self.start_time = start_time

    def insert(self, item: ImageMemoryItem | VLMMemoryItem):
        item.time -= self.start_time
        self.memory.append(item)

    def reset(self):
        self.memory = []

    def get_working_memory(self) -> list[ImageMemoryItem | VLMMemoryItem]:
        return self.memory



def format_memory(memory_item_list: list[MemoryItem]):
    return memory_item_list
