import sys, re
import base64
from typing import List, Dict, Any
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
sys.path.append(sys.path[0] + '/..')
from raven.agents.remembr_agent import ReMEmbRAgent

IMG_PATTERN = re.compile(r"\[IMG\](.*?)\[/IMG\]")

class RAVENAgent(ReMEmbRAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_descriptions = {
            "retrieve_from_text":
                "Search and return multimodal information from your video memory in the form of a list of closest images with their similarity scores",
            "retrieve_from_position":
                "Search and return multimodal information from your video memory by using a position array such as (x,y,z)",
            "retrieve_from_time":
                "Search and return multimodal information from your video memory by using an H:M:S time."
        }


    ### Overridden methods for nodes

    def agent_message_decorator(self, messages):
        """
        Workaround for Ollama-like backends that can't handle ToolMessage with images.

        - For ToolMessage containing [IMG]...[/IMG], we:
        1) Replace tool message content with a short placeholder text
        2) Append a HumanMessage that contains a multimodal payload (text + image_url)
        3) Add a short AIMessage before those HumanMessages
        """
        self._debug_print_image_process(messages, before=True)

        tool_call_num = 0
        image_messages = []

        def is_ollama_backend() -> bool:
            llm = getattr(self, "llm_type", "") or ""
            return ("gpt" not in llm) and ("nim" not in llm)

        def to_data_uri(image_file_path: str) -> str:
            if image_file_path.endswith(".pkl") or ".pkl" in image_file_path:
                from raven.embedder.embedders import file_to_data_url
                return file_to_data_url(image_file_path)

            with open(image_file_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            # TODO: if you can have png/webp, infer mime from extension.
            return f"data:image/jpeg;base64,{img_base64}"

        def build_multimodal_chunk(tool_idx: int, segment: str, img_path: str) -> List[Dict[str, Any]]:
            # Remove the full [IMG]...[/IMG] markup from the segment
            cleaned = IMG_PATTERN.sub("", segment)
            text = f"Tool call {tool_idx}: {cleaned}".strip()
            return [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": to_data_uri(img_path)}},
            ]

        for idx, msg in enumerate(messages):
            # Only process tool messages with [IMG]
            if type(msg) != ToolMessage:
                continue
            if not getattr(msg, "content", None):
                continue
            if "[IMG]" not in msg.content:
                continue

            old_content = msg.content

            # If content is not a string: optionally convert ToolMessage -> AIMessage for ollama
            if not isinstance(old_content, str):
                print(f"Ollama Message Processing: type(content)={type(old_content)}, content={old_content}")
                if is_ollama_backend():
                    print("Turn ToolMessage to AIMessage...")
                    messages[idx] = AIMessage(id=msg.id, content=old_content)  # ignore tool_call_id
                continue

            tool_call_num += 1

            # Build multimodal payload by scanning segments
            updated_contents: List[Dict[str, Any]] = []
            for segment in old_content.split("\n\n"):
                m = IMG_PATTERN.search(segment)
                if not m:
                    continue
                img_path = m.group(1).strip()
                updated_contents.extend(build_multimodal_chunk(tool_call_num, segment, img_path))

            # Mutate the tool message in-place: keep it as a short placeholder
            msg.content = f"Tool call {tool_call_num} results will be shown by the human user."

            # Add the actual multimodal content as a HumanMessage to the end
            if updated_contents:
                image_messages.append(HumanMessage(id=msg.id, content=updated_contents))

        if image_messages:
            messages.append(AIMessage(content="I see. Could the user provide more details?"))
            messages.extend(image_messages)

        self._debug_print_image_process(messages, before=False)
        return messages

    
    def _debug_print_image_process(self, messages, before=True):
        if not self.debug:
            return
        if before:
            pattern = r"\{'type': 'image_url', 'image_url': \{'url': 'data:image/jpeg;base64[^']*'\}\}"
            replacement = "{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64'}}"
            message_dump = [re.sub(pattern, replacement, str(message)) for message in messages]
            print(f"\n========================Start Processing IMG=======================")
            print("message dump:", str(message_dump)[:1000])
            print("Number of messages before processed:", len(messages))
        else:
            print("Number of messages after processed:", len(messages))
            print(f"========================End Processing IMG=======================")