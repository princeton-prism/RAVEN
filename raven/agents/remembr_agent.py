from typing import Annotated, Sequence, TypedDict
import traceback
import sys, re, os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools import StructuredTool

from pydantic import BaseModel, Field
sys.path.append(sys.path[0] + '/..')

from raven.utils.util import file_to_string, json_fixer, parse_json, safe_literal
from raven.tools.functions_wrapper import FunctionsWrapper
from raven.memory.memory import Memory
from raven.agents.agent import Agent, AgentOutput

# Default fallback response for generate node when response is in wrong format and retry does not work (e.g., too low temperature); 
# this can be parsed into the correct format, but contains all "None" or default values. 
DEFAULT_NULL_RESPONSE_FALLBACK = """```json
{
    "type_reasoning": "None", 
    "type": "None",
    "answer_reasoning": "None", 
    "text": "None",
    "binary": "yes/no",
    "position": "[0,0,0]", 
    "orientation": "[0,0,0]", 
    "time": "HH:MM:SS",
    "duration": "0.0"
}
```"""

# Default fallback response for generate node when max retries exceeded
DEFAULT_GENERATE_RESPONSE_FALLBACK = {
    "messages": [str({
        "type_reasoning": "Max retries exceeded", 
        "type": "None",
        "answer_reasoning": "Max retries exceeded", 
        "text": "None",
        "binary": "no",
        "position": [0, 0, 0], 
        "orientation": [0, 0, 0], 
        "time": "00:00:00",
        "duration": "0.0"
    })]
}

# Default fallback response for agent node when max retries exceeded; 
# in graph this will lead to generate node which can still try to produce an answer without retrieval
DEFAULT_AGENT_RESPONSE_FALLBACK = {
    "messages": [AIMessage(content="I will now provide a final answer based on my knowledge.")]
}

# Default max retry limit for try_except_continue
DEFAULT_MAX_RETRIES = 5

### Print out state of the system
def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    for k,v in state.items():
        if type(v) == str:
            print(v)
        elif type(v) == list:
            for item in v:
                if type(item) == str:
                    print(item)
                else:
                    print(item)
        else:
            print(v)
    return state


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

def try_except_continue(state, func, max_retries=DEFAULT_MAX_RETRIES, fallback_response=None):
    """
    Retry a function up to max_retries times. If all retries fail, return the fallback_response.
    
    Args:
        state: The current state to pass to the function
        func: The function to execute
        max_retries: Maximum number of retry attempts (default: DEFAULT_MAX_RETRIES)
        fallback_response: The response to return if all retries are exhausted
        
    Returns:
        The function result, or fallback_response if max retries exceeded
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            ret = func(state)
            return ret
        except Exception as e:
            retry_count += 1
            print(f"I crashed trying to run: {func} (attempt {retry_count}/{max_retries})")
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
            if retry_count < max_retries:
                print(f"Retrying... ({max_retries - retry_count} attempts remaining)")
            continue
    
    # Max retries exceeded, return fallback response
    print(f"Max retries ({max_retries}) exceeded for {func}. Returning fallback response.")
    if fallback_response is not None:
        return fallback_response
    else:
        raise RuntimeError(f"Max retries ({max_retries}) exceeded and no fallback response provided.")


class ReMEmbRAgent(Agent):

    def __init__(self, llm_type='gpt-4o', num_ctx=16384, num_gen_tokens=8192, temperature=0, debug=False, prompt_dir="unknown", max_tool_calls=3):
        self.debug = debug
        # Setup language models
        llm = self.llm_selector(llm_type, temperature, num_ctx, num_gen_tokens)
         # Wrapper that handles everything
        chat = FunctionsWrapper(llm, debug=debug, temperature=temperature)
        self.prompt_dir = prompt_dir
        # Set prompts from given directory
        self.prompt_selector()
        self.max_tool_calls = max_tool_calls
        self.num_ctx = num_ctx
        self.num_gen_tokens = num_gen_tokens
        self.temperature = temperature
        self.chat = chat
        self.llm_type = llm_type

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.chat_history = ChatMessageHistory()

        self.tool_args_descriptions = {
            "retrieve_from_text":
                "The query that will be searched by the vector similarity-based retriever.\
                Text embeddings of this description are used. There should always be text in here as a response! \
                Based on the question and your context, decide what text to search for in the database. \
                This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
                The query will then search your memories for you.",
            "retrieve_from_position":
                "The query that will be searched by finding the nearest memories at this (x,y,z) position.\
                The query must be an (x,y,z) array with floating point values \
                Based on the question and your context, decide what position to search for in the database. \
                This query argument should be a position such as (0.5, 0.2, 0.1). They should NOT be a string. \
                The query will then search your memories for you.",
            "retrieve_from_time":
                "The query that will be searched by finding the nearest memories at a specific time in H:M:S format.\
                The query must be a string containing only time. \
                Based on the question and your context, decide what time to search for in the database. \
                This query argument should be an HMS time such as 08:02:03 with leading zeros. \
                The query will then search your memories for you."
        }

        self.tool_descriptions = {
            "retrieve_from_text":
                "Search and return information from your video memory in the form of captions",
            "retrieve_from_position":
                "Search and return information from your video memory by using a position array such as (x,y,z)",
            "retrieve_from_time":
                "Search and return information from your video memory by using an H:M:S time."
        }

    def prompt_selector(self):
        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.agent_prompt = file_to_string(top_level_path+f'{self.prompt_dir}/agent_system_prompt.txt')
        self.generate_prompt = file_to_string(top_level_path+f'{self.prompt_dir}/generate_system_prompt.txt')
        self.agent_gen_only_prompt = file_to_string(top_level_path+f'{self.prompt_dir}/agent_gen_system_prompt.txt')


    def llm_selector(self, llm_type, temperature, num_ctx, num_gen_tokens):
        llm = None
        # Support for LLM Gateway
        if 'gpt' in llm_type:
            llm = ChatOpenAI(model=llm_type, temperature=temperature, max_tokens=num_gen_tokens)
        elif 'gemini' in llm_type:
            llm = ChatGoogleGenerativeAI(model=llm_type, temperature=temperature, max_output_tokens=num_gen_tokens)
        # Qwen-VL  
        elif ('qwen' in llm_type and 'vl' in llm_type) or ('gemma3' in llm_type.lower()):
            llm = ChatOllama(model=llm_type, temperature=temperature, num_ctx=num_ctx, base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        else:
            llm = ChatOllama(model=llm_type, format="json", temperature=temperature, num_ctx=num_ctx)

        if llm is None:
            raise Exception("No correct LLM provided")

        return llm


    def set_memory(self, memory: Memory):
        self.memory = memory
        self.create_tools(memory)
        self.build_graph()


    def create_tools(self, memory):
        # text-based tool
        class TextRetrieverInput(BaseModel):
            x: str = Field(description=self.tool_args_descriptions['retrieve_from_text'])

        self.retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_text(x),
            name="retrieve_from_text",
            description=self.tool_descriptions['retrieve_from_text'],
            args_schema=TextRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class PositionRetrieverInput(BaseModel):
            x: tuple = Field(description=self.tool_args_descriptions['retrieve_from_position'])
        # position-based tool
        self.position_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_position(x),
            name="retrieve_from_position",
            description=self.tool_descriptions['retrieve_from_position'],
            args_schema=PositionRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class TimeRetrieverInput(BaseModel):
            x: str = Field(description=self.tool_args_descriptions['retrieve_from_time'])

        # time-based tool
        self.time_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_time(x),
            name="retrieve_from_time",
            description=self.tool_descriptions['retrieve_from_time'],
            args_schema=TimeRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        self.tool_list = [self.retriever_tool, self.position_retriever_tool, self.time_retriever_tool]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]


    ### Overridden methods for nodes

    def agent_message_decorator(self, messages):
        """
        A decorator to process messages before sending to the agent LLM.
        This can be overridden in subclasses to modify the messages as needed.
        """
        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        if ('gpt-' not in self.llm_type) and ('nim' not in self.llm_type):
            for i in range(len(messages)):
                if type(messages[i]) == ToolMessage:
                    messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id

        return messages

    ### Nodes

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]

        model = self.chat
        model.bind(debug=self.debug)

        # Limit tool call number to max_tool_calls.
        if self.agent_call_count < self.max_tool_calls:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        else:
            prompt = self.agent_gen_only_prompt


        agent_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),
            ]
        )

        model = agent_prompt | model

        question = f"The question is: {messages[0]}"

        messages = self.agent_message_decorator(messages)

        response = model.invoke({"question": question, "chat_history": messages[:]})

        self._debug_print_agent(question, messages, response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub("\{.*?\}", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"
        else:
            # invalid response, calling generate directly if low temperature, otherwise retry
            if self.temperature < 0.5 and not response.content:
                response = AIMessage(content="I will now provide a final answer based on my knowledge.")
                
        self.agent_call_count += 1

        return {"messages": [response]}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content + "\n Please responsed in the desired format."
        last_message = str(messages[-1].content)

        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
            
        filled_prompt = prompt.invoke({'question':question})

        gen_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}")
            ]
        )
        
        # if eval NavQA use ollama qwen2.5-vl-3b-instruct model, use following
        # model = gen_prompt | self.chat.llm

        # otherwise use this
        model = gen_prompt | self.chat

        chat_history = messages[:-1]
        
        response = model.invoke({"question": question, "chat_history": chat_history, "context": last_message})

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())
        
        self._debug_print_generate(last_message, question, filled_prompt, chat_history, response)

        def parse_gen_json(response):
            response = json_fixer(response)
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = safe_literal(response)
            else:
                parsed = parse_json(response)

            # then check it has all the required keys
            keys_to_check_for = ["time", "text", "binary", "position", "duration"]

            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing all the required keys during generate. Retrying...")

            if type(parsed['position']) == str:
                parsed['position'] = safe_literal(parsed['position'])
            
            if (parsed['position'] is not None) and len(parsed['position']) != 3:
                raise ValueError(f"Shape of position was incorrect. {parsed['position']}. Retrying...")
            return parsed
        
        try:
            parsed = parse_gen_json(response)
        except:
            if self.temperature > 0.5:
                raise ValueError("Generate call failed. Retrying...")
            else:
                print("Met ERROR when generating response... Using default response template...")
                parsed = parse_gen_json(DEFAULT_NULL_RESPONSE_FALLBACK)

        # reset for next query
        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        return {"messages": [str(parsed)]}



    def build_graph(self):

        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node(
            "agent", 
            lambda state: try_except_continue(
                state, 
                self.agent, 
                max_retries=DEFAULT_MAX_RETRIES, 
                fallback_response=DEFAULT_AGENT_RESPONSE_FALLBACK
            )
        )  # agent
        # retrieve = ToolNode([self.retriever_tool])
        tool_node = ToolNode(self.tool_list)
        workflow.add_node("action", tool_node)
        # workflow.add_node("action", lambda state: try_except_continue(state, tool_node))


        # workflow.add_node("action", self.call_tool)

        workflow.add_node(
            "generate", 
            lambda state: try_except_continue(
                state, 
                self.generate, 
                max_retries=DEFAULT_MAX_RETRIES, 
                fallback_response=DEFAULT_GENERATE_RESPONSE_FALLBACK
            )
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not


        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_continue,
            {
                # Translate the condition outputs to nodes in our graph
                "continue": "action",
                "end": "generate",
            },
        )


        workflow.add_edge('action', 'agent')

        workflow.add_edge("generate", END)

        # Compile
        self.graph = workflow.compile()


    def query(self, question: str):

        inputs = { 
            "messages": [
                (("user", question)),
            ]
        }
        out = self.graph.invoke(inputs)
        response = out['messages'][-1]
        response = ''.join(response.content.splitlines())

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = safe_literal(response)
        else:
            parsed = parse_json(response)

        response = AgentOutput.from_dict(parsed)


        return response

    def _short(self, x, n=200):
        s = str(x)
        return s if len(s) <= n else s[:n] + " ..."

    def _debug_print_agent(self, question, messages, response=None):
        if not self.debug:
            return

        print("\n=======================Agent Call=======================")
        print({
            "question": question,
            "chat_history": [type(msg).__name__ for msg in messages]
        })

        for msg in messages:
            if isinstance(msg, ToolMessage):
                print("Tool:", self._short(msg.content))
            elif isinstance(msg, AIMessage):
                print("AI:", self._short(msg.content))
            elif isinstance(msg, HumanMessage):
                if isinstance(msg.content, list):
                    types = [x.get("type") for x in msg.content]
                    print(f"Human: {len(msg.content)} items {types}")
                else:
                    print("Human:", self._short(msg.content))

        if response is not None:
            print("\nResponse:", self._short(response))
            print("========================================================")

    def _debug_print_generate(self, last_message, question, filled_prompt, chat_history, response):
        if not self.debug:
            return

        print("\n=======================Generate Call=======================")
        print({"question": question, "chat_history": [type(msg) for msg in chat_history]})
        print("\n========Filled Prompt Call========")
        print(f"Context: {last_message[:1000]}")
        print(f"Question: {question}")
        print("\nFilled Prompt:", str(filled_prompt)[:1000])
        print("====================================") 
        print("\nResponse:", response)
        print("===========================================================")