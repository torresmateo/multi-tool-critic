import os

# Import necessary classes and modules
from typing import Callable, Any, List, Dict
from langchain_arcade import ToolManager
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from utils.hitl_commons import yes_no_loop
import pprint

from dotenv import load_dotenv


load_dotenv()


arcade_api_key = os.environ["ARCADE_API_KEY"]


class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str


class ToolPlan(BaseModel):
    tool_calls: List[ToolCall]
    overall_reasoning: str


class GraphState(TypedDict):
    messages: List[Any]
    plan: str
    plan_approved: bool
    execution_complete: bool


def add_human_in_the_loop(
    target_tool: Callable | BaseTool,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""
    if not isinstance(target_tool, BaseTool):
        target_tool = tool(target_tool)

    @tool(
        target_tool.name,
        description=target_tool.description,
        args_schema=target_tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):

        arguments = pprint.pformat(tool_input, indent=4)
        response = interrupt(
            f"Do you allow the call to {target_tool.name} with arguments:\n"
            f"{arguments}"
        )

        # approve the tool call
        if response == "yes":
            tool_response = target_tool.invoke(tool_input, config)
        # deny tool call
        elif response == "no":
            tool_response = "The User did not allow the tool to run"
        else:
            raise ValueError(
                f"Unsupported interrupt response type: {response}"
            )

        return tool_response

    return call_tool_with_interrupt


ENFORCE_HUMAN_CONFIRMATION = [
    "Google_SendEmail",
    "Slack_SendDmToUser",
]


def create_planner_agent(tools: List[BaseTool]):
    """Create a planner agent that generates tool execution plans."""

    # Create tool descriptions for the planner
    tool_descriptions = []
    for t in tools:
        tool_desc = f"- {t.name}: {t.description}"
        if hasattr(t, 'args_schema') and t.args_schema:
            # Get schema info
            schema = t.args_schema.model_json_schema()
            if 'properties' in schema:
                args_info = []
                for prop_name, prop_info in schema['properties'].items():
                    prop_type = prop_info.get('type', 'unknown')
                    prop_desc = prop_info.get('description', '')
                    args_info.append(f"{prop_name} ({prop_type}): {prop_desc}")
                tool_desc += f"\n  Arguments: {', '.join(args_info)}"
        tool_descriptions.append(tool_desc)

    tools_text = "\n".join(tool_descriptions)

    planner_prompt = f"""You are a tool planner. Your job is to create a sequence of tool calls to accomplish a given task.

Available tools:
{tools_text}

When given a task, you should:
1. Analyze what needs to be done
2. Determine which tools are needed and in what order
3. Generate a structured plan with tool calls

Respond with a JSON object following this schema:
{{
    "tool_calls": [
        {{
            "tool_name": "name_of_tool",
            "arguments": {{"arg1": "value1", "arg2": "value2"}},
            "reasoning": "why this tool call is needed"
        }}
    ],
    "overall_reasoning": "explanation of the overall plan"
}}

Be precise with tool names and argument types. Only use tools that are available in the list above."""

    return create_react_agent(
        model="openai:gpt-4o",
        tools=[],  # Planner doesn't execute tools, just plans
        prompt=planner_prompt,
        name="planner_agent"
    )


def create_critic_agent(tools: List[BaseTool]):
    """Create a critic agent that validates tool execution plans."""

    # Create detailed tool signatures for the critic
    tool_signatures = []
    for t in tools:
        signature = f"- {t.name}: {t.description}"
        if hasattr(t, 'args_schema') and t.args_schema:
            schema = t.args_schema.model_json_schema()
            if 'properties' in schema:
                required = schema.get('required', [])
                args_info = []
                for prop_name, prop_info in schema['properties'].items():
                    prop_type = prop_info.get('type', 'unknown')
                    prop_desc = prop_info.get('description', '')
                    is_required = prop_name in required
                    req_marker = " (required)" if is_required else " (optional)"
                    args_info.append(f"{prop_name} ({prop_type}){req_marker}: {prop_desc}")
                signature += f"\n  Arguments: {'; '.join(args_info)}"
        tool_signatures.append(signature)

    signatures_text = "\n".join(tool_signatures)

    critic_prompt = f"""You are a tool execution critic. Your job is to validate tool execution plans.

Available tool signatures:
{signatures_text}

When given a tool execution plan, you should:
1. Check if all tool names exist and are spelled correctly
2. Validate that all required arguments are provided
3. Check if argument types match expected types
4. Verify the logical sequence makes sense
5. Identify any potential issues or missing steps

Respond with either:
- "APPROVED: [brief explanation]" if the plan is valid
- "REJECTED: [detailed explanation of issues]" if there are problems

Be thorough in your analysis and explain your reasoning clearly."""

    return create_react_agent(
        model="openai:gpt-4o",
        tools=[],  # Critic doesn't execute tools, just validates
        prompt=critic_prompt,
        name="critic_agent"
    )


def create_executor_agent(tools: List[BaseTool]):
    """Create an executor agent that runs the approved tools."""

    executor_prompt = """You are a tool executor. Your job is to execute approved tool plans.

When given an approved plan, execute the tools in the specified order with the provided arguments.
Be precise and follow the plan exactly as specified."""

    return create_react_agent(
        model="openai:gpt-4o",
        tools=tools,
        prompt=executor_prompt,
        name="executor_agent"
    )


def planner_node(agent: CompiledStateGraph):
    def node(state: GraphState):
        """Node that generates a plan using the planner agent."""
        messages = state["messages"]

        # Get the latest user message
        user_message = messages[-1]["content"] if messages else ""

        # Generate plan
        plan_response = agent.invoke({"messages": [HumanMessage(content=user_message)]})
        plan = plan_response["messages"][-1].content

        return {
            "messages": messages + [{"role": "assistant", "content": f"Generated plan: {plan}"}],
            "plan": plan,
            "plan_approved": False,
            "execution_complete": False
        }

    return node


def critic_node(agent: CompiledStateGraph):
    def node(state: GraphState):
        """Node that validates the plan using the critic agent."""
        plan = state["plan"]
        messages = state["messages"]

        # Validate plan
        critic_response = agent.invoke({"messages": [HumanMessage(content=f"Please validate this plan: {plan}")]})
        validation = critic_response["messages"][-1].content

        # Check if approved
        approved = validation.startswith("APPROVED:")

        return {
            "messages": messages + [{"role": "assistant", "content": f"Plan validation: {validation}"}],
            "plan": plan,
            "plan_approved": approved,
            "execution_complete": False
        }

    return node


def executor_node(agent: CompiledStateGraph):
    def node(state: GraphState):
        print(f"Executing plan: {state}")
        """Node that executes the approved plan."""
        plan = state["plan"]
        messages = state["messages"]

        # Execute plan
        execution_response = agent.invoke({"messages": [HumanMessage(content=f"Execute this approved plan: {plan}")]})
        execution_result = execution_response["messages"][-1].content

        return {
            "messages": messages + [{"role": "assistant", "content": f"Execution result: {execution_result}"}],
            "plan": plan,
            "plan_approved": True,
            "execution_complete": True
        }

    return node


def should_execute(state: GraphState) -> str:
    """Conditional edge function to determine if plan should be executed."""
    if state["plan_approved"]:
        return "execute"
    else:
        return "replan"


def create_workflow_graph(tools: List[BaseTool], memory: MemorySaver):
    """Create the explicit workflow graph."""

    # Create agents
    planner_agent = create_planner_agent(tools)
    critic_agent = create_critic_agent(tools)
    executor_agent = create_executor_agent(tools)

    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("planner", planner_node(planner_agent))
    workflow.add_node("critic", critic_node(critic_agent))
    workflow.add_node("executor", executor_node(executor_agent))

    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_execute,
        {
            "execute": "executor",
            "replan": "planner"
        }
    )
    workflow.add_edge("executor", END)

    # Store agents in the compiled graph for access in nodes
    compiled_graph = workflow.compile(checkpointer=memory)
    compiled_graph.planner_agent = planner_agent
    compiled_graph.critic_agent = critic_agent
    compiled_graph.executor_agent = executor_agent

    return compiled_graph


def run_graph(graph: CompiledStateGraph, config, input: Any):
    for event in graph.stream(input, config=config, stream_mode="values"):
        if "messages" in event:
            # Pretty-print the last message
            message = event['messages'][-1]
            print(f'{message["role"]}: {pprint.pformat(message["content"])}')


def handle_interrupts(graph: CompiledStateGraph, config):
    for interr in graph.get_state(config).interrupts:
        approved = yes_no_loop(interr.value)
        run_graph(graph, config, Command(resume=approved))


if __name__ == "__main__":
    user_id = "mateo@arcade.dev"
    config = {"configurable": {"thread_id": "4",
                               "user_id": user_id}}
    # Set up memory for checkpointing the state
    memory = MemorySaver()

    # Initialize tools
    manager = ToolManager(api_key=arcade_api_key, api_url="asd")
    manager.init_tools(toolkits=["google", "slack"])

    for t in manager.tools:
        manager.authorize(tool_name=t, user_id=user_id)

    # Prepare all tools with human-in-the-loop where needed
    all_tools = []
    for t in manager.to_langchain(use_interrupts=True):
        print(t.name)
        if t.name in ENFORCE_HUMAN_CONFIRMATION:
            print(f"Adding hitl to {t.name}")
            all_tools.append(add_human_in_the_loop(t))
        else:
            all_tools.append(t)

    # Create the workflow graph
    workflow_graph = create_workflow_graph(all_tools, memory)

    first_prompt = "read my latest 3 emails, then summarize them, then read the #general channel on Slack, then send me (Mateo) a DM on Slack with a summary of the latest 10 messages in that channel"
    managed_first = False

    while True:
        if not managed_first:
            user_input = first_prompt
            managed_first = True
        else:
            user_input = input("User: ")

        if user_input.lower() == "exit":
            break

        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "plan": "",
            "plan_approved": False,
            "execution_complete": False,
        }

        run_graph(workflow_graph, config, initial_state)

        # handle all interrupts in case there's any
        handle_interrupts(workflow_graph, config)
