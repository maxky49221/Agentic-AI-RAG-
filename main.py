# # pip install -qU langchain "langchain[openai]"
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from embedding import retriever
from langchain_core.messages import AIMessage

load_dotenv()

model = init_chat_model(
    "azure_openai:gpt-5-mini",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

def data_retriever(state: MessagesState):
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    return {"messages": [AIMessage(content=context)]}

def report_agent(state: MessagesState):
    user_query = state["messages"][0].content
    last_response = state["messages"][-1].content
    system = SystemMessage(content=
                        f"""Question from user: {user_query}\n
                        Context from source:\n{last_response} 
                        \nProvide a cohesive, non-redundant, well-formatted answer.""")
    
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node(data_retriever)
graph.add_node(report_agent)
graph.add_edge(START, "data_retriever")
graph.add_edge("data_retriever", "report_agent")
graph.add_edge("report_agent", END)

graph = graph.compile()

user_input = input("You: ")
result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

print(result["messages"][-1].content)