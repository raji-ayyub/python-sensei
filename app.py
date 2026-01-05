import os
import sqlite3
import pdfplumber
import pandas as pd
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()


class PythonAssistant:
    def __init__(self, pdf_directory="memory", db_path="memory/docstore"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.pdf_directory = pdf_directory
        self.db_path = db_path

        # Initialize models
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key
        )

        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Tools and graph
        self.tools = self._setup_tools()
        self.tool_node = ToolNode(self.tools)
        self.builder = self._build_graph()

        # Initialize SQLite checkpointer once
        self.checkpointer = self._initialize_checkpointer()


    def _initialize_checkpointer(self):
        try:
            os.makedirs("memory", exist_ok=True)
            db_path = os.path.abspath("memory/conversations.db")
            print(f"SQLite database: {db_path}")

            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)

            print("SQLite checkpointer initialized successfully")
            return checkpointer

        except Exception as e:
            print(f"Failed to initialize SQLite checkpointer: {str(e)}")
            raise


    def _setup_tools(self):

        @tool
        def retriever_tool(query: str) -> str:
            """retrieves python related informations."""
            docs = self.retriever.invoke(query)
            return "\n\n---\n\n".join([
                f"Source: {d.metadata.get('source', 'Unknown')} | Page {d.metadata.get('page', 'N/A')}\n\n{d.page_content[:1000]}..."
                for d in docs
            ])
        
        return [retriever_tool]



    def _load_all_pdfs(self):
        documents = []
        if not os.path.exists(self.pdf_directory):
            os.makedirs(self.pdf_directory)
            return documents

        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith(".pdf"):
                with pdfplumber.open(os.path.join(self.pdf_directory, filename)) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": filename, "page": i + 1}
                            ))
        return documents


    def _initialize_vector_store(self):
        os.makedirs(self.db_path, exist_ok=True)

        if os.path.exists(os.path.join(self.db_path, "chroma.sqlite3")):
            return Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )

        docs = self._load_all_pdfs()
        if not docs:
            return Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )


    def _build_graph(self):

        def llm_node(state: MessagesState):
            system = SystemMessage(content="You are a python learning assistant")
            messages = [system] + state["messages"]
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_use_tools(state: MessagesState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(MessagesState)
        builder.add_node("assistant", llm_node)
        builder.add_node("tools", self.tool_node)

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", should_use_tools)
        builder.add_edge("tools", "assistant")

        return builder


    def ask_question(self, question: str, user_id: str = "default"):
        try:
            agent = self.builder.compile(checkpointer=self.checkpointer)
            result = agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                {"configurable": {"thread_id": user_id}}
            )
            return result["messages"][-1].content
        except Exception:
            response = self.llm.invoke([HumanMessage(content=question)])
            return response.content


assistant = None

def get_python_assistant():
    global assistant
    if assistant is None:
        assistant = PythonAssistant()
    return assistant
