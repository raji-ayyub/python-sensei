import os
import sqlite3
import pdfplumber
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

logging.getLogger("pdfminer").setLevel(logging.ERROR)

load_dotenv()


class PythonAssistant:
    def __init__(self, pdf_directory="files", db_path="docstore"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.pdf_directory = pdf_directory
        self.db_path = db_path

        # LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key,
        )

        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key,
        )

        # Vector store
        self.vector_store = self._initialize_vector_store()
        print(f"[VectorStore] Ready")

        # Tools
        self.tools = self._setup_tools()
        self.tool_node = ToolNode(self.tools)

        # Graph
        self.builder = self._build_graph()
        self.checkpointer = self._initialize_checkpointer()

        # Compile ONCE
        self.agent = self.builder.compile(checkpointer=self.checkpointer)

    def _initialize_checkpointer(self):
        os.makedirs("memory", exist_ok=True)
        conn = sqlite3.connect(
            os.path.abspath("memory/conversations.db"),
            check_same_thread=False,
        )
        return SqliteSaver(conn)

    def _setup_tools(self):
        @tool
        def retriever_tool(query: str) -> str:
            """Retrieve relevant Python documentation."""
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(query)

            if not docs:
                return "No relevant documentation found."

            return "\n\n---\n\n".join(
                f"Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}\n\n"
                f"{d.page_content[:1000]}"
                for d in docs
            )

        return [retriever_tool]

    def _load_all_pdfs(self):
        documents = []
        if not os.path.exists(self.pdf_directory):
            os.makedirs(self.pdf_directory)
            return documents

        for filename in os.listdir(self.pdf_directory):
            if not filename.lower().endswith(".pdf"):
                continue

            path = os.path.join(self.pdf_directory, filename)
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "page": i + 1,
                                },
                            )
                        )
        return documents

    def _initialize_vector_store(self):
        os.makedirs(self.db_path, exist_ok=True)

        try:
            return Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
            )
        except Exception:
            docs = self._load_all_pdfs()
            if not docs:
                return Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings,
                )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(docs)

            return Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.db_path,
            )

    def _is_basic_question(self, question: str) -> bool:
        q = question.lower()
        return q.startswith("what is") or q.startswith("define")

    def _build_graph(self):
        def assistant_node(state: MessagesState):
            user_msg = state["messages"][-1].content

            system = SystemMessage(
                content=(
                    "You are a Python learning assistant.\n"
                    "- Answer ONLY Python-related questions.\n"
                    "- If the question is NOT about Python, politely refuse.\n"
                    "- If the question is a basic definition, answer briefly WITHOUT tools.\n"
                    "- For explanations, comparisons, best practices, or examples, USE the retrieval tool.\n"
                    "- Be clear, concise, and educational."
                )
            )

            messages = [system] + state["messages"]

            if self._is_basic_question(user_msg):
                response = self.llm.invoke(messages)
            else:
                response = self.llm.bind_tools(self.tools).invoke(messages)

            return {"messages": [response]}

        def route(state: MessagesState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(MessagesState)
        builder.add_node("assistant", assistant_node)
        builder.add_node("tools", self.tool_node)

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", route)
        builder.add_edge("tools", "assistant")

        return builder

    def ask_question(self, question: str, user_id: str = "default"):
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": user_id}},
        )
        return result["messages"][-1].content


# Singleton
_assistant = None


def get_python_assistant():
    global _assistant
    if _assistant is None:
        _assistant = PythonAssistant()
    return _assistant
