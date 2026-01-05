import os
import sqlite3
import pdfplumber
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.documents import Document

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)


class PythonAssistant:
    def __init__(self, pdf_directory="files", db_path="docstore"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.pdf_directory = pdf_directory
        self.db_path = db_path

        # LLM & Embeddings
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key,
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key,
        )

        # Vector store
        self.vector_store = self._initialize_vector_store()

        count = self.vector_store._collection.count()
        print(f"[VectorStore] Loaded {count} chunks")

        # Tools & Graph
        self.tools = self._setup_tools()
        self.tool_node = ToolNode(self.tools)
        self.builder = self._build_graph()

        # Checkpointer
        self.checkpointer = self._initialize_checkpointer()

    # --------------------------------------------------
    # Checkpointer
    # --------------------------------------------------


    def _initialize_checkpointer(self):
        os.makedirs("memory", exist_ok=True)
        db_path = os.path.abspath("memory/conversations.db")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)


    # --------------------------------------------------
    # Tools
    # --------------------------------------------------
    def _setup_tools(self):
        @tool
        def retriever_tool(query: str) -> str:
            """Retrieve relevant documentation chunks."""
            docs = self.vector_store.similarity_search(query, k=3)

            if not docs:
                return "No relevant documents found."

            return "\n\n---\n\n".join(
                f"Source: {d.metadata.get('source', 'Unknown')} | "
                f"Page: {d.metadata.get('page', 'N/A')}\n\n"
                f"{d.page_content[:1200]}"
                for d in docs
            )

        return [retriever_tool]


    # --------------------------------------------------
    # Document Loading
    # --------------------------------------------------
    def _load_all_pdfs(self):
        documents = []

        if not os.path.exists(self.pdf_directory):
            os.makedirs(self.pdf_directory)
            return documents

        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith(".pdf"):
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


    # --------------------------------------------------
    # Vector Store Init
    # --------------------------------------------------
    def _initialize_vector_store(self):
        os.makedirs(self.db_path, exist_ok=True)

        chroma_path = os.path.join(self.db_path, "chroma.sqlite3")

        if os.path.exists(chroma_path):
            return Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
            )

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


    # --------------------------------------------------
    # Retrieval Decision Node
    # --------------------------------------------------
  
    def _needs_retrieval(self, state: MessagesState):
        """Determine if we need retrieval. Returns the next node name."""
        question = state["messages"][-1].content

        prompt = f"""
    Decide whether this question requires retrieval from documentation.

    Answer ONLY one word:
    - RETRIEVE (needs docs, APIs, standards, files)
    - NO_RETRIEVE (general Python knowledge)

    Question:
    {question}
    """

        decision = self.llm.invoke(
            [HumanMessage(content=prompt)]
        ).content.strip()

        if "RETRIEVE" in decision:
            return "retrieve_route"  
        else:
            return "direct_answer"   
        


    # --------------------------------------------------
    # Graph
    # --------------------------------------------------
    def _build_graph(self):
        def assistant_node(state: MessagesState):
            system = SystemMessage(
                content=(
                    "You are a Python assistant. "
                    "Use tools ONLY when documentation is required."
                )
            )

            messages = [system] + state["messages"]
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)

            return {"messages": [response]}

        def direct_answer_node(state: MessagesState):
            """Answer directly without retrieval."""
            system = SystemMessage(
                content=(
                    "You are a Python assistant. "
                    "Answer the question based on your general knowledge of Python."
                )
            )

            messages = [system] + state["messages"]
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        def should_use_tools(state: MessagesState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(MessagesState)

        # Add nodes
        builder.add_node("assistant", assistant_node)
        builder.add_node("direct_answer", direct_answer_node)
        builder.add_node("tools", self.tool_node)

        builder.add_conditional_edges(
            START,
            self._needs_retrieval, 
            {
                "retrieve_route": "assistant",
                "direct_answer": "direct_answer"
            }
        )
        
        builder.add_conditional_edges("assistant", should_use_tools)
        builder.add_edge("tools", "assistant")
        
        builder.add_edge("direct_answer", END)

        return builder
    


        # --------------------------------------------------
        # Public API 
        # --------------------------------------------------
    def ask_question(self, question: str, user_id: str = "default"):
        agent = self.builder.compile(checkpointer=self.checkpointer)

        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": user_id}},
        )

        return result["messages"][-1].content


assistant = None


def get_python_assistant():
    global assistant
    if assistant is None:
        assistant = PythonAssistant()
    return assistant
