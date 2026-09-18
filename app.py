import streamlit as st
import networkx as nx
import re
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama

st.set_page_config(page_title="Knowledge Graph RAG", page_icon="🕸️")
st.title("🕸️ Knowledge Graph RAG Chat")

# 1. Initialize LLM
@st.cache_resource
def load_llm():
    return ChatOllama(model="llama3:latest")

llm = load_llm()

# 2. Build the Knowledge Graph
@st.cache_resource
def build_graph():
    loader = TextLoader("sample.txt")
    documents = loader.load()
    raw_text = documents[0].page_content

    prompt = f"""
    Analyze the following text and extract relationships between entities. 
    Return the output strictly as triplets in this format:
    Subject | RELATIONSHIP | Object

    Do NOT include numbering.

    Example:
    Elon Musk | FOUNDED | SpaceX

    Text to analyze:
    '{raw_text}'
    """
    response = llm.invoke(prompt)

    G = nx.DiGraph()

    for line in response.content.split("\n"):
        if "|" in line:
            parts = line.split("|")
            if len(parts) == 3:
                # Strip out numbers, bullets, dots, and extra whitespace
                subject = re.sub(r"^[\s\d\.\-\*\)]+", "", parts[0]).strip()
                relation = parts[1].strip()
                obj = re.sub(r"^[\s\d\.\-\*\)]+", "", parts[2]).strip()

                # Ignore headers or empty nodes
                if subject.lower() != "subject" and subject and obj:
                    G.add_edge(subject, obj, label=relation)

    return G

with st.spinner("Building clean Knowledge Graph..."):
    graph = build_graph()

# Sidebar
with st.sidebar:
    st.header("📊 Graph Stats")
    st.write(f"**Nodes:** {graph.number_of_nodes()}")
    st.write(f"**Relationships:** {graph.number_of_edges()}")
    with st.expander("View All Stored Triplets"):
        for u, v, data in graph.edges(data=True):
            st.write(f"- `({u}) - [{data['label']}] -> ({v})`")

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Query & Retrieval
if user_query := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    query_lower = user_query.lower()
    # Extract clean individual words from query
    query_tokens = set(re.findall(r"\b\w+\b", query_lower))
    stopwords = {"what", "who", "where", "are", "all", "is", "in", "the", "a", "an", "of", "did", "do", "that"}
    meaningful_tokens = query_tokens - stopwords

    retrieved_facts = []

    # Check for general overview requests
    overview_keywords = ["what in", "what is in", "what do you know", "everything", "all facts"]
    if any(kw in query_lower for kw in overview_keywords):
        for s, o, d in graph.edges(data=True):
            retrieved_facts.append(f"{s} {d['label']} {o}")
    else:
        # 1. First, find primary matching nodes
        matched_nodes = set()
        for node in graph.nodes():
            node_clean = node.lower()
            node_tokens = set(re.findall(r"\b\w+\b", node_clean))
            if (node_clean in query_lower) or bool(meaningful_tokens & node_tokens):
                matched_nodes.add(node)

        # 2. Collect 1-hop facts (Direct connections)
        traversed_nodes = set(matched_nodes)
        for node in matched_nodes:
            # Outgoing edges: node -> neighbor
            for neighbor in graph.successors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                retrieved_facts.append(f"{node} {edge_data['label']} {neighbor}")
                traversed_nodes.add(neighbor)

            # Incoming edges: neighbor -> node
            for predecessor in graph.predecessors(node):
                edge_data = graph.get_edge_data(predecessor, node)
                retrieved_facts.append(f"{predecessor} {edge_data['label']} {node}")
                traversed_nodes.add(predecessor)

        # 3. Collect 2-hop facts (Neighbors of neighbors)
        for node in traversed_nodes:
            for neighbor in graph.successors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                retrieved_facts.append(f"{node} {edge_data['label']} {neighbor}")

    # Remove duplicates and create the context string
    retrieved_facts = list(set(retrieved_facts))
    context_string = "\n".join(retrieved_facts) if retrieved_facts else "No direct matching facts found in graph."

    rag_prompt = f"""
    Answer the user's question based on the facts provided below. 
    You are allowed to use logical deduction to interpret the relationships (e.g., if A is rebranded to B, then A was the original name).
    If the provided facts do not contain enough information to deduce the answer, simply state that you don't know based on the graph. Do not use outside knowledge.

    Facts:
    {context_string}

    Question: {user_query}
    Answer:
    """

    with st.chat_message("assistant"):
        with st.spinner("Searching graph..."):
            ai_response = llm.invoke(rag_prompt).content
            st.markdown(ai_response)

            if retrieved_facts:
                with st.expander("🔍 Retrieved Graph Facts"):
                    for fact in retrieved_facts:
                        st.markdown(f"- `{fact}`")

    st.session_state.messages.append({"role": "assistant", "content": ai_response})