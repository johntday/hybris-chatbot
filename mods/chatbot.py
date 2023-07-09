import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval import prompts


class Chatbot:
    qa_template = """"Answer the question based on the context and chat history below. If the
    question cannot be answered using the information provided answer with "Sorry, I do not know".

    Context: SAP Commerce Cloud (Hybris) is the world's leading B2C and B2B commerce solution. 
    SAP Commerce Cloud is a multi-tenant, cloud-based commerce platform that empowers brands to create intelligent, 
    unified buying experiences across all channels — mobile, social, web, and store.
    
    Chat History: {chat_history}

    Question: {question}
    """
    context = """SAP Commerce Cloud (Hybris) is the world's leading B2C and B2B commerce solution. 
    SAP Commerce Cloud is a multi-tenant, cloud-based commerce platform that empowers brands to create intelligent, 
    unified buying experiences across all channels — mobile, social, web, and store."""

    def __init__(self, model_name, temperature, vectors, search_type="similarity", k=4):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors
        self.search_type = search_type
        self.k = k

    def conversational_chat(self, query):
        """
        Starts a conversational chat with a model via Langchain
        """
        prompt_template = PromptTemplate(template=self.qa_template, input_variables=["question", "chat_history"])

        # fixme: https://stackoverflow.com/questions/76178954/giving-systemmessage-context-to-conversationalretrievalchain-and-conversationbuf
        # chain = load_qa_with_sources_chain(
        #     llm=ChatOpenAI(model_name=self.model_name, temperature=self.temperature),
        #     prompt=prompt_template,
        #     retriever=self.vectors.as_retriever(search_type=self.search_type),
        #     return_source_documents=True,
        # )

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=self.model_name, temperature=self.temperature),
            retriever=self.vectors.as_retriever(search_type=self.search_type),
            return_source_documents=True,
        )

        result = chain({"question": query, "context": self.context, "chat_history": st.session_state["history"]})
        # print("source_documents: " + result['source_documents'])
        # print(*result['source_documents'], sep="\n\n")

        st.session_state["history"].append((query, result["answer"]))
        st.session_state["chat_sources"] = result['source_documents']

        return result["answer"]
