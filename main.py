from operator import itemgetter
from dotenv import load_dotenv

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from langchain.schema import BaseRetriever
from langchain.llms.base import LLM
from langchain.llms.openai import OpenAI
from watsonx_lc.llm import WatsonxLLM, DecodingMethods, FoundationModel
from watsonx_lc.retrievers import DiscoveryRetriever

from utils import docstore

import langchain
langchain.debug = True

load_dotenv()


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# agent for wds
# agent for web search


def rag_qa(doc_retriever: BaseRetriever, llm: LLM) -> None:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc_retriever)
    while True:
        print("Bot:", qa.run(input("User: ")))
        print()


def rag_chain_qa(doc_retriever: BaseRetriever, llm: LLM) -> None:
    template = """Answer the question based only on the following context:
    
    ---
        {context}
    ---
    
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    rag_qa_chain = (
            {
                "context": itemgetter("question") | doc_retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    while True:
        res_rag_qa = rag_qa_chain.invoke({"question": input("User: ")})
        print("Bot:", res_rag_qa)


def rag_chain_conv(doc_retriever: BaseRetriever, llm: LLM) -> None:
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=doc_retriever,
        chain_type='stuff',
    )

    history = []
    while True:
        question = input("User:")
        res = chain({"question": question, "chat_history": history})["answer"]
        history.append((question, res))
        print("Bot:", res)


if __name__ == "__main__":

    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vec_store = docstore.get_vector_store(embeddings=emb, source_docs='data/', index='rag_index')

    wds_retriever = DiscoveryRetriever()  # search types: document, passage. k
    mmr_retriever = vec_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50})
    sim_retriever = vec_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # meta-llama/llama-2-70b-chat
    model_flan_ul2 = WatsonxLLM(
        decoding_method=DecodingMethods.SAMPLE,
        min_new_tokens=10, max_new_tokens=200,
        temperature=0.65, top_p=0.4, top_k=20,
        repeat_penalty=1.8,
    )
    model_flan_t5 = WatsonxLLM(
        model_id=FoundationModel.FLAN_T5_XXL,
        decoding_method=DecodingMethods.SAMPLE,
        min_new_tokens=10, max_new_tokens=200,
        temperature=0.65, top_p=0.4, top_k=20,
        repeat_penalty=1.8,
    )
    model_llama2 = WatsonxLLM(
        model_id=FoundationModel.LLAMA2,
        decoding_method=DecodingMethods.SAMPLE,
        min_new_tokens=10, max_new_tokens=200,
        temperature=0.65, top_p=0.4, top_k=20,
        repeat_penalty=1.8,
    )
    model_openai = OpenAI(temperature=0.65)

    model = model_llama2
    retriever = sim_retriever

    # rag_qa(retriever, model)
    # rag_chain_qa(retriever, model)
    rag_chain_conv(retriever, model)
