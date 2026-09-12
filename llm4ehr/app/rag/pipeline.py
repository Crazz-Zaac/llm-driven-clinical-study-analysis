import logging

from app.rag.services import RetrievalService
from app.rag.llm.ollama_chat_model import OllamaChatModel
from app.rag.prompts.system_prompt import (
    SYSTEM_PROMPT,
    ALL_PAPER_PROMPT,
    RELEVANT_PAPERS_TABLE_PROMPT,
)
from app.schemas.chat_schema import ChatMessage, ChatRequest, ChatResponse
from app.schemas.retrieval_schema import QueryRequest, RetrievalScope

logger = logging.getLogger(__name__)


def _markdown_table_cell(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("|", "\\|").replace("\n", "<br>").strip()


def _markdown_link(label: str | None, url: str | None) -> str:
    escaped_label = _markdown_table_cell(label).replace("[", "\\[").replace("]", "\\]")
    if not url:
        return escaped_label
    escaped_url = url.replace(")", "%29").strip()
    return f"[{escaped_label}]({escaped_url})"


class RAGPipeline:
    def __init__(self):
        self.retrieval_service = RetrievalService()
        # this is the local model used for generating responses in the RAG pipeline, it can be swapped out if needed
        self.chat_model = OllamaChatModel()

    def run(self, request: ChatRequest) -> ChatResponse:
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return ChatResponse(
                response="No user messages found in the conversation history.",
                source_documents=None,
                retrieval_scope=request.retrieval_scope,
            )

        last_user_message = user_messages[-1].content

        # Build a retrieval query from the most recent user turns.
        # Consecutive duplicate messages are ignored to avoid
        # duplicated queries caused by frontend message echoes.
        retrieval_turns: list[str] = []
        for msg in request.messages:
            if msg.role == "user" and (
                not retrieval_turns or msg.content != retrieval_turns[-1]
            ):
                retrieval_turns.append(msg.content)
        retrieval_query = " ".join(retrieval_turns[-3:])

        logger.debug("Retrieval query: %s", retrieval_query)

        query_response = self.retrieval_service.retrieve(
            QueryRequest(
                query=retrieval_query,
                retrieval_scope=request.retrieval_scope,
                article_id=request.article_id,
            )
        )

        # the documents retrieved from the vector database are in the llm_docs field of the response schema
        # and will be used as the context for the LLM to generate a response.
        # The source_documents field is what will be returned to the frontend
        docs = query_response.llm_docs or []
        if not docs:
            return ChatResponse(
                response="Not found in retrieved documents. Please ensure the article is indexed.",
                source_documents=[],
                contexts=[],
                retrieval_scope=request.retrieval_scope,
            )

        # Depending on the retrieval scope, we may want to provide different instructions to the LLM.
        if request.retrieval_scope == RetrievalScope.ALL_PAPERS:
            per_doc_response = [
                "| Paper | Answer | Evidence |",
                "| --- | --- | --- |",
            ]
            for doc in docs:
                doc_context = (
                    f"Article ID: {doc.article_id}\n"
                    f"Title: {doc.title}\n"
                    f"URL: {doc.url}\n"
                    f"Sections: {doc.section}\n\n"
                    f"Content:\n{doc.combined_text}\n"
                )
                sys_prompt = (
                    f"{ALL_PAPER_PROMPT}\n\nRetrieved document:\n{doc_context}".strip()
                )
                single_doc_request = ChatRequest(
                    messages=[
                        ChatMessage(role="system", content=sys_prompt),
                        ChatMessage(role="user", content=last_user_message),
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    retrieval_scope=request.retrieval_scope,
                    article_id=request.article_id,
                )
                resp = self.chat_model.generate_response(single_doc_request)
                if resp.response.startswith("Error generating response:"):
                    logger.warning(
                        "Chat model failed for article_id=%s: %s",
                        doc.article_id,
                        resp.response,
                    )
                    answer_text = "Unable to generate a response for this document due to a system error."
                else:
                    answer_text = resp.response
                per_doc_response.append(
                    "| "
                    f"{_markdown_link(doc.title, doc.url)} | "
                    f"{_markdown_table_cell(answer_text)} | "
                    f"{_markdown_table_cell(doc.section)} |"
                )
            final_response = "\n".join(per_doc_response)

        else:  # for RetrievalScope.RELEVANT_PAPERS and RetrievalScope.SPECIFIC_PAPER
            retrieved_docs = "\n\n".join(
                (
                    f"===== DOCUMENT {i + 1} OF {len(docs)} =====\n"
                    f"Article ID: {doc.article_id}\n"
                    f"Title: {doc.title}\n"
                    f"URL: {doc.url}\n"
                    f"Sections: {doc.section}\n\n"
                    f"Content:\n{doc.combined_text}\n"
                    f"===== END DOCUMENT {i + 1} ====="
                )
                for i, doc in enumerate(docs)
            )

            logger.debug("Retrieved docs:\n%s", retrieved_docs)
            if request.retrieval_scope == RetrievalScope.RELEVANT_PAPERS:
                response_prompt = RELEVANT_PAPERS_TABLE_PROMPT
            else:
                response_prompt = SYSTEM_PROMPT
            sys_prompt = f"{response_prompt}\n\nRetrieved documents:\n{retrieved_docs}".strip()
            llm_request = ChatRequest(
                messages=[
                    ChatMessage(role="system", content=sys_prompt),
                    ChatMessage(role="user", content=last_user_message),
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                retrieval_scope=request.retrieval_scope,
                article_id=request.article_id,
            )
            response = self.chat_model.generate_response(llm_request)
            final_response = response.response

        return ChatResponse(
            response=final_response,
            source_documents=query_response.source_documents,
            contexts=query_response.contexts if request.include_contexts else None,
            retrieval_scope=request.retrieval_scope,
        )
