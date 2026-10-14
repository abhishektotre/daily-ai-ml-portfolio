class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def query(self, user_question, top_k=2):
        retrieved = self.vector_store.search(user_question, top_k=top_k)
        
        if not retrieved:
            return {
                "question": user_question,
                "synthesized_response": "I could not find relevant knowledge in the verified corpus to answer your question.",
                "citations": []
            }
            
        context_blocks = []
        citations = []
        for item in retrieved:
            doc = item["document"]
            context_blocks.append(f"[{doc['id']}: {doc['title']}] {doc['content']}")
            citations.append({"doc_id": doc["id"], "title": doc["title"], "relevance_score": item["score"]})
            
        synthesized_text = (
            f"Based on the verified retrieved documents, here is the answer:\n\n"
            f"For your inquiry regarding '{user_question}':\n"
            f"The verified knowledge indicates that {retrieved[0]['document']['content']}\n"
            f"Source references: {', '.join([c['doc_id'] for c in citations])}."
        )
        
        return {
            "question": user_question,
            "synthesized_response": synthesized_text,
            "citations": citations
        }
