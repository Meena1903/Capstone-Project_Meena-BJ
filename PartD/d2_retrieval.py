"""D2 - Build the TF-IDF index and retrieve the top three policy chunks."""
from d_rag import Retriever

retriever = Retriever(40, 10)
for result in retriever.retrieve("refund policy late delivery", k=3):
    print(result["source"], round(result["score"], 3), result["text"])
