"""D3 - Answer a policy question from the retrieved context."""
from d_rag import Retriever, answer

reply, sources = answer("Who must approve a refund above INR 1000?", Retriever(40, 10))
print("answer:", reply)
print("sources:", sources)
