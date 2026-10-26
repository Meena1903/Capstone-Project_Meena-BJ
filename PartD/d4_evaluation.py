"""D4 - Evaluate retrieval and grounding with overlap 10 and overlap 0."""
from d_rag import Retriever, evaluate

evaluate(Retriever(40, 10), "D4 overlap 10")
zero_overlap = Retriever(40, 0)
print("chunks with overlap 0:", len(zero_overlap.chunks))
evaluate(zero_overlap, "D4 overlap 0")
