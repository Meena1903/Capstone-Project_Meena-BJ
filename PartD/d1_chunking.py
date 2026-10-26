"""D1 - Chunk each policy document into 40-word passages with 10-word overlap."""
import pandas as pd

from d_rag import Retriever

chunks = Retriever(40, 10).chunks
counts = pd.Series([chunk["source"] for chunk in chunks]).value_counts().to_dict()
print("D1 total chunks:", len(chunks), counts)
