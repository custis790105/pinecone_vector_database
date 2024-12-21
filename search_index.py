# Import the Pinecone library
from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(
    api_key="pcsk_4cVKjt_JcdVd4fnekDbdDd91XjKKfViZwGD8mKJ2ACDNe3452jqW92k9MR6rPg37hC153V"
)
index = pc.Index("example-index")

# Define your query
query = "Tell me about the tech company known as Apple."

# Convert the query into a numerical vector that Pinecone can search with
query_embedding = pc.inference.embed(
    model="multilingual-e5-large", inputs=[query], parameters={"input_type": "query"}
)


# Search the index for the three most similar vectors
results = index.query(
    namespace="example-namespace",
    vector=query_embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True,
)

print(results)
