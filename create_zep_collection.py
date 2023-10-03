from zep_python import ZepClient
import settings_private

# Replace with Zep API URL and (optionally) API key
zep = ZepClient("http://localhost:8000")

client = ZepClient(base_url=settings_private.ZEP_API_URL)

collection_name = "LagrangeDocs" # the name of your collection. alphanum values only

collection = client.document.add_collection(
    name=collection_name,  # required
    description="Documents about Lagrange",  # optional
    metadata={"category": "about"},  # optional metadata to associate with this collection
    embedding_dimensions=384,  # this must match the model you've configured for
    is_auto_embedded=True,  # use Zep's built-in embedder. Defaults to True
)