from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
            keep_separator=True
        )

    def split_text(self, text: str) -> list:
        if text:
            return self.text_splitter.create_documents([text])
        return []