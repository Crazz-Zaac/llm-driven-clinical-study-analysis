"""Test suite for TextEmbedder"""
import pytest
import logging
from app.rag.embeddings.embedder import TextEmbedder


class TestTextEmbedder:
    """Tests for embedding functionality"""
    
    @pytest.fixture
    def embedder(self):
        """Initialize embedder for tests"""
        return TextEmbedder(show_progress=False)  # Disable progress bar in tests
    
    def test_embedder_initialization(self, embedder):
        """Test that embedder initializes correctly"""
        assert embedder.model is not None
        assert embedder.model.get_sentence_embedding_dimension() == 384  # all-MiniLM-L6-v2 produces 384-dim vectors
    
    def test_embed_single_text(self, embedder):
        """Test embedding a single text"""
        text = "This is a test sentence."
        embedding = embedder.embed(text, show_timing=False)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Correct dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_multiple_texts(self, embedder):
        """Test embedding multiple texts"""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed(texts, show_timing=False)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_embed_with_timing(self, embedder, caplog):
        """Test that timing information is logged"""
        with caplog.at_level(logging.INFO):
            text = "Test sentence for timing"
            embeddings = embedder.embed(text, show_timing=True)
        
        assert len(embeddings) == 384
        # Check that timing was logged
        assert "Embedded" in caplog.text
        assert "text(s) in" in caplog.text
        assert "avg:" in caplog.text
    
    def test_embed_returns_normalized_vectors(self, embedder):
        """Test that embeddings are normalized (approximately unit vectors)"""
        text = "Test sentence for normalization check."
        embedding = embedder.embed(text, show_timing=False)
        
        # Calculate magnitude
        magnitude = sum(x**2 for x in embedding) ** 0.5
        
        # Should be close to 1 for normalized vectors
        assert 0.9 < magnitude < 1.1, f"Magnitude {magnitude} is not close to 1"
    
    def test_embed_empty_text_handling(self, embedder):
        """Test handling of empty text"""
        text = ""
        embedding = embedder.embed(text, show_timing=False)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
    
    def test_semantic_similarity(self, embedder):
        """Test that similar texts produce similar embeddings"""
        text1 = "The cat is sitting on the mat"
        text2 = "The cat sits on the mat"
        text3 = "The weather is sunny today"
        
        emb1 = embedder.embed(text1, show_timing=False)
        emb2 = embedder.embed(text2, show_timing=False)
        emb3 = embedder.embed(text3, show_timing=False)
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            dot_product = sum(x*y for x, y in zip(a, b))
            mag_a = sum(x**2 for x in a) ** 0.5
            mag_b = sum(x**2 for x in b) ** 0.5
            return dot_product / (mag_a * mag_b)
        
        sim_1_2 = cosine_similarity(emb1, emb2)  # Should be high
        sim_1_3 = cosine_similarity(emb1, emb3)  # Should be lower
        
        assert sim_1_2 > sim_1_3, "Similar texts should have higher similarity"
