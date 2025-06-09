import os
import json
from typing import Dict, List, Optional
import openai
from anthropic import Anthropic
from pathlib import Path
import numpy as np
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

class ScientificKnowledgeLLM:
    """LLM for scientific knowledge from academic papers"""
    
    def __init__(self, openai_key: str, anthropic_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.anthropic_client = Anthropic(api_key=anthropic_key)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a PDF and extract structured knowledge"""
        # Extract text and structure from PDF
        sections = self._extract_pdf_content(pdf_path)
        
        # Generate embeddings for each section
        embeddings = self.embedding_generator.generate_embeddings(sections)
        
        # Store in vector database
        self.vector_store.add_documents(sections, embeddings)
        
        return {
            "sections": sections,
            "embeddings": embeddings
        }
    
    def query(self, query: str) -> str:
        """Query the scientific knowledge base"""
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Find relevant sections
        relevant_sections = self.vector_store.search(query_embedding, top_k=5)
        
        # Prepare context from relevant sections
        context = self._prepare_context(relevant_sections)
        
        # Query both LLMs
        openai_response = self._query_openai(query, context)
        claude_response = self._query_claude(query, context)
        
        # Combine and synthesize responses
        final_response = self._synthesize_responses(openai_response, claude_response)
        
        return final_response
    
    def _extract_pdf_content(self, pdf_path: str) -> List[Dict]:
        """Extract structured content from PDF"""
        # Implementation for PDF text extraction
        pass
    
    def _prepare_context(self, sections: List[Dict]) -> str:
        """Prepare context from relevant sections"""
        context = "Scientific Knowledge Context:\n\n"
        for section in sections:
            context += f"From {section['title']}:\n{section['content']}\n\n"
        return context
    
    def _query_openai(self, query: str, context: str) -> str:
        """Query OpenAI with context"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in electronic music and music perception."},
                {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    
    def _query_claude(self, query: str, context: str) -> str:
        """Query Claude with context"""
        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
            ]
        )
        return response.content[0].text
    
    def _synthesize_responses(self, openai_response: str, claude_response: str) -> str:
        """Synthesize responses from both LLMs"""
        # Use Claude to synthesize the responses
        synthesis_prompt = f"""
        Synthesize these two responses about electronic music into a single, coherent answer:
        
        OpenAI Response: {openai_response}
        Claude Response: {claude_response}
        
        Provide a unified response that combines the best insights from both sources.
        """
        
        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        return response.content[0].text

class MixAnalysisLLM:
    """LLM for continuous learning from mix analyses"""
    
    def __init__(self, openai_key: str, anthropic_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.anthropic_client = Anthropic(api_key=anthropic_key)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.mix_analyses_dir = Path("mix_analyses")
    
    def add_mix_analysis(self, mix_dir: str) -> Dict:
        """Add a new mix analysis to the knowledge base"""
        # Load mix analysis files
        metadata = self._load_json(mix_dir, "metadata.json")
        summary = self._load_text(mix_dir, "summary.txt")
        segments = self._load_json(mix_dir, "segments.json")
        emotional_profile = self._load_json(mix_dir, "emotional_profile.json")
        technical_features = self._load_jsonl(mix_dir, "technical_features.jsonl")
        
        # Generate embeddings for each component
        embeddings = self.embedding_generator.generate_embeddings([
            summary,
            json.dumps(segments),
            json.dumps(emotional_profile),
            json.dumps(technical_features)
        ])
        
        # Store in vector database
        self.vector_store.add_documents([
            {"type": "summary", "content": summary},
            {"type": "segments", "content": json.dumps(segments)},
            {"type": "emotional", "content": json.dumps(emotional_profile)},
            {"type": "technical", "content": json.dumps(technical_features)}
        ], embeddings)
        
        return {
            "mix_id": metadata["mix_id"],
            "components": ["summary", "segments", "emotional", "technical"]
        }
    
    def query(self, query: str) -> str:
        """Query the mix analysis knowledge base"""
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Find relevant components
        relevant_components = self.vector_store.search(query_embedding, top_k=5)
        
        # Prepare context from relevant components
        context = self._prepare_context(relevant_components)
        
        # Query both LLMs
        openai_response = self._query_openai(query, context)
        claude_response = self._query_claude(query, context)
        
        # Combine and synthesize responses
        final_response = self._synthesize_responses(openai_response, claude_response)
        
        return final_response
    
    def _load_json(self, mix_dir: str, filename: str) -> Dict:
        """Load JSON file from mix directory"""
        with open(os.path.join(mix_dir, filename)) as f:
            return json.load(f)
    
    def _load_text(self, mix_dir: str, filename: str) -> str:
        """Load text file from mix directory"""
        with open(os.path.join(mix_dir, filename)) as f:
            return f.read()
    
    def _load_jsonl(self, mix_dir: str, filename: str) -> List[Dict]:
        """Load JSONL file from mix directory"""
        results = []
        with open(os.path.join(mix_dir, filename)) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    
    def _prepare_context(self, components: List[Dict]) -> str:
        """Prepare context from relevant components"""
        context = "Mix Analysis Context:\n\n"
        for component in components:
            context += f"{component['type'].title()}:\n{component['content']}\n\n"
        return context
    
    def _query_openai(self, query: str, context: str) -> str:
        """Query OpenAI with context"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in electronic music analysis and mix engineering."},
                {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    
    def _query_claude(self, query: str, context: str) -> str:
        """Query Claude with context"""
        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
            ]
        )
        return response.content[0].text
    
    def _synthesize_responses(self, openai_response: str, claude_response: str) -> str:
        """Synthesize responses from both LLMs"""
        # Use Claude to synthesize the responses
        synthesis_prompt = f"""
        Synthesize these two responses about mix analysis into a single, coherent answer:
        
        OpenAI Response: {openai_response}
        Claude Response: {claude_response}
        
        Provide a unified response that combines the best insights from both sources.
        """
        
        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        return response.content[0].text 