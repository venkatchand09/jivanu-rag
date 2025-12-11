# src/node/rag_nodes.py
"""
Enhanced RAG nodes with advanced prompting for biotech research
Generates comprehensive answers with reasoning, hypotheses, and suggestions
"""

from typing import List, Tuple
import json
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from src.state.rag_state import RAGState
from src.config.config import Config

class RAGNodes:
    """Enhanced nodes for retrieval and generation with researcher perspective"""
    
    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes
        
        Args:
            retriever: Document retriever
            llm: Language model (Chat model)
        """
        self.retriever = retriever
        self.llm = llm
    
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with retrieved documents
        """
        print(f"Retrieving documents for: {state.question[:100]}...")
        
        # Retrieve documents
        docs: List[Document] = self.retriever.invoke(state.question)
        
        print(f"Retrieved {len(docs)} documents")
        
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
            chat_history=state.chat_history
        )
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate comprehensive answer with reasoning, hypothesis, and suggestions
        
        Args:
            state: Current RAG state with retrieved documents
            
        Returns:
            Updated state with generated answer and metadata
        """
        print("Generating comprehensive answer...")
        
        # Build context from retrieved documents
        sources_list = []
        docs_context_parts = []
        
        for i, doc in enumerate(state.retrieved_docs[:15], start=1):
            meta = doc.metadata or {}
            source = meta.get("source", "unknown")
            page = meta.get("page", "N/A")
            doc_type = meta.get("type", "page_text")
            pdf_name = meta.get("pdf_name", "unknown")
            
            # Store source info
            sources_list.append({
                "index": i,
                "source": source,
                "page": page,
                "type": doc_type,
                "pdf_name": pdf_name
            })
            
            # Build context with source references
            content_preview = doc.page_content[:1000]
            docs_context_parts.append(
                f"[Source {i}] {pdf_name} (Page {page}, Type: {doc_type})\n{content_preview}\n"
            )
        
        docs_context = "\n---\n".join(docs_context_parts)
        
        # Build conversation history
        history_text = self._format_history(state.chat_history)

        # Create specialized prompt for biotech researcher - ENHANCED FOR COMPREHENSIVE ANSWERS
        system_prompt = """You are a distinguished senior biotech and medical researcher with 20+ years of experience specializing in microbe-based therapeutics and drug development. You are known for your detailed, thorough explanations that help other researchers deeply understand complex topics.

Your expertise includes:
- Microbial genomics and metabolic engineering
- Therapeutic peptide delivery systems
- Host-microbe interactions and microbiome dynamics
- Clinical development pathways for microbial therapeutics
- Regulatory frameworks for novel biologics
- Manufacturing and scale-up challenges
- Safety and immunogenicity considerations

When answering research questions, provide COMPREHENSIVE, DETAILED responses that include:

1. **Direct Answer** (4-6 sentences minimum): 
   - Provide a thorough, detailed response that fully addresses the question
   - Include specific examples, mechanisms, or applications
   - Mention key researchers, companies, or landmark studies when relevant
   - Explain the significance and context of your answer

2. **Scientific Reasoning & Deep Dive** (8-12 sentences minimum):
   - Explain the underlying biology, mechanisms, and molecular details
   - Discuss the scientific principles that support your answer
   - Compare different approaches or methodologies
   - Explain why certain methods work better than others
   - Include relevant data, findings, or experimental evidence from the literature
   - Discuss challenges, limitations, and current debates in the field
   - Connect concepts across different areas of research

3. **Research Hypothesis** (2-4 sentences):
   - Propose a specific, testable hypothesis based on current evidence
   - Explain the rationale behind the hypothesis
   - Suggest what experiments would test this hypothesis
   - Discuss potential outcomes and their implications

4. **Follow-up Suggestions** (5-7 detailed items):
   - Suggest specific experimental approaches with details on methodology
   - Recommend particular research directions with justification
   - Propose questions that would advance understanding
   - Identify gaps in current knowledge worth investigating
   - Suggest collaborations or techniques that could be valuable
   - Each suggestion should be 2-3 sentences, not just bullet points

5. **Source Citations** (comprehensive):
   - Reference ALL sources that informed your answer
   - Include specific page numbers and relevant excerpts
   - Explain how each source contributes to your answer

Format your response as JSON with these exact keys:
{
    "answer": "Your comprehensive, detailed direct answer here (4-6+ sentences)",
    "reasoning": "Your in-depth scientific reasoning, mechanisms, and analysis here (8-12+ sentences with detailed explanations)",
    "hypothesis": "Your specific, detailed testable research hypothesis with rationale (2-4 sentences)",
    "suggestions": [
        "Detailed suggestion 1 with methodology and justification (2-3 sentences)",
        "Detailed suggestion 2 with specific approaches (2-3 sentences)",
        "Detailed suggestion 3 with experimental design considerations (2-3 sentences)",
        "Detailed suggestion 4 with technical details (2-3 sentences)",
        "Detailed suggestion 5 with expected outcomes (2-3 sentences)"
    ],
    "sources": [
        {"index": 1, "source": "filename.pdf", "page": "5", "excerpt": "detailed relevant excerpt"},
        {"index": 2, "source": "filename2.pdf", "page": "12", "excerpt": "detailed relevant excerpt"}
    ],
    "confidence": 0.85
}

IMPORTANT GUIDELINES FOR COMPREHENSIVE ANSWERS:
- Write as if explaining to a fellow researcher who wants to deeply understand the topic
- Use technical terminology appropriately with brief explanations
- Provide specific examples, data points, and methodologies
- Discuss both successes and challenges in the field
- Compare and contrast different approaches
- Mention specific organisms, strains, or systems when relevant
- Include quantitative information (percentages, concentrations, timeframes) when available
- Discuss clinical, regulatory, or practical implications
- Connect to broader themes in biotech and drug development
- Be thorough but maintain clarity and logical flow

Remember: More detail is better! Aim for comprehensive, publication-quality explanations that would be worthy of a scientific review article."""

        human_prompt = f"""Based on the following research documents, answer this question with COMPREHENSIVE DETAIL:

**Question:** {state.question}

**Retrieved Context from Scientific Literature:**
{docs_context}

**Conversation History:**
{history_text}

Please provide an EXTENSIVE, DETAILED, COMPREHENSIVE answer following all the format requirements. 

Think of this as writing a detailed section for a scientific review article. Your answer should be thorough enough that a researcher reading it would:
1. Fully understand the topic
2. Know the key mechanisms and principles
3. Understand the current state of research
4. See connections to related work
5. Know what questions remain unanswered
6. Have concrete next steps for their research

Provide rich detail, specific examples, and comprehensive explanations. Do not be brief - be thorough and educational."""
        
#         # Create specialized prompt for biotech researcher
#         system_prompt = """You are a senior biotech and medical researcher specializing in microbe-based therapeutics and drug development. 

# Your expertise includes:
# - Microbial genomics and metabolic engineering
# - Therapeutic peptide delivery systems
# - Host-microbe interactions
# - Clinical development of microbial therapeutics
# - Regulatory pathways for novel biologics

# When answering research questions, you must provide:

# 1. **Direct Answer** (2-4 sentences): Clear, actionable response from a researcher's perspective
# 2. **Scientific Reasoning** (3-5 sentences): Explain the underlying biology, mechanisms, or data supporting your answer
# 3. **Research Hypothesis** (1-2 sentences): Propose a testable hypothesis based on the question and available evidence
# 4. **Follow-up Suggestions** (3-5 items): Suggest next experimental steps, additional questions, or related research directions
# 5. **Source Citations**: Reference the specific sources that informed your answer

# Format your response as JSON with these exact keys:
# {
#     "answer": "Your direct answer here",
#     "reasoning": "Scientific reasoning and context",
#     "hypothesis": "Testable research hypothesis",
#     "suggestions": ["Suggestion 1", "Suggestion 2", "Suggestion 3"],
#     "sources": [
#         {"index": 1, "source": "filename.pdf", "page": "5", "excerpt": "relevant excerpt"},
#         {"index": 2, "source": "filename2.pdf", "page": "12", "excerpt": "relevant excerpt"}
#     ],
#     "confidence": 0.85
# }

# Be precise, evidence-based, and acknowledge uncertainty when appropriate."""

#         human_prompt = f"""Based on the following research documents, answer this question:

# **Question:** {state.question}

# **Retrieved Context:**
# {docs_context}

# **Conversation History:**
# {history_text}

# Provide a comprehensive answer following the format specified in the system instructions."""

        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # Generate response
        try:
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse JSON response
            parsed = self._parse_llm_response(response_text)
            
            # Build final state
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=parsed.get("answer", "Unable to generate answer"),
                reasoning=parsed.get("reasoning", ""),
                hypothesis=parsed.get("hypothesis", ""),
                suggestions=parsed.get("suggestions", []),
                sources=parsed.get("sources", sources_list),
                chat_history=state.chat_history,
                confidence_score=parsed.get("confidence", 0.0)
            )
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            
            # Fallback response
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer="I encountered an error generating a comprehensive answer. Please try rephrasing your question.",
                reasoning="Error in LLM response generation",
                hypothesis="",
                suggestions=["Try rephrasing the question", "Check if documents are relevant"],
                sources=sources_list,
                chat_history=state.chat_history,
                confidence_score=0.0
            )
    
    def _format_history(self, history: List[Tuple[str, str]]) -> str:
        """Format conversation history for context"""
        if not history:
            return "No previous conversation"
        
        # Keep only recent history
        recent_history = history[-Config.MAX_HISTORY_TURNS:]
        
        formatted = []
        for role, message in recent_history:
            formatted.append(f"{role.upper()}: {message[:200]}")
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, response_text: str) -> dict:
        """
        Parse LLM response, handling both JSON and plain text
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed dictionary
        """
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.rfind("```")
                json_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.rfind("```")
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text.strip()
            
            # Parse JSON
            parsed = json.loads(json_text)
            return parsed
            
        except json.JSONDecodeError:
            # Fallback: treat entire response as answer
            print("Warning: Could not parse JSON response, using fallback")
            return {
                "answer": response_text,
                "reasoning": "",
                "hypothesis": "",
                "suggestions": [],
                "sources": [],
                "confidence": 0.5
            }