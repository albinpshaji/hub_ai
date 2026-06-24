"""
Context Assembler — Phase 6

Formats retrieved chunks into a context block for the LLM prompt.
"""

from typing import List

def assemble_context(chunks: List[str]) -> str:
    """
    Combine multiple text chunks into a single context string.
    """
    if not chunks:
        return ""
        
    context_parts = ["--- DOCUMENT CONTEXT ---"]
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Excerpt {i}]\n{chunk}\n")
    context_parts.append("------------------------")
    
    return "\n".join(context_parts)
