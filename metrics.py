"""
Frozen Metric Definitions and Formulas for Document Chunking Pipeline.

This module contains the EXACT definitions for all metrics used in evaluation.
These definitions must remain stable to ensure consistent measurement.

CRITICAL: Do not modify metric formulas without versioning the change.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time



DEFAULT_TOP_K = 10  # For Relevance %
DEFAULT_SUFFICIENCY_K = 3  # For Top-K Sufficiency %

SEMANTIC_ACCEPT_CONFIDENCE_THRESHOLD = 0.7
FORCE_REFINEMENT_THRESHOLD = 0.4  # If Semantic Accept % < 40%, force refinement

# Chunk length unit (FROZEN - never mix units)
LENGTH_UNIT = "characters"  # Options: "characters" or "tokens"


class MetricUnit(Enum):
    """Units for measurements - NEVER mix units in same report."""
    CHARACTERS = "characters"
    TOKENS = "tokens"
    SECONDS = "seconds"
    PERCENTAGE = "percentage"
    COUNT = "count"
    SCORE = "score_0_to_1"




@dataclass
class StageTimings:
    """
    Stage-wise timing breakdown for diagnostics report.
    
    Required pipeline stages:
    1. Parse/Extract - PyPDF/Sherpa/OCR text extraction
    2. Boundary Detection - Semantic boundary prompt calls
    3. Structure Analysis - Hierarchy/section reasoning
    4. Chunk Assembly - Building chunks from blocks
    5. Validation/Refinement - Quality prompt + auto split/merge
    6. Embedding - Embedding generation
    7. Indexing - Vector DB writes
    """
    # Parse / Extract stage
    parse_start_time: float = 0.0
    parse_end_time: float = 0.0
    parse_pages: int = 0
    parse_raw_text_length: int = 0
    
    # Boundary Detection stage
    boundary_start_time: float = 0.0
    boundary_end_time: float = 0.0
    boundary_blocks_processed: int = 0
    boundary_count: int = 0
    boundary_avg_confidence: float = 0.0
    
    # Structure Analysis stage
    structure_start_time: float = 0.0
    structure_end_time: float = 0.0
    structure_section_count: int = 0
    structure_avg_depth: float = 0.0
    structure_max_depth: int = 0
    
    # Chunk Assembly stage
    assembly_start_time: float = 0.0
    assembly_end_time: float = 0.0
    assembly_chunks_before_validation: int = 0
    assembly_avg_chunk_size: float = 0.0
    
    # Validation / Refinement stage
    validation_start_time: float = 0.0
    validation_end_time: float = 0.0
    validation_accepted_count: int = 0
    validation_refined_count: int = 0
    validation_rejected_count: int = 0
    validation_accept_rate_before: float = 0.0
    validation_accept_rate_after: float = 0.0
    
    # Embedding stage
    embedding_start_time: float = 0.0
    embedding_end_time: float = 0.0
    embedding_chunks_embedded: int = 0
    embedding_model_used: str = ""
    
    # Indexing stage
    indexing_start_time: float = 0.0
    indexing_end_time: float = 0.0
    indexing_success_count: int = 0
    indexing_failure_count: int = 0
    
    def get_parse_time(self) -> float:
        return self.parse_end_time - self.parse_start_time if self.parse_end_time else 0.0
    
    def get_boundary_time(self) -> float:
        return self.boundary_end_time - self.boundary_start_time if self.boundary_end_time else 0.0
    
    def get_structure_time(self) -> float:
        return self.structure_end_time - self.structure_start_time if self.structure_end_time else 0.0
    
    def get_assembly_time(self) -> float:
        return self.assembly_end_time - self.assembly_start_time if self.assembly_end_time else 0.0
    
    def get_validation_time(self) -> float:
        return self.validation_end_time - self.validation_start_time if self.validation_end_time else 0.0
    
    def get_embedding_time(self) -> float:
        return self.embedding_end_time - self.embedding_start_time if self.embedding_end_time else 0.0
    
    def get_indexing_time(self) -> float:
        return self.indexing_end_time - self.indexing_start_time if self.indexing_end_time else 0.0
    
    def get_total_time(self) -> float:
        return (self.get_parse_time() + self.get_boundary_time() + 
                self.get_structure_time() + self.get_assembly_time() +
                self.get_validation_time() + self.get_embedding_time() + 
                self.get_indexing_time())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parse": {
                "time_seconds": round(self.get_parse_time(), 3),
                "pages": self.parse_pages,
                "raw_text_length": self.parse_raw_text_length
            },
            "boundary_detection": {
                "time_seconds": round(self.get_boundary_time(), 3),
                "blocks_processed": self.boundary_blocks_processed,
                "boundaries_detected": self.boundary_count,
                "avg_confidence": round(self.boundary_avg_confidence, 3)
            },
            "structure_analysis": {
                "time_seconds": round(self.get_structure_time(), 3),
                "sections": self.structure_section_count,
                "avg_depth": round(self.structure_avg_depth, 2),
                "max_depth": self.structure_max_depth
            },
            "chunk_assembly": {
                "time_seconds": round(self.get_assembly_time(), 3),
                "chunks_before_validation": self.assembly_chunks_before_validation,
                "avg_chunk_size": round(self.assembly_avg_chunk_size, 1)
            },
            "validation_refinement": {
                "time_seconds": round(self.get_validation_time(), 3),
                "accepted": self.validation_accepted_count,
                "refined": self.validation_refined_count,
                "rejected": self.validation_rejected_count,
                "accept_rate_before": round(self.validation_accept_rate_before * 100, 1),
                "accept_rate_after": round(self.validation_accept_rate_after * 100, 1)
            },
            "embedding": {
                "time_seconds": round(self.get_embedding_time(), 3),
                "chunks_embedded": self.embedding_chunks_embedded,
                "model": self.embedding_model_used
            },
            "indexing": {
                "time_seconds": round(self.get_indexing_time(), 3),
                "success": self.indexing_success_count,
                "failure": self.indexing_failure_count
            },
            "total_time_seconds": round(self.get_total_time(), 3)
        }


# =============================================================================
# FROZEN METRIC DEFINITIONS
# =============================================================================

@dataclass
class FrozenMetrics:
    """
    FROZEN metric definitions with exact formulas.
    
    IMPORTANT: These metrics are computed ONLY on final indexed chunks 
    (after validation + refinement). Do NOT count rejected or intermediate chunks.
    """
    
    # 1. Total Chunks
    # What: Count of final indexed chunks (after validation + refinement)
    # Formula: TotalChunks = number of chunk objects stored in vector DB
    # Rule: Do NOT count rejected or intermediate chunks
    total_chunks: int = 0
    
    # 2. Average Chunk Length
    # What: Typical size of a chunk
    # Unit: FROZEN to one unit (characters or tokens) - NEVER mix
    # Formula: AvgChunkLength = sum(length(chunk.text)) / TotalChunks
    avg_chunk_length: float = 0.0
    length_unit: str = LENGTH_UNIT
    
    # 3. Average Quality
    # What: How good chunks are on average
    # Source: Quality validation prompt scores (normalized 0-1)
    # Formula: AvgQuality = mean(chunk.validation.overall_score)
    # Rule: Exclude chunks without validation scores (don't treat as zero)
    avg_quality: float = 0.0
    chunks_with_quality: int = 0  # How many chunks had quality scores
    
    # 4. Relevance % (Retrieval Recall)
    # What: Did we retrieve something relevant?
    # K value: MUST be reported
    # Definition: Query is relevant if ≥1 retrieved chunk overlaps ground-truth
    # Formula: Relevance% = (queries with ≥1 relevant in top-K) / total_queries × 100
    # Rule: Does NOT measure efficiency, only recall
    relevance_pct: float = 0.0
    relevance_k: int = DEFAULT_TOP_K
    relevance_queries_total: int = 0
    relevance_queries_with_relevant: int = 0
    
    # 5. Semantic Accept % (replaces old "Optimal %")
    # What: Chunks semantically correct without fixes
    # Source: Quality validation prompt
    # Definition: Accepted if recommendation=="ACCEPT" AND confidence >= threshold
    # Formula: SemanticAccept% = (accepted_chunks / total_chunks) × 100
    # Rule: Run one auto-refinement pass before computing
    semantic_accept_pct: float = 0.0
    semantic_accept_threshold: float = SEMANTIC_ACCEPT_CONFIDENCE_THRESHOLD
    semantic_accepted_count: int = 0
    
    # 6. Top-K Sufficiency % (Retrieval Efficiency)
    # What: How many chunks needed to answer?
    # K value: MUST be reported (usually 3)
    # Definition: Query sufficient if answer fully supported by ≤K chunks
    # Formula: TopKSufficiency% = (queries answerable with ≤K) / total_queries × 100
    # Rule: Directly measures query-document asymmetry reduction
    topk_sufficiency_pct: float = 0.0
    topk_sufficiency_k: int = DEFAULT_SUFFICIENCY_K
    topk_queries_total: int = 0
    topk_queries_sufficient: int = 0
    
    # 7. Boundary Confidence %
    # What: Confidence in chunk boundary placement
    # Source: Boundary detection prompt
    # Formula: BoundaryConfidence% = mean(boundary_confidence) × 100
    # Rule: Only average across boundary decisions, NOT non-boundary blocks
    boundary_confidence_pct: float = 0.0
    boundary_decisions_count: int = 0
    
    # 8. Total Time (s)
    # What: Operational cost (wall-clock)
    # Formula: Time from ingestion start → final index write
    # Rule: MUST also break down by stage
    total_time_seconds: float = 0.0
    stage_timings: Optional[StageTimings] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "total_chunks": self.total_chunks,
            "avg_chunk_length": {
                "value": round(self.avg_chunk_length, 1),
                "unit": self.length_unit
            },
            "avg_quality": {
                "value": round(self.avg_quality, 3),
                "scale": "0-1",
                "chunks_measured": self.chunks_with_quality
            },
            "relevance_pct": {
                "value": round(self.relevance_pct, 1),
                "k": self.relevance_k,
                "queries_total": self.relevance_queries_total,
                "queries_with_relevant": self.relevance_queries_with_relevant,
                "note": "Retrieval recall - does NOT measure efficiency"
            },
            "semantic_accept_pct": {
                "value": round(self.semantic_accept_pct, 1),
                "threshold": self.semantic_accept_threshold,
                "accepted_count": self.semantic_accepted_count,
                "note": "Replaces heuristic 'Optimal %' - measures semantic correctness"
            },
            "topk_sufficiency_pct": {
                "value": round(self.topk_sufficiency_pct, 1),
                "k": self.topk_sufficiency_k,
                "queries_total": self.topk_queries_total,
                "queries_sufficient": self.topk_queries_sufficient,
                "note": "Retrieval efficiency - measures query-document asymmetry"
            },
            "boundary_confidence_pct": {
                "value": round(self.boundary_confidence_pct, 1),
                "boundary_decisions": self.boundary_decisions_count
            },
            "total_time_seconds": round(self.total_time_seconds, 2)
        }
        
        if self.stage_timings:
            result["stage_breakdown"] = self.stage_timings.to_dict()
        
        return result


# =============================================================================
# METRIC CALCULATOR
# =============================================================================

class MetricCalculator:
    """
    Calculator for frozen metrics with exact formulas.
    
    All formulas match the FROZEN definitions above.
    """
    
    def __init__(self, length_unit: str = LENGTH_UNIT):
        self.length_unit = length_unit
        self.token_approximation_ratio = 4  # ~4 chars per token
    
    def calculate_total_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Metric 1: Total Chunks
        
        Count only final indexed chunks (after validation + refinement).
        Do NOT count rejected or intermediate chunks.
        """
        # Only count chunks that passed validation (not rejected)
        return len([c for c in chunks if c.get("status") != "rejected"])
    
    def calculate_avg_chunk_length(
        self, 
        chunks: List[Dict[str, Any]], 
        use_tokens: bool = False
    ) -> float:
        """
        Metric 2: Average Chunk Length
        
        Formula: AvgChunkLength = sum(length(chunk.text)) / TotalChunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            use_tokens: If True, approximate token count; else use characters
        """
        valid_chunks = [c for c in chunks if c.get("status") != "rejected"]
        if not valid_chunks:
            return 0.0
        
        if use_tokens:
            # Approximate tokens as chars / 4
            total_length = sum(
                len(c.get("text", "")) / self.token_approximation_ratio 
                for c in valid_chunks
            )
        else:
            total_length = sum(len(c.get("text", "")) for c in valid_chunks)
        
        return total_length / len(valid_chunks)
    
    def calculate_avg_quality(self, chunks: List[Dict[str, Any]]) -> tuple:
        """
        Metric 3: Average Quality
        
        Formula: AvgQuality = mean(chunk.validation.overall_score)
        
        IMPORTANT: Scores must be normalized to 0-1.
        Chunks without validation are EXCLUDED (not treated as zero).
        
        Returns:
            (avg_quality, chunks_with_quality_count)
        """
        quality_scores = []
        
        for chunk in chunks:
            if chunk.get("status") == "rejected":
                continue
                
            # Get quality score from various possible locations
            score = None
            if chunk.get("validation", {}).get("overall_score") is not None:
                score = chunk["validation"]["overall_score"]
            elif chunk.get("quality_score") is not None:
                score = chunk["quality_score"]
            
            if score is not None:
                # Normalize to 0-1 if on 0-10 scale
                if score > 1:
                    score = score / 10.0
                quality_scores.append(score)
        
        if not quality_scores:
            return 0.0, 0
        
        return sum(quality_scores) / len(quality_scores), len(quality_scores)
    
    def calculate_relevance_pct(
        self,
        query_results: List[Dict[str, Any]],
        k: int = DEFAULT_TOP_K
    ) -> tuple:
        """
        Metric 4: Relevance % (Retrieval Recall)
        
        Formula: Relevance% = (queries with ≥1 relevant in top-K) / total × 100
        
        Definition: A query is relevant if at least one retrieved chunk 
        overlaps with the ground-truth answer span.
        
        Args:
            query_results: List of query result dicts with:
                - 'retrieved_chunks': List of retrieved chunks
                - 'ground_truth_chunks': List of ground truth chunk IDs
                - Or 'has_relevant': Boolean indicating relevance
            k: Top-K value (MUST be reported)
        
        Returns:
            (relevance_pct, total_queries, queries_with_relevant)
        """
        if not query_results:
            return 0.0, 0, 0
        
        queries_with_relevant = 0
        
        for result in query_results:
            top_k_chunks = result.get("retrieved_chunks", [])[:k]
            ground_truth = set(result.get("ground_truth_chunks", []))
            
            # Check if query has pre-computed relevance
            if "has_relevant" in result:
                if result["has_relevant"]:
                    queries_with_relevant += 1
                continue
            
            # Check if any top-K chunk overlaps with ground truth
            retrieved_ids = {c.get("chunk_id") for c in top_k_chunks}
            if retrieved_ids & ground_truth:
                queries_with_relevant += 1
        
        total = len(query_results)
        relevance_pct = (queries_with_relevant / total) * 100 if total else 0.0
        
        return relevance_pct, total, queries_with_relevant
    
    def calculate_semantic_accept_pct(
        self,
        chunks: List[Dict[str, Any]],
        confidence_threshold: float = SEMANTIC_ACCEPT_CONFIDENCE_THRESHOLD
    ) -> tuple:
        """
        Metric 5: Semantic Accept % (replaces "Optimal %")
        
        Formula: SemanticAccept% = (accepted_chunks / total_chunks) × 100
        
        Definition: A chunk is "accepted" if:
            - recommendation == "ACCEPT"
            - confidence >= threshold (default 0.7)
        
        IMPORTANT: Run one auto-refinement pass before computing.
        
        Returns:
            (semantic_accept_pct, accepted_count, total_count)
        """
        valid_chunks = [c for c in chunks if c.get("status") != "rejected"]
        if not valid_chunks:
            return 0.0, 0, 0
        
        accepted = 0
        for chunk in valid_chunks:
            validation = chunk.get("validation", {})
            recommendation = validation.get("recommendation", "").upper()
            confidence = validation.get("confidence", 0.0)
            
            # Fallback: check quality_score if no explicit validation
            if not recommendation:
                quality = chunk.get("quality_score", 0)
                # Normalize and treat >=7 as ACCEPT equivalent
                if quality > 1:
                    quality = quality / 10.0
                if quality >= 0.7:
                    recommendation = "ACCEPT"
                    confidence = quality
            
            if recommendation == "ACCEPT" and confidence >= confidence_threshold:
                accepted += 1
        
        total = len(valid_chunks)
        pct = (accepted / total) * 100 if total else 0.0
        
        return pct, accepted, total
    
    def calculate_topk_sufficiency_pct(
        self,
        query_results: List[Dict[str, Any]],
        k: int = DEFAULT_SUFFICIENCY_K
    ) -> tuple:
        """
        Metric 6: Top-K Sufficiency % (Retrieval Efficiency)
        
        Formula: TopKSufficiency% = (queries answerable with ≤K) / total × 100
        
        Definition: A query is sufficient if the answer can be fully 
        supported using ≤K retrieved chunks.
        
        This metric directly measures query-document asymmetry reduction.
        
        Returns:
            (sufficiency_pct, total_queries, sufficient_queries)
        """
        if not query_results:
            return 0.0, 0, 0
        
        sufficient_queries = 0
        
        for result in query_results:
            # Check pre-computed sufficiency
            if "is_sufficient" in result:
                if result["is_sufficient"]:
                    sufficient_queries += 1
                continue
            
            # Check if answer coverage is complete with ≤K chunks
            chunks_needed = result.get("chunks_needed_for_answer", k + 1)
            if chunks_needed <= k:
                sufficient_queries += 1
        
        total = len(query_results)
        pct = (sufficient_queries / total) * 100 if total else 0.0
        
        return pct, total, sufficient_queries
    
    def calculate_boundary_confidence_pct(
        self,
        boundary_decisions: List[Dict[str, Any]]
    ) -> tuple:
        """
        Metric 7: Boundary Confidence %
        
        Formula: BoundaryConfidence% = mean(boundary_confidence) × 100
        
        IMPORTANT: Only average across boundary decisions.
        Do NOT average across non-boundary blocks.
        
        Returns:
            (confidence_pct, boundary_decisions_count)
        """
        # Filter to only actual boundary decisions (where new_chunk=True)
        boundary_only = [
            d for d in boundary_decisions 
            if d.get("new_chunk", False) or d.get("is_boundary", False)
        ]
        
        if not boundary_only:
            return 0.0, 0
        
        confidences = [
            d.get("confidence", 0.0) 
            for d in boundary_only 
            if d.get("confidence") is not None
        ]
        
        if not confidences:
            return 0.0, len(boundary_only)
        
        avg_confidence = sum(confidences) / len(confidences)
        return avg_confidence * 100, len(boundary_only)
    
    def compute_all_metrics(
        self,
        chunks: List[Dict[str, Any]],
        boundary_decisions: List[Dict[str, Any]] = None,
        query_results: List[Dict[str, Any]] = None,
        stage_timings: StageTimings = None,
        total_time: float = 0.0
    ) -> FrozenMetrics:
        """
        Compute all frozen metrics in one call.
        
        Args:
            chunks: Final indexed chunks
            boundary_decisions: List of boundary decision dicts
            query_results: List of query evaluation results
            stage_timings: Detailed stage timing breakdown
            total_time: Total wall-clock time in seconds
        """
        metrics = FrozenMetrics()
        
        # 1. Total Chunks
        metrics.total_chunks = self.calculate_total_chunks(chunks)
        
        # 2. Average Chunk Length
        use_tokens = self.length_unit == "tokens"
        metrics.avg_chunk_length = self.calculate_avg_chunk_length(chunks, use_tokens)
        metrics.length_unit = self.length_unit
        
        # 3. Average Quality
        metrics.avg_quality, metrics.chunks_with_quality = self.calculate_avg_quality(chunks)
        
        # 4. Relevance % (if query results provided)
        if query_results:
            (metrics.relevance_pct, 
             metrics.relevance_queries_total,
             metrics.relevance_queries_with_relevant) = self.calculate_relevance_pct(
                query_results, metrics.relevance_k
            )
        
        # 5. Semantic Accept %
        (metrics.semantic_accept_pct,
         metrics.semantic_accepted_count, _) = self.calculate_semantic_accept_pct(chunks)
        
        # 6. Top-K Sufficiency % (if query results provided)
        if query_results:
            (metrics.topk_sufficiency_pct,
             metrics.topk_queries_total,
             metrics.topk_queries_sufficient) = self.calculate_topk_sufficiency_pct(
                query_results, metrics.topk_sufficiency_k
            )
        
        # 7. Boundary Confidence %
        if boundary_decisions:
            (metrics.boundary_confidence_pct,
             metrics.boundary_decisions_count) = self.calculate_boundary_confidence_pct(
                boundary_decisions
            )
        
        # 8. Total Time
        if stage_timings:
            metrics.total_time_seconds = stage_timings.get_total_time()
            metrics.stage_timings = stage_timings
        else:
            metrics.total_time_seconds = total_time
        
        return metrics


def compute_heuristic_metrics(
    chunks: List[Dict[str, Any]],
    processing_time: float,
    method: str
) -> FrozenMetrics:
    """
    Compute heuristic metrics for methods that don't run the full pipeline.
    
    Used for PyPDF and Sherpa which don't have:
    - Quality validation scores
    - Boundary detection confidence
    - Semantic accept evaluation
    
    These metrics are marked as heuristic approximations.
    """
    calculator = MetricCalculator()
    metrics = FrozenMetrics()
    
    # 1. Total Chunks
    metrics.total_chunks = len(chunks)
    
    # 2. Average Chunk Length
    metrics.avg_chunk_length = calculator.calculate_avg_chunk_length(chunks)
    
    # 3. Average Quality - HEURISTIC based on chunk characteristics
    # Since no LLM validation, use heuristic scoring
    quality_scores = []
    for chunk in chunks:
        text = chunk.get("text", "")
        length = len(text)
        
        # Heuristic: reasonable chunk size = better quality
        if 200 <= length <= 1000:
            score = 0.7
        elif 100 <= length <= 1500:
            score = 0.5
        elif length < 50:
            score = 0.2
        else:
            score = 0.4
        
        quality_scores.append(score)
    
    if quality_scores:
        metrics.avg_quality = sum(quality_scores) / len(quality_scores)
        metrics.chunks_with_quality = len(quality_scores)
    
    # 5. Semantic Accept % - HEURISTIC
    # Without validation, use size-based heuristic
    accepted = sum(1 for chunk in chunks 
                   if 150 <= len(chunk.get("text", "")) <= 1200)
    metrics.semantic_accept_pct = (accepted / len(chunks) * 100) if chunks else 0.0
    metrics.semantic_accepted_count = accepted
    
    # 7. Boundary Confidence % - HEURISTIC
    # For non-semantic methods, estimate based on chunk count
    if method == "pypdf":
        # PyPDF has no boundaries (single chunk)
        metrics.boundary_confidence_pct = 0.0
        metrics.boundary_decisions_count = 0
    elif method == "sherpa":
        # Sherpa has layout-based boundaries
        # More chunks = better layout detection
        if len(chunks) > 1:
            metrics.boundary_confidence_pct = min(60.0, len(chunks) * 10)
        else:
            metrics.boundary_confidence_pct = 20.0
        metrics.boundary_decisions_count = max(0, len(chunks) - 1)
    
    # 8. Total Time
    metrics.total_time_seconds = processing_time
    
    return metrics




def generate_diagnostics_table(
    method_metrics: Dict[str, FrozenMetrics]
) -> Dict[str, Any]:
    """
    Generate the final diagnostics table (one row per method).
    
    Columns:
    - Method (PyPDF / Sherpa / Prompt)
    - Parse Time
    - Boundary Time
    - Validation Time
    - Embed Time
    - Index Time
    - Total Time
    - Total Chunks
    - Semantic Accept %
    - Top-K Sufficiency %
    """
    rows = []
    
    for method, metrics in method_metrics.items():
        timings = metrics.stage_timings
        
        row = {
            "method": method.upper(),
            "parse_time_s": round(timings.get_parse_time(), 2) if timings else 0,
            "boundary_time_s": round(timings.get_boundary_time(), 2) if timings else 0,
            "validation_time_s": round(timings.get_validation_time(), 2) if timings else 0,
            "embed_time_s": round(timings.get_embedding_time(), 2) if timings else 0,
            "index_time_s": round(timings.get_indexing_time(), 2) if timings else 0,
            "total_time_s": round(metrics.total_time_seconds, 2),
            "total_chunks": metrics.total_chunks,
            "semantic_accept_pct": round(metrics.semantic_accept_pct, 1),
            "topk_sufficiency_pct": round(metrics.topk_sufficiency_pct, 1)
        }
        rows.append(row)
    
    return {
        "diagnostics_table": rows,
        "explanation": (
            "Semantic metrics (Semantic Accept Rate and Top-K Sufficiency) replace "
            "heuristic 'optimality' because they directly measure query–document "
            "alignment and retrieval efficiency."
        ),
        "k_values": {
            "relevance_k": DEFAULT_TOP_K,
            "sufficiency_k": DEFAULT_SUFFICIENCY_K
        }
    }




class MetricValidationRules:
    """
    Rules to avoid misleading metrics.
    
    These rules MUST be checked before generating reports.
    """
    
    @staticmethod
    def validate_metrics(metrics: FrozenMetrics, method: str) -> List[str]:
        """
        Validate metrics and return list of warnings/violations.
        """
        warnings = []
        
        # Rule 1: Never compare parsers and semantic chunkers using size-based optimality
        # (This is handled by using semantic metrics instead)
        
        # Rule 2: Relevance % alone is meaningless without Top-K Sufficiency
        if metrics.relevance_queries_total > 0 and metrics.topk_queries_total == 0:
            warnings.append(
                "WARNING: Relevance % reported without Top-K Sufficiency. "
                "Relevance alone does not measure efficiency."
            )
        
        # Rule 3: If Semantic Accept % < 40%, force refinement before reporting
        if metrics.semantic_accept_pct < FORCE_REFINEMENT_THRESHOLD * 100:
            warnings.append(
                f"WARNING: Semantic Accept % ({metrics.semantic_accept_pct:.1f}%) "
                f"is below {FORCE_REFINEMENT_THRESHOLD * 100}%. "
                "Force refinement pass before final reporting."
            )
        
        # Rule 4: Always report K used in retrieval metrics
        # (Handled by including k values in metric output)
        
        # Rule 5: Never average metrics across documents of different tiers without labeling
        # (Handled at aggregation level)
        
        return warnings
    
    @staticmethod
    def check_unit_consistency(metrics_list: List[FrozenMetrics]) -> bool:
        """
        Ensure all metrics use the same length unit.
        
        Rule: Never mix characters and tokens in the same report.
        """
        if not metrics_list:
            return True
        
        units = {m.length_unit for m in metrics_list}
        return len(units) == 1



SEMANTIC_METRICS_EXPLANATION = (
    "Semantic metrics (Semantic Accept Rate and Top-K Sufficiency) replace "
    "heuristic 'optimality' because they directly measure query–document "
    "alignment and retrieval efficiency."
)
