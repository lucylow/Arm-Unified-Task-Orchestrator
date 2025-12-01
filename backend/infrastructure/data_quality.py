"""
Data Quality Checks for AI Agent ETL Pipeline
==============================================
Comprehensive data quality assessment and validation
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class QualityIssue(Enum):
    """Types of quality issues"""
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"
    INCONSISTENT_DATA = "inconsistent_data"
    LOW_CONFIDENCE = "low_confidence"
    DUPLICATE_DATA = "duplicate_data"
    INCOMPLETE_DATA = "incomplete_data"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


@dataclass
class QualityCheckResult:
    """Result of a quality check"""
    passed: bool
    issue_type: Optional[QualityIssue] = None
    severity: str = "low"  # low, medium, high, critical
    message: str = ""
    suggested_fix: Optional[str] = None
    score: float = 1.0  # 0.0 to 1.0


class DataQualityChecker:
    """Comprehensive data quality checker"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.checks: List[callable] = []
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default quality checks"""
        self.checks = [
            self.check_completeness,
            self.check_consistency,
            self.check_validity,
            self.check_confidence_levels,
            self.check_data_freshness,
            self.check_duplicates,
            self.check_suspicious_patterns
        ]
    
    def assess_quality(self, data: Dict[str, Any], data_type: str = "general") -> tuple:
        """
        Assess overall data quality
        
        Returns:
            Tuple of (quality_score, list_of_issues)
            quality_score: 0.0 to 1.0 (1.0 is perfect)
        """
        results = []
        
        for check in self.checks:
            try:
                result = check(data, data_type)
                if isinstance(result, QualityCheckResult):
                    results.append(result)
                elif isinstance(result, list):
                    results.extend(result)
            except Exception as e:
                logger.warning(f"Quality check {check.__name__} failed: {e}")
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=None,
                    severity="medium",
                    message=f"Check failed: {e}",
                    score=0.5
                ))
        
        # Calculate overall score
        if not results:
            return 1.0, []
        
        total_score = sum(r.score for r in results)
        avg_score = total_score / len(results)
        
        return avg_score, results
    
    def check_completeness(self, data: Dict[str, Any], data_type: str) -> List[QualityCheckResult]:
        """Check if all required fields are present"""
        results = []
        required_fields = self._get_required_fields(data_type)
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=QualityIssue.MISSING_REQUIRED_FIELD,
                    severity="high",
                    message=f"Missing required field: {field}",
                    suggested_fix=f"Provide value for {field}",
                    score=0.0
                ))
        
        if not results:
            results.append(QualityCheckResult(
                passed=True,
                message="All required fields present",
                score=1.0
            ))
        
        return results
    
    def check_consistency(self, data: Dict[str, Any], data_type: str) -> List[QualityCheckResult]:
        """Check data consistency"""
        results = []
        
        # Check UI state consistency
        if 'normalized_ui_state' in data or 'ui_state' in data:
            ui_state = data.get('normalized_ui_state') or data.get('ui_state', {})
            elements = ui_state.get('elements', [])
            element_count = ui_state.get('element_count', len(elements))
            
            if element_count != len(elements):
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=QualityIssue.INCONSISTENT_DATA,
                    severity="medium",
                    message=f"Element count mismatch: {element_count} vs {len(elements)}",
                    suggested_fix="Recalculate element_count",
                    score=0.7
                ))
        
        # Check action plan consistency
        if 'normalized_action_plan' in data or 'action_plan' in data:
            plan = data.get('normalized_action_plan') or data.get('action_plan', {})
            steps = plan.get('steps', [])
            total_steps = plan.get('total_steps', len(steps))
            
            if total_steps != len(steps):
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=QualityIssue.INCONSISTENT_DATA,
                    severity="medium",
                    message=f"Step count mismatch: {total_steps} vs {len(steps)}",
                    suggested_fix="Recalculate total_steps",
                    score=0.7
                ))
        
        if not results:
            results.append(QualityCheckResult(
                passed=True,
                message="Data is consistent",
                score=1.0
            ))
        
        return results
    
    def check_validity(self, data: Dict[str, Any], data_type: str) -> List[QualityCheckResult]:
        """Check data validity and format"""
        results = []
        
        # Check UI elements validity
        if 'normalized_ui_state' in data or 'ui_state' in data:
            ui_state = data.get('normalized_ui_state') or data.get('ui_state', {})
            elements = ui_state.get('elements', [])
            
            for i, elem in enumerate(elements):
                if not isinstance(elem, dict):
                    results.append(QualityCheckResult(
                        passed=False,
                        issue_type=QualityIssue.INVALID_FORMAT,
                        severity="high",
                        message=f"Element {i} is not a dictionary",
                        score=0.0
                    ))
                    continue
                
                # Check bbox format
                if 'bbox' in elem:
                    bbox = elem['bbox']
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        results.append(QualityCheckResult(
                            passed=False,
                            issue_type=QualityIssue.INVALID_FORMAT,
                            severity="medium",
                            message=f"Element {i} has invalid bbox format",
                            suggested_fix="Bbox should be [x, y, width, height]",
                            score=0.5
                        ))
                
                # Check confidence range
                if 'confidence' in elem:
                    conf = elem['confidence']
                    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                        results.append(QualityCheckResult(
                            passed=False,
                            issue_type=QualityIssue.OUT_OF_RANGE,
                            severity="medium",
                            message=f"Element {i} has invalid confidence: {conf}",
                            suggested_fix="Confidence should be between 0 and 1",
                            score=0.5
                        ))
        
        # Check action plan validity
        if 'normalized_action_plan' in data or 'action_plan' in data:
            plan = data.get('normalized_action_plan') or data.get('action_plan', {})
            steps = plan.get('steps', [])
            
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    results.append(QualityCheckResult(
                        passed=False,
                        issue_type=QualityIssue.INVALID_FORMAT,
                        severity="high",
                        message=f"Step {i} is not a dictionary",
                        score=0.0
                    ))
                    continue
                
                if 'action' not in step:
                    results.append(QualityCheckResult(
                        passed=False,
                        issue_type=QualityIssue.MISSING_REQUIRED_FIELD,
                        severity="high",
                        message=f"Step {i} missing required 'action' field",
                        score=0.0
                    ))
        
        if not results:
            results.append(QualityCheckResult(
                passed=True,
                message="Data format is valid",
                score=1.0
            ))
        
        return results
    
    def check_confidence_levels(self, data: Dict[str, Any], data_type: str) -> List[QualityCheckResult]:
        """Check confidence levels in data"""
        results = []
        low_confidence_threshold = self.config.get('low_confidence_threshold', 0.6)
        
        # Check UI element confidence
        if 'normalized_ui_state' in data or 'ui_state' in data:
            ui_state = data.get('normalized_ui_state') or data.get('ui_state', {})
            elements = ui_state.get('elements', [])
            
            low_confidence_elements = [
                i for i, elem in enumerate(elements)
                if elem.get('confidence', 1.0) < low_confidence_threshold
            ]
            
            if low_confidence_elements:
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=QualityIssue.LOW_CONFIDENCE,
                    severity="low",
                    message=f"{len(low_confidence_elements)} elements have low confidence",
                    suggested_fix="Review detection model or improve image quality",
                    score=0.8
                ))
        
        # Check action plan confidence
        if 'normalized_action_plan' in data or 'action_plan' in data:
            plan = data.get('normalized_action_plan') or data.get('action_plan', {})
            overall_conf = plan.get('overall_confidence', 1.0)
            
            if overall_conf < low_confidence_threshold:
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=QualityIssue.LOW_CONFIDENCE,
                    severity="medium",
                    message=f"Action plan has low overall confidence: {overall_conf:.2f}",
                    suggested_fix="Review plan generation or provide more context",
                    score=0.6
                ))
        
        if not results:
            results.append(QualityCheckResult(
                passed=True,
                message="Confidence levels are acceptable",
                score=1.0
            ))
        
        return results
    
    def check_data_freshness(self, data: Dict[str, Any], data_type: str) -> List[QualityCheckResult]:
        """Check if data is fresh (not stale)"""
        results = []
        max_age_seconds = self.config.get('max_data_age_seconds', 300)  # 5 minutes default
        
        # Check timestamps
        timestamps = []
        if 'normalized_ui_state' in data:
            ts = data['normalized_ui_state'].get('timestamp')
            if ts:
                timestamps.append(ts)
        
        if 'normalized_task' in data:
            ts = data['normalized_task'].get('created_at')
            if ts:
                timestamps.append(ts)
        
        # This is a simplified check - in production, parse timestamps properly
        if timestamps:
            results.append(QualityCheckResult(
                passed=True,
                message="Data freshness check passed",
                score=1.0
            ))
        else:
            results.append(QualityCheckResult(
                passed=False,
                issue_type=QualityIssue.INCOMPLETE_DATA,
                severity="low",
                message="Missing timestamp information",
                score=0.9
            ))
        
        return results
    
    def check_duplicates(self, data: Dict[str, Any], data_type: str) -> List[QualityCheckResult]:
        """Check for duplicate data"""
        results = []
        
        # Check for duplicate UI elements
        if 'normalized_ui_state' in data or 'ui_state' in data:
            ui_state = data.get('normalized_ui_state') or data.get('ui_state', {})
            elements = ui_state.get('elements', [])
            
            element_ids = [elem.get('id') for elem in elements if elem.get('id')]
            duplicates = [id for id in element_ids if element_ids.count(id) > 1]
            
            if duplicates:
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=QualityIssue.DUPLICATE_DATA,
                    severity="low",
                    message=f"Found {len(set(duplicates))} duplicate element IDs",
                    suggested_fix="Remove or rename duplicate elements",
                    score=0.8
                ))
        
        if not results:
            results.append(QualityCheckResult(
                passed=True,
                message="No duplicates found",
                score=1.0
            ))
        
        return results
    
    def check_suspicious_patterns(self, data: Dict[str, Any], data_type: str) -> List[QualityCheckResult]:
        """Check for suspicious patterns"""
        results = []
        
        # Check for empty or very short instructions
        if 'normalized_task' in data or 'task' in data:
            task = data.get('normalized_task') or data.get('task', {})
            instruction = task.get('instruction', '')
            
            if len(instruction.strip()) < 3:
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=QualityIssue.SUSPICIOUS_PATTERN,
                    severity="medium",
                    message="Instruction is too short",
                    suggested_fix="Provide a more detailed instruction",
                    score=0.5
                ))
        
        # Check for empty action plans
        if 'normalized_action_plan' in data or 'action_plan' in data:
            plan = data.get('normalized_action_plan') or data.get('action_plan', {})
            steps = plan.get('steps', [])
            
            if not steps:
                results.append(QualityCheckResult(
                    passed=False,
                    issue_type=QualityIssue.INCOMPLETE_DATA,
                    severity="high",
                    message="Action plan has no steps",
                    suggested_fix="Generate a valid action plan",
                    score=0.0
                ))
        
        if not results:
            results.append(QualityCheckResult(
                passed=True,
                message="No suspicious patterns detected",
                score=1.0
            ))
        
        return results
    
    def _get_required_fields(self, data_type: str) -> List[str]:
        """Get required fields for data type"""
        field_map = {
            'ui_state': ['elements', 'timestamp'],
            'action_plan': ['steps', 'instruction'],
            'task': ['instruction', 'task_id']
        }
        return field_map.get(data_type, [])


def get_quality_score_label(score: float) -> str:
    """Get human-readable quality label"""
    if score >= 0.9:
        return "excellent"
    elif score >= 0.7:
        return "good"
    elif score >= 0.5:
        return "acceptable"
    elif score >= 0.3:
        return "poor"
    else:
        return "invalid"

