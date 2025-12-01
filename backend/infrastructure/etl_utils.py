"""
ETL Utility Functions
=====================
Helper functions for easy ETL pipeline integration
"""

import logging
from typing import Dict, Any, Optional
from .data_etl_pipeline import DataETLPipeline, ETLResult
from .data_quality import DataQualityChecker, get_quality_score_label

logger = logging.getLogger(__name__)

# Global ETL pipeline instance (singleton pattern)
_etl_pipeline: Optional[DataETLPipeline] = None
_quality_checker: Optional[DataQualityChecker] = None


def get_etl_pipeline(config: Dict[str, Any] = None) -> DataETLPipeline:
    """Get or create global ETL pipeline instance"""
    global _etl_pipeline
    if _etl_pipeline is None:
        _etl_pipeline = DataETLPipeline(config or {})
    return _etl_pipeline


def get_quality_checker(config: Dict[str, Any] = None) -> DataQualityChecker:
    """Get or create global quality checker instance"""
    global _quality_checker
    if _quality_checker is None:
        _quality_checker = DataQualityChecker(config or {})
    return _quality_checker


def process_ui_state(ui_state: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
    """
    Process UI state through ETL pipeline
    
    Args:
        ui_state: Raw UI state dictionary
        use_cache: Whether to use cache
    
    Returns:
        Processed data dictionary with normalized_ui_state
    """
    pipeline = get_etl_pipeline()
    result = pipeline.process(
        source_type='ui_state',
        source_data=ui_state,
        load_targets=['memory', 'cache'] if use_cache else ['memory'],
        use_cache=use_cache
    )
    
    if result.success:
        return result.data
    else:
        logger.warning(f"ETL processing failed: {result.errors}")
        return {'ui_state': ui_state}  # Return original as fallback


def process_action_plan(action_plan: Dict[str, Any], instruction: str = "") -> Dict[str, Any]:
    """
    Process action plan through ETL pipeline
    
    Args:
        action_plan: Action plan dictionary
        instruction: Task instruction (optional)
    
    Returns:
        Processed data dictionary with normalized_action_plan
    """
    pipeline = get_etl_pipeline()
    result = pipeline.process(
        source_type='action_plan',
        source_data={'action_plan': action_plan, 'instruction': instruction},
        load_targets=['memory']
    )
    
    if result.success:
        return result.data
    else:
        logger.warning(f"ETL processing failed: {result.errors}")
        return {'action_plan': action_plan}  # Return original as fallback


def process_task(instruction: str, device_id: str = "", parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process task instruction through ETL pipeline
    
    Args:
        instruction: Task instruction text
        device_id: Device ID (optional)
        parameters: Task parameters (optional)
    
    Returns:
        Processed data dictionary with normalized_task
    """
    pipeline = get_etl_pipeline()
    task_data = {
        'instruction': instruction,
        'device_id': device_id,
        'parameters': parameters or {}
    }
    
    result = pipeline.process(
        source_type='task_instruction',
        source_data=instruction,
        load_targets=['memory']
    )
    
    if result.success:
        # Merge device_id and parameters
        if 'normalized_task' in result.data:
            result.data['normalized_task']['device_id'] = device_id
            result.data['normalized_task']['parameters'] = parameters or {}
        return result.data
    else:
        logger.warning(f"ETL processing failed: {result.errors}")
        return {'task': task_data}  # Return original as fallback


def assess_data_quality(data: Dict[str, Any], data_type: str = "general") -> Dict[str, Any]:
    """
    Assess data quality
    
    Args:
        data: Data dictionary to assess
        data_type: Type of data (ui_state, action_plan, task, etc.)
    
    Returns:
        Quality assessment dictionary with score and issues
    """
    checker = get_quality_checker()
    score, issues = checker.assess_quality(data, data_type)
    
    return {
        'score': score,
        'label': get_quality_score_label(score),
        'issues': [
            {
                'type': issue.issue_type.value if issue.issue_type else None,
                'severity': issue.severity,
                'message': issue.message,
                'suggested_fix': issue.suggested_fix,
                'score': issue.score
            }
            for issue in issues
        ],
        'passed_checks': sum(1 for issue in issues if issue.passed),
        'total_checks': len(issues)
    }


def quick_etl_process(source_type: str, source_data: Any, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Quick ETL processing with sensible defaults
    
    Args:
        source_type: Type of source (ui_state, action_plan, task_instruction, etc.)
        source_data: Source data
        **kwargs: Additional arguments
    
    Returns:
        Processed data or None if processing failed
    """
    pipeline = get_etl_pipeline()
    
    # Determine load targets based on source type
    load_targets = kwargs.pop('load_targets', None)
    if load_targets is None:
        if source_type == 'ui_state':
            load_targets = ['memory', 'cache']
        else:
            load_targets = ['memory']
    
    result = pipeline.process(
        source_type=source_type,
        source_data=source_data,
        load_targets=load_targets,
        **kwargs
    )
    
    if result.success:
        return result.data
    else:
        logger.error(f"Quick ETL processing failed: {result.errors}")
        return None

