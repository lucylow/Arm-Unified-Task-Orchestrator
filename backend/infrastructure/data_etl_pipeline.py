"""
Enhanced Data ETL Pipeline for AI Agents
========================================
Provides Extract, Transform, Load capabilities with:
- Multi-source data extraction
- Schema validation and normalization
- Data enrichment and quality checks
- Caching and versioning
- Error handling and recovery
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import time
from pathlib import Path
import pickle
import gzip

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class ETLStage(Enum):
    """ETL pipeline stages"""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"


@dataclass
class DataMetadata:
    """Metadata for data records"""
    source: str
    timestamp: datetime
    version: str
    quality: DataQuality
    schema_version: str
    checksum: str
    transformations: List[str]
    enrichment_applied: bool = False
    validation_passed: bool = False
    error_count: int = 0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ETLResult:
    """Result of ETL operation"""
    success: bool
    stage: ETLStage
    data: Any
    metadata: DataMetadata
    errors: List[str]
    warnings: List[str]
    processing_time_ms: float
    cache_hit: bool = False
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class DataExtractor:
    """Extracts data from various sources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.extractors: Dict[str, Callable] = {}
        self._register_default_extractors()
    
    def _register_default_extractors(self):
        """Register default data extractors"""
        self.extractors = {
            'screenshot': self._extract_screenshot,
            'ui_state': self._extract_ui_state,
            'task_instruction': self._extract_task_instruction,
            'memory': self._extract_memory,
            'device_info': self._extract_device_info,
            'user_context': self._extract_user_context,
        }
    
    def extract(self, source_type: str, source_data: Any, **kwargs) -> Dict[str, Any]:
        """Extract data from specified source"""
        extractor = self.extractors.get(source_type)
        if not extractor:
            raise ValueError(f"Unknown extractor type: {source_type}")
        
        try:
            raw_data = extractor(source_data, **kwargs)
            return {
                'source_type': source_type,
                'raw_data': raw_data,
                'extracted_at': datetime.utcnow().isoformat(),
                'extraction_params': kwargs
            }
        except Exception as e:
            logger.error(f"Extraction failed for {source_type}: {e}")
            raise
    
    def _extract_screenshot(self, screenshot_path: str, **kwargs) -> Dict[str, Any]:
        """Extract data from screenshot"""
        from PIL import Image
        import os
        
        if not os.path.exists(screenshot_path):
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
        
        image = Image.open(screenshot_path)
        return {
            'path': screenshot_path,
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'file_size_bytes': os.path.getsize(screenshot_path),
            'timestamp': os.path.getmtime(screenshot_path)
        }
    
    def _extract_ui_state(self, ui_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Extract structured data from UI state"""
        return {
            'elements': ui_state.get('ui_elements', []),
            'ocr_text': ui_state.get('full_text_ocr', ''),
            'screenshot_path': ui_state.get('screenshot_path', ''),
            'device_info': ui_state.get('device_info', {}),
            'timestamp': ui_state.get('timestamp', ''),
            'element_count': len(ui_state.get('ui_elements', []))
        }
    
    def _extract_task_instruction(self, instruction: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """Extract task instruction data"""
        if isinstance(instruction, dict):
            return instruction
        return {
            'text': instruction,
            'length': len(instruction),
            'word_count': len(instruction.split()),
            'has_parameters': '{' in instruction or '[' in instruction
        }
    
    def _extract_memory(self, memory_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Extract data from memory/episodes"""
        return {
            'episodes': memory_data.get('episodes', []),
            'episode_count': len(memory_data.get('episodes', [])),
            'similar_episodes': memory_data.get('similar_episodes', []),
            'success_rate': memory_data.get('success_rate', 0.0)
        }
    
    def _extract_device_info(self, device_data: Union[Dict, Any], **kwargs) -> Dict[str, Any]:
        """Extract device information"""
        if isinstance(device_data, dict):
            return device_data
        return {
            'device_id': getattr(device_data, 'device_id', None),
            'capabilities': getattr(device_data, 'get_capabilities', lambda: {})()
        }
    
    def _extract_user_context(self, user_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Extract user context data"""
        return {
            'username': user_data.get('username', ''),
            'subscription': user_data.get('subscription', ''),
            'quota': user_data.get('quota', 0),
            'tasks_used': user_data.get('tasks_used', 0),
            'history_count': len(user_data.get('history', []))
        }
    
    def register_extractor(self, name: str, extractor_func: Callable):
        """Register a custom extractor"""
        self.extractors[name] = extractor_func


class DataTransformer:
    """Transforms and normalizes data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.transformers: Dict[str, Callable] = {}
        self.validators: Dict[str, Callable] = {}
        self.enrichers: List[Callable] = []
        self._register_default_transformers()
    
    def _register_default_transformers(self):
        """Register default transformers"""
        self.transformers = {
            'normalize_ui_state': self._normalize_ui_state,
            'normalize_action_plan': self._normalize_action_plan,
            'normalize_task': self._normalize_task,
            'sanitize_text': self._sanitize_text,
            'enrich_with_metadata': self._enrich_with_metadata,
        }
        
        self.validators = {
            'validate_ui_state': self._validate_ui_state,
            'validate_action_plan': self._validate_action_plan,
            'validate_task': self._validate_task,
        }
    
    def transform(self, data: Dict[str, Any], transformations: List[str] = None) -> Dict[str, Any]:
        """Apply transformations to data"""
        transformed = data.copy()
        applied_transforms = []
        
        # Apply default transformations if none specified
        if transformations is None:
            transformations = self._infer_transformations(data)
        
        for transform_name in transformations:
            if transform_name in self.transformers:
                try:
                    transformed = self.transformers[transform_name](transformed)
                    applied_transforms.append(transform_name)
                except Exception as e:
                    logger.warning(f"Transform {transform_name} failed: {e}")
        
        # Apply enrichment
        for enricher in self.enrichers:
            try:
                transformed = enricher(transformed)
            except Exception as e:
                logger.warning(f"Enrichment failed: {e}")
        
        transformed['_transformations_applied'] = applied_transforms
        return transformed
    
    def _infer_transformations(self, data: Dict[str, Any]) -> List[str]:
        """Infer which transformations to apply based on data structure"""
        transforms = []
        
        if 'ui_elements' in data or 'ui_state' in data:
            transforms.append('normalize_ui_state')
            transforms.append('sanitize_text')
        
        if 'action_plan' in data or 'steps' in data:
            transforms.append('normalize_action_plan')
        
        if 'instruction' in data or 'task' in data:
            transforms.append('normalize_task')
        
        transforms.append('enrich_with_metadata')
        return transforms
    
    def _normalize_ui_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize UI state to standard format"""
        ui_state = data.get('ui_state') or data
        
        normalized = {
            'elements': [],
            'text_content': '',
            'metadata': {}
        }
        
        # Normalize elements
        elements = ui_state.get('ui_elements', [])
        for elem in elements:
            normalized_elem = {
                'id': elem.get('id', ''),
                'type': elem.get('type', 'unknown'),
                'text': elem.get('text', ''),
                'bbox': elem.get('bbox', []),
                'confidence': elem.get('confidence', 1.0),
                'is_interactable': elem.get('type') in ['button', 'input', 'link'],
                'normalized_at': datetime.utcnow().isoformat()
            }
            normalized['elements'].append(normalized_elem)
        
        # Normalize text content
        normalized['text_content'] = ui_state.get('full_text_ocr', '')
        normalized['metadata'] = {
            'screenshot_path': ui_state.get('screenshot_path', ''),
            'timestamp': ui_state.get('timestamp', ''),
            'element_count': len(normalized['elements'])
        }
        
        data['normalized_ui_state'] = normalized
        return data
    
    def _normalize_action_plan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize action plan to standard format"""
        plan = data.get('action_plan') or data.get('plan') or {}
        steps = plan.get('steps', []) or data.get('steps', [])
        
        normalized_steps = []
        for i, step in enumerate(steps):
            normalized_step = {
                'step_id': step.get('id', f"step_{i+1}"),
                'step_number': step.get('step_number', i + 1),
                'action': step.get('action', 'unknown'),
                'target_id': step.get('target_id', ''),
                'value': step.get('value', ''),
                'description': step.get('description', ''),
                'confidence': float(step.get('confidence', 0.8)),
                'estimated_duration_ms': step.get('estimated_duration_ms', 1000),
                'dependencies': step.get('dependencies', []),
                'normalized_at': datetime.utcnow().isoformat()
            }
            normalized_steps.append(normalized_step)
        
        normalized_plan = {
            'plan_id': plan.get('plan_id', data.get('plan_id', '')),
            'instruction': plan.get('instruction', data.get('instruction', '')),
            'steps': normalized_steps,
            'total_steps': len(normalized_steps),
            'estimated_duration_seconds': plan.get('estimated_duration_seconds', sum(
                s.get('estimated_duration_ms', 1000) for s in normalized_steps
            ) / 1000),
            'overall_confidence': plan.get('overall_confidence', 
                sum(s['confidence'] for s in normalized_steps) / len(normalized_steps) if normalized_steps else 0.8),
            'normalized_at': datetime.utcnow().isoformat()
        }
        
        data['normalized_action_plan'] = normalized_plan
        return data
    
    def _normalize_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize task data to standard format"""
        task = data.get('task') or {}
        instruction = task.get('instruction') or data.get('instruction', '')
        
        normalized_task = {
            'task_id': task.get('task_id', data.get('task_id', '')),
            'instruction': instruction,
            'instruction_normalized': instruction.lower().strip(),
            'device_id': task.get('device_id', data.get('device_id', '')),
            'parameters': task.get('parameters', data.get('parameters', {})),
            'priority': task.get('priority', 'medium'),
            'created_at': task.get('created_at', datetime.utcnow().isoformat()),
            'normalized_at': datetime.utcnow().isoformat(),
            'metadata': {
                'instruction_length': len(instruction),
                'word_count': len(instruction.split()),
                'has_location_context': 'location' in str(task.get('parameters', {})).lower()
            }
        }
        
        data['normalized_task'] = normalized_task
        return data
    
    def _sanitize_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive information from text"""
        from backend.guardrails.safety_guardrails import SafetyGuardrails
        
        guardrails = SafetyGuardrails({})
        
        # Sanitize UI state text
        if 'normalized_ui_state' in data:
            ui_state = data['normalized_ui_state']
            for elem in ui_state.get('elements', []):
                if 'text' in elem:
                    elem['text'] = guardrails.mask_sensitive_info(elem['text'])
            if 'text_content' in ui_state:
                ui_state['text_content'] = guardrails.mask_sensitive_info(ui_state['text_content'])
        
        # Sanitize task instruction if needed
        if 'normalized_task' in data:
            task = data['normalized_task']
            # Only sanitize if explicitly requested (usually we want to keep instructions)
            if self.config.get('sanitize_instructions', False):
                task['instruction'] = guardrails.mask_sensitive_info(task['instruction'])
        
        return data
    
    def _enrich_with_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with additional metadata"""
        if '_metadata' not in data:
            data['_metadata'] = {}
        
        metadata = data['_metadata']
        metadata.update({
            'enriched_at': datetime.utcnow().isoformat(),
            'data_types': list(data.keys()),
            'has_ui_state': 'normalized_ui_state' in data or 'ui_state' in data,
            'has_action_plan': 'normalized_action_plan' in data or 'action_plan' in data,
            'has_task': 'normalized_task' in data or 'task' in data
        })
        
        return data
    
    def validate(self, data: Dict[str, Any], validators: List[str] = None) -> tuple:
        """Validate data and return (is_valid, errors, warnings)"""
        errors = []
        warnings = []
        
        if validators is None:
            validators = self._infer_validators(data)
        
        for validator_name in validators:
            if validator_name in self.validators:
                try:
                    is_valid, val_errors, val_warnings = self.validators[validator_name](data)
                    if not is_valid:
                        errors.extend(val_errors)
                    warnings.extend(val_warnings)
                except Exception as e:
                    errors.append(f"Validator {validator_name} failed: {e}")
        
        return len(errors) == 0, errors, warnings
    
    def _validate_ui_state(self, data: Dict[str, Any]) -> tuple:
        """Validate UI state data"""
        errors = []
        warnings = []
        
        ui_state = data.get('normalized_ui_state') or data.get('ui_state')
        if not ui_state:
            return True, errors, warnings  # No UI state to validate
        
        if 'elements' in ui_state:
            if not isinstance(ui_state['elements'], list):
                errors.append("UI state elements must be a list")
            else:
                for i, elem in enumerate(ui_state['elements']):
                    if not isinstance(elem, dict):
                        errors.append(f"Element {i} must be a dictionary")
                    else:
                        if 'type' not in elem:
                            warnings.append(f"Element {i} missing 'type' field")
                        if 'bbox' in elem and len(elem['bbox']) != 4:
                            warnings.append(f"Element {i} has invalid bbox format")
        
        return len(errors) == 0, errors, warnings
    
    def _validate_action_plan(self, data: Dict[str, Any]) -> tuple:
        """Validate action plan data"""
        errors = []
        warnings = []
        
        plan = data.get('normalized_action_plan') or data.get('action_plan')
        if not plan:
            return True, errors, warnings
        
        if 'steps' in plan:
            if not isinstance(plan['steps'], list):
                errors.append("Action plan steps must be a list")
            else:
                if len(plan['steps']) == 0:
                    warnings.append("Action plan has no steps")
                
                for i, step in enumerate(plan['steps']):
                    if not isinstance(step, dict):
                        errors.append(f"Step {i} must be a dictionary")
                    else:
                        if 'action' not in step:
                            errors.append(f"Step {i} missing required 'action' field")
                        if 'confidence' in step:
                            conf = step['confidence']
                            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                                warnings.append(f"Step {i} has invalid confidence value: {conf}")
        
        return len(errors) == 0, errors, warnings
    
    def _validate_task(self, data: Dict[str, Any]) -> tuple:
        """Validate task data"""
        errors = []
        warnings = []
        
        task = data.get('normalized_task') or data.get('task')
        if not task:
            return True, errors, warnings
        
        if 'instruction' not in task:
            errors.append("Task missing required 'instruction' field")
        elif not task['instruction'] or not task['instruction'].strip():
            errors.append("Task instruction cannot be empty")
        
        if 'device_id' in task and not task['device_id']:
            warnings.append("Task has empty device_id")
        
        return len(errors) == 0, errors, warnings
    
    def register_transformer(self, name: str, transformer_func: Callable):
        """Register a custom transformer"""
        self.transformers[name] = transformer_func
    
    def register_validator(self, name: str, validator_func: Callable):
        """Register a custom validator"""
        self.validators[name] = validator_func
    
    def register_enricher(self, enricher_func: Callable):
        """Register a data enricher"""
        self.enrichers.append(enricher_func)


class DataLoader:
    """Loads transformed data with caching and versioning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cache_dir = Path(self.config.get('cache_dir', './.etl_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.versioning_enabled = self.config.get('versioning_enabled', True)
        self.max_cache_size_mb = self.config.get('max_cache_size_mb', 100)
    
    def load(self, data: Dict[str, Any], target: str = 'memory', **kwargs) -> Dict[str, Any]:
        """Load data to target destination"""
        loaders = {
            'memory': self._load_to_memory,
            'store': self._load_to_store,
            'cache': self._load_to_cache,
            'file': self._load_to_file,
        }
        
        loader = loaders.get(target)
        if not loader:
            raise ValueError(f"Unknown load target: {target}")
        
        return loader(data, **kwargs)
    
    def _load_to_memory(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Load data to memory (return as-is, ready for agent use)"""
        return {
            'target': 'memory',
            'data': data,
            'loaded_at': datetime.utcnow().isoformat(),
            'ready_for_agents': True
        }
    
    def _load_to_store(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Load data to persistent store"""
        from backend.agent_service.store import record_user_task
        
        username = kwargs.get('username')
        if username:
            record = {
                'data': data,
                'etl_processed': True,
                'loaded_at': datetime.utcnow().isoformat()
            }
            record_user_task(username, record)
        
        return {
            'target': 'store',
            'success': True,
            'loaded_at': datetime.utcnow().isoformat()
        }
    
    def _load_to_cache(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Load data to cache with versioning"""
        if not self.cache_enabled:
            return {'target': 'cache', 'cached': False, 'reason': 'cache disabled'}
        
        # Generate cache key
        cache_key = self._generate_cache_key(data)
        cache_path = self.cache_dir / f"{cache_key}.pkl.gz"
        
        # Check cache size
        if self._get_cache_size_mb() > self.max_cache_size_mb:
            self._cleanup_cache()
        
        # Store with metadata
        cache_data = {
            'data': data,
            'cached_at': datetime.utcnow().isoformat(),
            'version': self._get_data_version(data),
            'checksum': self._calculate_checksum(data)
        }
        
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            return {
                'target': 'cache',
                'cached': True,
                'cache_key': cache_key,
                'cache_path': str(cache_path)
            }
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            return {'target': 'cache', 'cached': False, 'error': str(e)}
    
    def _load_to_file(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Load data to file"""
        file_path = kwargs.get('file_path', self.cache_dir / f"data_{int(time.time())}.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return {
                'target': 'file',
                'success': True,
                'file_path': str(file_path)
            }
        except Exception as e:
            logger.error(f"Failed to save to file: {e}")
            return {'target': 'file', 'success': False, 'error': str(e)}
    
    def get_cached(self, cache_key: str = None, data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached data"""
        if not self.cache_enabled:
            return None
        
        if cache_key is None and data is not None:
            cache_key = self._generate_cache_key(data)
        
        if cache_key is None:
            return None
        
        cache_path = self.cache_dir / f"{cache_key}.pkl.gz"
        if not cache_path.exists():
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                return cached.get('data')
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data"""
        # Create a stable key from data content
        key_data = {
            'instruction': data.get('normalized_task', {}).get('instruction', ''),
            'device_id': data.get('normalized_task', {}).get('device_id', ''),
            'ui_elements_count': len(data.get('normalized_ui_state', {}).get('elements', []))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_data_version(self, data: Dict[str, Any]) -> str:
        """Get data version"""
        return data.get('_metadata', {}).get('schema_version', '1.0')
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB"""
        total_size = 0
        for file in self.cache_dir.glob('*.pkl.gz'):
            total_size += file.stat().st_size
        return total_size / (1024 * 1024)
    
    def _cleanup_cache(self):
        """Cleanup old cache files"""
        files = sorted(self.cache_dir.glob('*.pkl.gz'), key=lambda p: p.stat().st_mtime)
        # Remove oldest 20% of files
        remove_count = max(1, len(files) // 5)
        for file in files[:remove_count]:
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {file}: {e}")


class DataETLPipeline:
    """Main ETL pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.extractor = DataExtractor(config.get('extractor', {}))
        self.transformer = DataTransformer(config.get('transformer', {}))
        self.loader = DataLoader(config.get('loader', {}))
        self.logger = logging.getLogger(f"{__name__}.ETLPipeline")
    
    def process(self, 
                source_type: str,
                source_data: Any,
                transformations: List[str] = None,
                load_targets: List[str] = None,
                use_cache: bool = True,
                **kwargs) -> ETLResult:
        """Execute full ETL pipeline"""
        start_time = time.time()
        errors = []
        warnings = []
        cache_hit = False
        
        try:
            # Check cache first
            if use_cache and self.loader.cache_enabled:
                cached_data = self.loader.get_cached(data={'source_type': source_type})
                if cached_data:
                    cache_hit = True
                    self.logger.info("Cache hit, returning cached data")
                    return ETLResult(
                        success=True,
                        stage=ETLStage.LOAD,
                        data=cached_data,
                        metadata=self._create_metadata(source_type, cached_data, []),
                        errors=[],
                        warnings=['Data retrieved from cache'],
                        processing_time_ms=(time.time() - start_time) * 1000,
                        cache_hit=True
                    )
            
            # EXTRACT
            self.logger.info(f"Extracting data from {source_type}")
            extracted = self.extractor.extract(source_type, source_data, **kwargs)
            
            # TRANSFORM
            self.logger.info("Transforming data")
            transformed = self.transformer.transform(extracted, transformations)
            
            # Validate
            is_valid, val_errors, val_warnings = self.transformer.validate(transformed)
            errors.extend(val_errors)
            warnings.extend(val_warnings)
            
            if not is_valid and self.config.get('strict_validation', False):
                raise ValueError(f"Data validation failed: {val_errors}")
            
            # Assess data quality
            quality = self._assess_quality(transformed, errors, warnings)
            
            # LOAD
            load_targets = load_targets or ['memory']
            load_results = []
            for target in load_targets:
                self.logger.info(f"Loading data to {target}")
                load_result = self.loader.load(transformed, target=target, **kwargs)
                load_results.append(load_result)
            
            # Create metadata
            metadata = self._create_metadata(
                source_type,
                transformed,
                self.transformer._infer_transformations(transformed),
                quality=quality,
                validation_passed=is_valid,
                error_count=len(errors),
                warnings=warnings
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ETLResult(
                success=True,
                stage=ETLStage.LOAD,
                data=transformed,
                metadata=metadata,
                errors=errors,
                warnings=warnings,
                processing_time_ms=processing_time,
                cache_hit=cache_hit
            )
            
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {e}", exc_info=True)
            errors.append(str(e))
            return ETLResult(
                success=False,
                stage=ETLStage.EXTRACT,  # Failed at earliest stage
                data=None,
                metadata=self._create_metadata(source_type, {}, []),
                errors=errors,
                warnings=warnings,
                processing_time_ms=(time.time() - start_time) * 1000,
                cache_hit=False
            )
    
    def _create_metadata(self, 
                        source: str,
                        data: Dict[str, Any],
                        transformations: List[str],
                        quality: DataQuality = DataQuality.GOOD,
                        validation_passed: bool = True,
                        error_count: int = 0,
                        warnings: List[str] = None) -> DataMetadata:
        """Create metadata for processed data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()
        
        return DataMetadata(
            source=source,
            timestamp=datetime.utcnow(),
            version='1.0',
            quality=quality,
            schema_version='1.0',
            checksum=checksum,
            transformations=transformations,
            enrichment_applied='enrich_with_metadata' in transformations,
            validation_passed=validation_passed,
            error_count=error_count,
            warnings=warnings or []
        )
    
    def _assess_quality(self, data: Dict[str, Any], errors: List[str], warnings: List[str]) -> DataQuality:
        """Assess data quality"""
        if errors:
            return DataQuality.INVALID
        if len(warnings) > 5:
            return DataQuality.POOR
        if len(warnings) > 2:
            return DataQuality.ACCEPTABLE
        if len(warnings) > 0:
            return DataQuality.GOOD
        return DataQuality.EXCELLENT

