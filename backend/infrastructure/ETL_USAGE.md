# Data ETL Pipeline Usage Guide

## Overview

The Data ETL (Extract, Transform, Load) pipeline provides comprehensive data processing for AI agents with:
- **Extract**: Multi-source data extraction
- **Transform**: Validation, normalization, enrichment, sanitization
- **Load**: Efficient loading with caching and versioning
- **Quality Checks**: Comprehensive data quality assessment

## Quick Start

### Basic Usage

```python
from backend.infrastructure.etl_utils import process_ui_state, process_action_plan, assess_data_quality

# Process UI state
ui_state = {
    'ui_elements': [...],
    'full_text_ocr': '...',
    'screenshot_path': '...'
}
processed = process_ui_state(ui_state)

# Access normalized data
normalized_ui = processed.get('normalized_ui_state', {})

# Assess quality
quality = assess_data_quality(processed, data_type='ui_state')
print(f"Quality: {quality['label']} ({quality['score']:.2f})")
```

### Using with Agent Orchestrator

```python
from backend.orchestration.etl_enhanced_orchestrator import ETLEnhancedAgentOrchestrator

# Initialize with ETL
orchestrator = ETLEnhancedAgentOrchestrator(
    visual_perception=visual_perception,
    llm_planner=llm_planner,
    executor=executor,
    etl_config={
        'enabled': True,
        'cache_enabled': True,
        'strict_validation': False
    }
)

# Execute workflow (ETL is automatically applied)
context = await orchestrator.execute_workflow(
    driver=driver,
    instruction="Login to the app",
    use_etl=True
)

# Access normalized data
normalized_ui = context.get('normalized_ui_state', {})
normalized_plan = context.get('normalized_action_plan', {})
```

## Advanced Usage

### Custom ETL Pipeline

```python
from backend.infrastructure.data_etl_pipeline import DataETLPipeline

# Create custom pipeline
pipeline = DataETLPipeline(config={
    'extractor': {
        'custom_extractor_enabled': True
    },
    'transformer': {
        'sanitize_instructions': False
    },
    'loader': {
        'cache_enabled': True,
        'cache_dir': './custom_cache',
        'max_cache_size_mb': 200
    },
    'strict_validation': True
})

# Process data
result = pipeline.process(
    source_type='ui_state',
    source_data=ui_state,
    transformations=['normalize_ui_state', 'sanitize_text'],
    load_targets=['memory', 'cache', 'store'],
    use_cache=True
)

if result.success:
    print(f"Quality: {result.metadata.quality.value}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    data = result.data
else:
    print(f"Errors: {result.errors}")
```

### Custom Extractors

```python
from backend.infrastructure.data_etl_pipeline import DataExtractor

extractor = DataExtractor()

def my_custom_extractor(source_data, **kwargs):
    # Custom extraction logic
    return {
        'custom_field': source_data.get('field'),
        'processed_at': datetime.utcnow().isoformat()
    }

extractor.register_extractor('custom_source', my_custom_extractor)

# Use custom extractor
result = extractor.extract('custom_source', {'field': 'value'})
```

### Custom Transformers

```python
from backend.infrastructure.data_etl_pipeline import DataTransformer

transformer = DataTransformer()

def my_custom_transformer(data):
    # Custom transformation logic
    data['custom_field'] = data.get('field', '').upper()
    return data

transformer.register_transformer('uppercase_field', my_custom_transformer)

# Use custom transformer
transformed = transformer.transform(data, transformations=['uppercase_field'])
```

### Custom Validators

```python
from backend.infrastructure.data_etl_pipeline import DataTransformer

transformer = DataTransformer()

def my_custom_validator(data):
    errors = []
    warnings = []
    
    if 'required_field' not in data:
        errors.append("Missing required_field")
    
    return len(errors) == 0, errors, warnings

transformer.register_validator('validate_custom', my_custom_validator)

# Use custom validator
is_valid, errors, warnings = transformer.validate(data, validators=['validate_custom'])
```

## Data Schemas

The ETL pipeline uses standardized schemas for data structures:

```python
from backend.infrastructure.data_schemas import (
    UIStateSchema,
    ActionPlanSchema,
    TaskSchema,
    AgentDataSchema
)

# Create from dictionary
ui_schema = UIStateSchema.from_dict(ui_state_dict)
plan_schema = ActionPlanSchema.from_dict(plan_dict)

# Convert to dictionary
ui_dict = ui_schema.to_dict()
plan_dict = plan_schema.to_dict()
```

## Quality Assessment

```python
from backend.infrastructure.data_quality import DataQualityChecker

checker = DataQualityChecker(config={
    'low_confidence_threshold': 0.6,
    'max_data_age_seconds': 300
})

score, issues = checker.assess_quality(data, data_type='ui_state')

for issue in issues:
    if not issue.passed:
        print(f"Issue: {issue.message}")
        print(f"Severity: {issue.severity}")
        print(f"Fix: {issue.suggested_fix}")
```

## Caching

The ETL pipeline includes automatic caching:

```python
from backend.infrastructure.data_etl_pipeline import DataLoader

loader = DataLoader(config={
    'cache_enabled': True,
    'cache_dir': './.etl_cache',
    'max_cache_size_mb': 100
})

# Load to cache
loader.load(data, target='cache')

# Retrieve from cache
cached_data = loader.get_cached(cache_key='abc123')
# Or
cached_data = loader.get_cached(data=similar_data)  # Auto-generates key
```

## Integration Points

### With Perception Agent

```python
# In perception agent
from backend.infrastructure.etl_utils import process_ui_state

ui_state = visual_perception.capture_and_analyze(driver)
processed = process_ui_state(ui_state)
normalized_ui = processed.get('normalized_ui_state', {})
```

### With Planning Agent

```python
# In planning agent
from backend.infrastructure.etl_utils import process_action_plan

action_plan = llm_planner.generate_action_plan(instruction, ui_state)
processed = process_action_plan(action_plan, instruction=instruction)
normalized_plan = processed.get('normalized_action_plan', {})
```

### With Execution Agent

```python
# Execution agent can use normalized data
normalized_plan = context.get('normalized_action_plan', {})
if normalized_plan:
    steps = normalized_plan.get('steps', [])
    # Use normalized steps with confidence scores, dependencies, etc.
```

## Configuration Options

```python
etl_config = {
    # Pipeline settings
    'enabled': True,
    'strict_validation': False,
    
    # Extractor settings
    'extractor': {
        'custom_extractors': {}
    },
    
    # Transformer settings
    'transformer': {
        'sanitize_instructions': False,
        'custom_transformers': {},
        'custom_validators': {}
    },
    
    # Loader settings
    'loader': {
        'cache_enabled': True,
        'cache_dir': './.etl_cache',
        'max_cache_size_mb': 100,
        'versioning_enabled': True
    }
}
```

## Benefits

1. **Data Quality**: Automatic validation and quality checks
2. **Consistency**: Standardized data formats across agents
3. **Performance**: Caching reduces redundant processing
4. **Reliability**: Error handling and fallback mechanisms
5. **Extensibility**: Easy to add custom extractors, transformers, validators
6. **Observability**: Metadata tracking and quality metrics

## Best Practices

1. Always check `result.success` before using processed data
2. Use caching for frequently accessed data
3. Assess data quality before critical operations
4. Register custom validators for domain-specific checks
5. Monitor ETL metadata for quality trends
6. Use schemas for type safety and validation

