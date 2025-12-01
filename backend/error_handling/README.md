# AutoRL Error Handling System

> Enterprise-grade error handling for the AutoRL mobile automation platform.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Components](#components)
- [Documentation](#documentation)
- [Examples](#examples)
- [Testing](#testing)

## Overview

The AutoRL Error Handling System provides a comprehensive, production-ready solution for managing errors, failures, and recovery in the mobile automation platform. It includes custom exceptions, validation, retry mechanisms, circuit breakers, error tracking, recovery strategies, and comprehensive logging.

## Features

### ✅ Custom Exception Hierarchy
- **40+ specialized exceptions** organized by category
- **Rich error context** with device_id, task_id, user_id, etc.
- **Recovery suggestions** for each error type
- **Severity levels** (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### ✅ Error Tracking & Monitoring
- **Centralized error tracking** with metrics
- **Real-time error rates** and trends
- **Alert generation** on error rate spikes
- **Export capabilities** for analysis

### ✅ Circuit Breaker Pattern
- **Prevent cascading failures** in distributed systems
- **Automatic recovery** after cooldown period
- **Per-service monitoring** and statistics

### ✅ Input Validation
- **Comprehensive validators** for common data types
- **Custom validators** support
- **Detailed error messages** for debugging
- **Decorator-based validation**

### ✅ Error Recovery
- **Dead Letter Queue** for failed operations
- **Multiple recovery strategies** (RETRY, FALLBACK, ESCALATE, etc.)
- **Automatic retry** management
- **Custom recovery handlers**

### ✅ Retry Mechanisms
- **Exponential backoff**
- **Configurable attempts**
- **Exception filtering**
- **Custom retry callbacks**

### ✅ Comprehensive Logging
- **Structured logging** (JSON/text formats)
- **Multiple handlers** (console, file, rotating, daily)
- **Error-specific formatting**
- **Context enrichment**

## Quick Start

### Installation

The error handling system is part of the AutoRL platform. No additional installation required.

### Basic Usage

```python
from src.error_handling import (
    DeviceConnectionException,
    with_error_handling,
    with_retry,
    validate_string
)

# 1. Use custom exceptions
@with_retry(max_attempts=3, delay=2.0)
@with_error_handling()
async def connect_device(device_id: str):
    # Validate input
    device_id = validate_string(device_id, "device_id", not_empty=True)
    
    try:
        # Connection logic
        driver = await connect_to_appium(device_id)
        return driver
    except Exception as e:
        raise DeviceConnectionException(
            f"Failed to connect to {device_id}",
            device_id=device_id,
            original_exception=e
        )

# 2. Use circuit breaker for external services
from src.error_handling import circuit_breaker

@circuit_breaker("llm_service")
async def call_llm_api(prompt: str):
    response = await llm_client.generate(prompt)
    return response

# 3. Setup logging at startup
from src.error_handling import setup_error_logging
import logging

setup_error_logging(
    log_dir="./logs",
    console_level=logging.INFO,
    json_format=True
)
```

## Components

### 1. Exceptions (`exceptions.py`)
Custom exception hierarchy with rich context and recovery suggestions.

**Categories:**
- Infrastructure (device, network, resources)
- Validation (input, configuration)
- Business Logic (tasks, planning, perception)
- External Services (LLM, database, APIs)
- Timeouts
- Security
- Data
- UI Interactions

### 2. Error Tracker (`tracker.py`)
Centralized error tracking with metrics and reporting.

**Features:**
- In-memory and persistent storage
- Error rate calculation
- Alert generation
- Trend analysis
- Export capabilities

### 3. Circuit Breaker (`circuit_breaker.py`)
Prevent cascading failures with circuit breaker pattern.

**States:**
- CLOSED: Normal operation
- OPEN: Blocking requests (service failing)
- HALF_OPEN: Testing recovery

### 4. Validators (`validators.py`)
Input validation with detailed error messages.

**Validators:**
- String, Number, List, Dict
- Path, Email, URL
- Custom validators

### 5. Decorators (`decorators.py`)
Simplify error handling with decorators.

**Available:**
- `@with_error_handling()` - Automatic error handling
- `@with_retry()` - Retry with exponential backoff
- `@with_timeout()` - Timeout protection
- `@log_execution()` - Execution logging
- `@circuit_breaker()` - Circuit breaker protection
- `@validate_args()` - Argument validation

### 6. Recovery System (`recovery.py`)
Sophisticated error recovery with Dead Letter Queue.

**Strategies:**
- RETRY: Retry with backoff
- FALLBACK: Use alternative
- IGNORE: Skip and continue
- ESCALATE: Human intervention
- ROLLBACK: Undo changes
- COMPENSATION: Compensate for failure

### 7. Logging (`logging.py`)
Comprehensive structured logging.

**Features:**
- Multiple handlers
- JSON/text formats
- Log rotation
- Error filtering
- Context enrichment

## Documentation

### 📚 Complete Guides

- **[Comprehensive Guide](../../ERROR_HANDLING_GUIDE.md)** - Full documentation with examples
- **[Quick Reference](../../QUICK_ERROR_HANDLING_REFERENCE.md)** - Common use cases
- **[Summary](../../ERROR_HANDLING_SUMMARY.md)** - Overview and benefits

### 💻 Code Examples

- **[Integration Example](./example_integration.py)** - Complete integration example
- **[Test Suite](../../tests/test_error_handling_basic.py)** - Usage examples in tests

## Examples

### Example 1: Robust Function

```python
from src.error_handling import (
    with_error_handling,
    with_retry,
    with_timeout,
    log_execution,
    validate_args,
    validate_string,
    validate_number
)

@log_execution(include_duration=True)
@with_timeout(60)
@with_retry(max_attempts=3, delay=2.0, backoff=2.0)
@validate_args(
    device_id=lambda x: validate_string(x, "device_id", not_empty=True),
    timeout=lambda x: validate_number(x, "timeout", positive=True)
)
@with_error_handling()
async def execute_on_device(device_id: str, timeout: int):
    """Robust function with comprehensive error handling"""
    # Implementation
    ...
```

### Example 2: Error Recovery

```python
from src.error_handling import (
    get_recovery_manager,
    RecoveryStrategy,
    TaskExecutionException
)

recovery_manager = get_recovery_manager()

try:
    await execute_task()
except Exception as e:
    error = TaskExecutionException(
        "Task execution failed",
        task_id="task123",
        original_exception=e
    )
    
    # Attempt recovery
    recovered = await recovery_manager.recover(
        operation_name="task_execution",
        error=error,
        context={"task_id": "task123"},
        strategy=RecoveryStrategy.RETRY
    )
    
    if not recovered:
        raise error
```

### Example 3: Monitoring

```python
from src.error_handling import (
    get_error_tracker,
    get_circuit_breaker_manager,
    get_dead_letter_queue
)

# Error statistics
tracker = get_error_tracker()
summary = tracker.get_error_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"By category: {summary['by_category']}")

# Circuit breaker status
cb_manager = get_circuit_breaker_manager()
stats = cb_manager.get_all_stats()
for service, data in stats.items():
    print(f"{service}: {data['state']} - {data['success_rate']:.1f}%")

# Dead letter queue
dlq = get_dead_letter_queue()
dlq_stats = dlq.get_stats()
print(f"Pending operations: {dlq_stats['pending_operations']}")
```

## Testing

### Run Tests

```bash
# Run all error handling tests
pytest tests/test_error_handling_basic.py -v

# Run specific test
pytest tests/test_error_handling_basic.py::TestExceptions -v

# Run with coverage
pytest tests/test_error_handling_basic.py --cov=src/error_handling --cov-report=html
```

### Test Coverage

The test suite covers:
- ✅ Exception creation and serialization
- ✅ Validators (string, number, list, dict, path)
- ✅ Decorators (error handling, retry, timeout)
- ✅ Circuit breaker (states, transitions)
- ✅ Error tracking (tracking, rates, export)
- ✅ Dead letter queue (add, resolve, retry)

## Best Practices

### 1. Always Use Specific Exceptions

```python
# ❌ Bad
raise Exception("Connection failed")

# ✅ Good
raise DeviceConnectionException(
    "Failed to connect to device",
    device_id=device_id
)
```

### 2. Add Rich Context

```python
from src.error_handling import ErrorContext

context = ErrorContext(
    device_id="emulator-5554",
    task_id="task123",
    user_id="user456",
    additional_data={"attempt": 3}
)

raise TaskExecutionException(
    "Task failed",
    task_id="task123",
    context=context
)
```

### 3. Validate Inputs Early

```python
@validate_args(
    device_id=lambda x: validate_string(x, "device_id", not_empty=True),
    timeout=lambda x: validate_number(x, "timeout", positive=True)
)
async def my_function(device_id: str, timeout: int):
    # Inputs are validated before function runs
    ...
```

### 4. Use Circuit Breakers for External Services

```python
@circuit_breaker("external_api", config=CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    timeout=60.0
))
async def call_external_api():
    ...
```

### 5. Monitor and Alert

```python
# Setup monitoring at startup
async def monitor_errors():
    while True:
        await asyncio.sleep(60)
        
        tracker = get_error_tracker()
        rates = tracker.get_error_rate(time_window_seconds=300)
        
        for error_code, rate in rates.items():
            if rate > 10:  # More than 10 errors per minute
                logger.critical(f"High error rate for {error_code}: {rate}/min")
```

## Contributing

When adding new error types:

1. Add exception to `exceptions.py`
2. Include in `__all__` in `__init__.py`
3. Add tests in `tests/test_error_handling_basic.py`
4. Update documentation

## Performance

The error handling system is designed to be:
- **Low overhead** - Minimal performance impact
- **Async-first** - Fully async/await compatible
- **Thread-safe** - Safe for concurrent use
- **Memory efficient** - Configurable memory limits

## FAQ

**Q: When should I use circuit breakers?**  
A: Use circuit breakers for any external service calls (APIs, databases, LLM services) to prevent cascading failures.

**Q: How do I customize retry behavior?**  
A: Use the `@with_retry()` decorator with custom parameters or implement custom retry logic.

**Q: Can I add custom validators?**  
A: Yes! Extend the `Validator` base class. See `validators.py` for examples.

**Q: How are errors persisted?**  
A: Errors are logged to daily JSONL files in the configured log directory and also in rotating log files.

**Q: Can I export error data?**  
A: Yes! Use `tracker.export_errors()` to export to JSON format.

## Support

- **Documentation**: See guides in repository root
- **Examples**: Check `example_integration.py`
- **Tests**: See `tests/test_error_handling_basic.py`
- **Issues**: Report via GitHub issues

## License

Part of the AutoRL project. See main project license.

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Maintainer**: AutoRL Team

