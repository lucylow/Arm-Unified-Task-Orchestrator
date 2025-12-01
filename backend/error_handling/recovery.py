"""
Error Recovery Strategies and Dead Letter Queue

Implements sophisticated error recovery mechanisms and manages failed operations.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from threading import Lock

from .exceptions import (
    AutoRLBaseException,
    RecoveryFailedException,
    ErrorSeverity
)


logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    ESCALATE = "escalate"
    ROLLBACK = "rollback"
    COMPENSATION = "compensation"


@dataclass
class FailedOperation:
    """Represents a failed operation"""
    operation_id: str
    operation_name: str
    error: AutoRLBaseException
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    last_retry_time: Optional[float] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "error": self.error.to_dict(),
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "context": self.context,
            "recovery_strategy": self.recovery_strategy.value,
            "last_retry_time": self.last_retry_time,
            "resolved": self.resolved
        }


class DeadLetterQueue:
    """
    Dead Letter Queue for managing failed operations.
    
    Features:
    - Store failed operations
    - Retry management
    - Persistence to disk
    - Monitoring and alerts
    """
    
    def __init__(
        self,
        storage_dir: str = "./logs/dlq",
        max_memory_items: int = 1000,
        auto_retry: bool = True,
        retry_interval: float = 60.0
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_items = max_memory_items
        self.auto_retry = auto_retry
        self.retry_interval = retry_interval
        
        # In-memory queue
        self._queue = deque(maxlen=max_memory_items)
        self._operation_index: Dict[str, FailedOperation] = {}
        
        # Thread-safe access
        self._lock = Lock()
        
        # Retry task
        self._retry_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"DeadLetterQueue initialized. Storage: {self.storage_dir}, "
            f"Auto-retry: {auto_retry}"
        )
    
    def add(
        self,
        operation_name: str,
        error: AutoRLBaseException,
        context: Optional[Dict[str, Any]] = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        max_retries: int = 3
    ) -> str:
        """
        Add failed operation to queue.
        
        Args:
            operation_name: Name of the failed operation
            error: The exception that occurred
            context: Additional context information
            recovery_strategy: Strategy for recovery
            max_retries: Maximum retry attempts
            
        Returns:
            Operation ID
        """
        with self._lock:
            operation_id = f"{operation_name}_{int(time.time() * 1000)}"
            
            failed_op = FailedOperation(
                operation_id=operation_id,
                operation_name=operation_name,
                error=error,
                timestamp=time.time(),
                context=context or {},
                recovery_strategy=recovery_strategy,
                max_retries=max_retries
            )
            
            self._queue.append(failed_op)
            self._operation_index[operation_id] = failed_op
            
            # Persist to disk
            self._persist_operation(failed_op)
            
            logger.warning(
                f"Added failed operation to DLQ: {operation_name} "
                f"(ID: {operation_id}, Strategy: {recovery_strategy.value})"
            )
            
            return operation_id
    
    def _persist_operation(self, operation: FailedOperation):
        """Persist failed operation to disk"""
        try:
            date_str = time.strftime("%Y-%m-%d", time.localtime(operation.timestamp))
            storage_file = self.storage_dir / f"dlq_{date_str}.jsonl"
            
            with open(storage_file, 'a') as f:
                json.dump(operation.to_dict(), f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Failed to persist operation to disk: {e}")
    
    def get(self, operation_id: str) -> Optional[FailedOperation]:
        """Get failed operation by ID"""
        with self._lock:
            return self._operation_index.get(operation_id)
    
    def mark_resolved(self, operation_id: str):
        """Mark operation as resolved"""
        with self._lock:
            if operation_id in self._operation_index:
                operation = self._operation_index[operation_id]
                operation.resolved = True
                logger.info(f"Marked operation {operation_id} as resolved")
    
    def get_pending_operations(
        self,
        strategy: Optional[RecoveryStrategy] = None
    ) -> List[FailedOperation]:
        """Get pending (unresolved) operations"""
        with self._lock:
            operations = [
                op for op in self._queue
                if not op.resolved
            ]
            
            if strategy:
                operations = [
                    op for op in operations
                    if op.recovery_strategy == strategy
                ]
            
            return operations
    
    def get_retryable_operations(self) -> List[FailedOperation]:
        """Get operations that can be retried"""
        current_time = time.time()
        
        with self._lock:
            retryable = []
            
            for op in self._queue:
                if op.resolved:
                    continue
                
                if op.retry_count >= op.max_retries:
                    continue
                
                if op.recovery_strategy != RecoveryStrategy.RETRY:
                    continue
                
                # Check retry interval
                if op.last_retry_time:
                    time_since_retry = current_time - op.last_retry_time
                    if time_since_retry < self.retry_interval:
                        continue
                
                retryable.append(op)
            
            return retryable
    
    async def retry_operation(
        self,
        operation: FailedOperation,
        retry_func: Callable
    ) -> bool:
        """
        Retry a failed operation.
        
        Args:
            operation: The failed operation to retry
            retry_func: Function to call for retry
            
        Returns:
            True if retry succeeded, False otherwise
        """
        operation.retry_count += 1
        operation.last_retry_time = time.time()
        
        logger.info(
            f"Retrying operation {operation.operation_id} "
            f"(attempt {operation.retry_count}/{operation.max_retries})"
        )
        
        try:
            # Call retry function
            if asyncio.iscoroutinefunction(retry_func):
                await retry_func(operation)
            else:
                retry_func(operation)
            
            # Mark as resolved on success
            self.mark_resolved(operation.operation_id)
            logger.info(f"Retry succeeded for operation {operation.operation_id}")
            return True
            
        except Exception as e:
            logger.warning(
                f"Retry attempt {operation.retry_count} failed for "
                f"operation {operation.operation_id}: {e}"
            )
            
            if operation.retry_count >= operation.max_retries:
                logger.error(
                    f"Operation {operation.operation_id} exhausted all retries. "
                    f"Moving to permanent failure."
                )
                # Could trigger escalation here
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics"""
        with self._lock:
            total_operations = len(self._queue)
            resolved_operations = sum(1 for op in self._queue if op.resolved)
            pending_operations = total_operations - resolved_operations
            
            by_strategy = {}
            for strategy in RecoveryStrategy:
                count = sum(
                    1 for op in self._queue
                    if op.recovery_strategy == strategy and not op.resolved
                )
                by_strategy[strategy.value] = count
            
            by_severity = {}
            for op in self._queue:
                if not op.resolved:
                    severity = op.error.severity.value
                    by_severity[severity] = by_severity.get(severity, 0) + 1
            
            return {
                "total_operations": total_operations,
                "resolved_operations": resolved_operations,
                "pending_operations": pending_operations,
                "by_strategy": by_strategy,
                "by_severity": by_severity,
                "timestamp": time.time()
            }
    
    def clear_resolved(self):
        """Clear resolved operations from memory"""
        with self._lock:
            # Keep only unresolved operations
            unresolved = [op for op in self._queue if not op.resolved]
            self._queue.clear()
            self._queue.extend(unresolved)
            
            # Update index
            self._operation_index = {
                op.operation_id: op for op in self._queue
            }
            
            logger.info(f"Cleared resolved operations. {len(unresolved)} remain.")
    
    async def start_auto_retry(self, retry_func: Callable):
        """Start automatic retry task"""
        if not self.auto_retry:
            logger.info("Auto-retry is disabled")
            return
        
        async def retry_loop():
            while True:
                try:
                    await asyncio.sleep(self.retry_interval)
                    
                    retryable = self.get_retryable_operations()
                    if not retryable:
                        continue
                    
                    logger.info(f"Auto-retry: Processing {len(retryable)} operations")
                    
                    for operation in retryable:
                        try:
                            await self.retry_operation(operation, retry_func)
                        except Exception as e:
                            logger.error(f"Auto-retry failed for {operation.operation_id}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error in auto-retry loop: {e}")
        
        self._retry_task = asyncio.create_task(retry_loop())
        logger.info("Started auto-retry task")
    
    def stop_auto_retry(self):
        """Stop automatic retry task"""
        if self._retry_task:
            self._retry_task.cancel()
            logger.info("Stopped auto-retry task")


class RecoveryManager:
    """
    Manages error recovery strategies and coordinates recovery actions.
    """
    
    def __init__(self, dlq: Optional[DeadLetterQueue] = None):
        self.dlq = dlq or DeadLetterQueue()
        self._recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
        
        logger.info("RecoveryManager initialized")
    
    def _register_default_handlers(self):
        """Register default recovery handlers"""
        self.register_handler(RecoveryStrategy.IGNORE, self._handle_ignore)
        self.register_handler(RecoveryStrategy.ESCALATE, self._handle_escalate)
    
    def register_handler(self, strategy: RecoveryStrategy, handler: Callable):
        """Register recovery handler for a strategy"""
        self._recovery_handlers[strategy] = handler
        logger.info(f"Registered recovery handler for strategy: {strategy.value}")
    
    async def _handle_ignore(self, operation: FailedOperation) -> bool:
        """Ignore the error and mark as resolved"""
        logger.info(f"Ignoring error for operation {operation.operation_id}")
        self.dlq.mark_resolved(operation.operation_id)
        return True
    
    async def _handle_escalate(self, operation: FailedOperation) -> bool:
        """Escalate to human intervention"""
        logger.critical(
            f"ESCALATION REQUIRED: Operation {operation.operation_id} "
            f"({operation.operation_name}) requires manual intervention"
        )
        
        # Could send notifications, create tickets, etc.
        return False
    
    async def recover(
        self,
        operation_name: str,
        error: AutoRLBaseException,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[RecoveryStrategy] = None
    ) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            operation_name: Name of the failed operation
            error: The exception that occurred
            context: Additional context
            strategy: Recovery strategy (auto-determined if not provided)
            
        Returns:
            True if recovery succeeded
        """
        # Determine recovery strategy if not provided
        if strategy is None:
            strategy = self._determine_strategy(error)
        
        logger.info(
            f"Attempting recovery for {operation_name} "
            f"using strategy: {strategy.value}"
        )
        
        # Add to DLQ
        operation_id = self.dlq.add(
            operation_name=operation_name,
            error=error,
            context=context,
            recovery_strategy=strategy
        )
        
        # Get recovery handler
        handler = self._recovery_handlers.get(strategy)
        if not handler:
            logger.warning(f"No handler registered for strategy: {strategy.value}")
            return False
        
        # Execute recovery
        try:
            operation = self.dlq.get(operation_id)
            if asyncio.iscoroutinefunction(handler):
                success = await handler(operation)
            else:
                success = handler(operation)
            
            return success
            
        except Exception as e:
            logger.error(f"Recovery handler failed: {e}")
            return False
    
    def _determine_strategy(self, error: AutoRLBaseException) -> RecoveryStrategy:
        """Determine appropriate recovery strategy based on error"""
        
        # Critical errors should escalate
        if error.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE
        
        # Recoverable errors should retry
        if error.recoverable:
            return RecoveryStrategy.RETRY
        
        # Non-recoverable errors should escalate
        return RecoveryStrategy.ESCALATE


# Global instances
_global_dlq: Optional[DeadLetterQueue] = None
_global_recovery_manager: Optional[RecoveryManager] = None


def get_dead_letter_queue() -> DeadLetterQueue:
    """Get global dead letter queue instance"""
    global _global_dlq
    if _global_dlq is None:
        _global_dlq = DeadLetterQueue()
    return _global_dlq


def get_recovery_manager() -> RecoveryManager:
    """Get global recovery manager instance"""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = RecoveryManager(get_dead_letter_queue())
    return _global_recovery_manager

