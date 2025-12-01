"""Mobile-optimized action executor with platform-specific adaptations"""
from typing import Dict, Any, List
import logging
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class MobilePlatform(Enum):
    ANDROID = "android"
    IOS = "ios"

class MobileOptimizedExecutor:
    """Mobile-optimized action executor with platform-specific adaptations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_platform = None
        self.gesture_registry = self._initialize_gesture_registry()
    
    def _initialize_gesture_registry(self) -> Dict[str, Any]:
        """Initialize mobile-specific gesture library"""
        return {
            'swipe_up': {'type': 'swipe', 'direction': 'up', 'duration': 300},
            'swipe_down': {'type': 'swipe', 'direction': 'down', 'duration': 300},
            'swipe_left': {'type': 'swipe', 'direction': 'left', 'duration': 300},
            'swipe_right': {'type': 'swipe', 'direction': 'right', 'duration': 300},
            'long_press': {'type': 'press', 'duration': 2000},
            'double_tap': {'type': 'tap', 'count': 2},
            'pinch_zoom': {'type': 'pinch', 'scale': 1.5}
        }
    
    async def execute_mobile_optimized_action(self, action: Dict[str, Any], 
                                            platform: MobilePlatform,
                                            driver: Any = None) -> Dict[str, Any]:
        """Execute action with platform-specific optimizations"""
        
        self.current_platform = platform
        action_type = action.get('type', '')
        
        # Platform-specific adaptations
        adapted_action = self._adapt_action_for_platform(action, platform)
        
        try:
            if action_type in ['tap', 'click']:
                result = await self._execute_tap(adapted_action, driver)
            elif action_type == 'swipe':
                result = await self._execute_swipe(adapted_action, driver)
            elif action_type == 'type':
                result = await self._execute_type(adapted_action, driver)
            elif action_type == 'wait':
                result = await self._execute_wait(adapted_action)
            else:
                result = await self._execute_custom_gesture(adapted_action, driver)
            
            # Add platform context to result
            result['platform'] = platform.value
            result['optimized'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Mobile action failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': platform.value
            }
    
    def _adapt_action_for_platform(self, action: Dict[str, Any], 
                                 platform: MobilePlatform) -> Dict[str, Any]:
        """Adapt action parameters for specific mobile platform"""
        
        adapted_action = action.copy()
        
        if platform == MobilePlatform.IOS:
            # iOS-specific adaptations
            if action.get('type') == 'swipe':
                adapted_action['duration'] = action.get('duration', 500) * 1.2
            elif action.get('type') == 'tap':
                adapted_action['duration'] = action.get('duration', 100) * 0.8
                
        elif platform == MobilePlatform.ANDROID:
            # Android-specific adaptations
            if action.get('type') == 'swipe':
                adapted_action['duration'] = action.get('duration', 500) * 0.9
        
        return adapted_action
    
    async def _execute_tap(self, action: Dict[str, Any], driver: Any) -> Dict[str, Any]:
        """Execute tap action"""
        logger.info(f"Executing tap action: {action}")
        # Placeholder for actual tap implementation
        await asyncio.sleep(0.1)
        return {'success': True, 'action': 'tap', 'duration': 0.1}
    
    async def _execute_swipe(self, action: Dict[str, Any], driver: Any) -> Dict[str, Any]:
        """Execute swipe action"""
        logger.info(f"Executing swipe action: {action}")
        duration = action.get('duration', 300) / 1000.0
        await asyncio.sleep(duration)
        return {'success': True, 'action': 'swipe', 'duration': duration}
    
    async def _execute_type(self, action: Dict[str, Any], driver: Any) -> Dict[str, Any]:
        """Execute type action"""
        text = action.get('text', '')
        logger.info(f"Executing type action: {len(text)} characters")
        await asyncio.sleep(len(text) * 0.05)
        return {'success': True, 'action': 'type', 'text': text}
    
    async def _execute_wait(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute wait action"""
        duration = action.get('duration', 1.0)
        await asyncio.sleep(duration)
        return {'success': True, 'action': 'wait', 'duration': duration}
    
    async def _execute_custom_gesture(self, action: Dict[str, Any], driver: Any) -> Dict[str, Any]:
        """Execute custom gesture"""
        gesture_name = action.get('gesture', '')
        gesture_def = self.gesture_registry.get(gesture_name, {})
        
        if not gesture_def:
            logger.warning(f"Unknown gesture: {gesture_name}")
            return {'success': False, 'error': 'Unknown gesture'}
        
        logger.info(f"Executing custom gesture: {gesture_name}")
        await asyncio.sleep(gesture_def.get('duration', 300) / 1000.0)
        return {'success': True, 'action': 'gesture', 'gesture': gesture_name}
    
    async def execute_cross_platform_workflow(self, workflow: List[Dict[str, Any]],
                                            platforms: List[MobilePlatform],
                                            driver: Any = None) -> Dict[str, Any]:
        """Execute same workflow across multiple platforms"""
        
        results = {}
        
        for platform in platforms:
            platform_results = []
            for step in workflow:
                result = await self.execute_mobile_optimized_action(step, platform, driver)
                platform_results.append(result)
            
            results[platform.value] = {
                'success': all(r.get('success', False) for r in platform_results),
                'steps': platform_results,
                'total_duration': sum(r.get('duration', 0) for r in platform_results)
            }
        
        return results
