"""Action executor wrappers for Appium/Selenium interactions with retries.

This module provides robust wait and retry helpers. It uses safe patterns so
it can be executed without Appium installed (useful for offline tests).
"""
import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ActionExecutor:
    def __init__(self, driver: Any = None, default_timeout: float = 15.0, retry_delay: float = 0.8):
        self.driver = driver
        self.default_timeout = default_timeout
        self.retry_delay = retry_delay

    def _find(self, locator_type, locator):
        if not self.driver:
            raise RuntimeError("No driver available")
        return self.driver.find_element(locator_type, locator)

    async def wait_for_element(self, locator_type, locator, timeout: Optional[float] = None):
        timeout = timeout or self.default_timeout
        end = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < end:
            try:
                elem = self._find(locator_type, locator)
                if hasattr(elem, 'is_displayed') and elem.is_displayed():
                    return elem
                # best-effort: if element exists, return it
                return elem
            except Exception:
                await asyncio.sleep(0.3)
        raise TimeoutError(f"Element {locator} not found after {timeout}s")

    async def tap(self, locator_type, locator, max_retries: int = 2):
        attempt = 0
        while True:
            attempt += 1
            try:
                elem = await self.wait_for_element(locator_type, locator)
                if hasattr(elem, 'click'):
                    elem.click()
                return True
            except Exception as e:
                logger.warning("Tap attempt %d failed for %s: %s", attempt, locator, e)
                if attempt > max_retries:
                    raise
                await asyncio.sleep(self.retry_delay * attempt)

    async def type_text(self, locator_type, locator, text: str, max_retries: int = 2):
        attempt = 0
        while True:
            attempt += 1
            try:
                elem = await self.wait_for_element(locator_type, locator)
                if hasattr(elem, 'clear'):
                    elem.clear()
                if hasattr(elem, 'send_keys'):
                    elem.send_keys(text)
                return True
            except Exception as e:
                logger.warning("Type attempt %d failed for %s: %s", attempt, locator, e)
                if attempt > max_retries:
                    raise
                await asyncio.sleep(self.retry_delay * attempt)
from appium.webdriver.common.appiumby import AppiumBy
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from appium.webdriver.webdriver import WebDriver
import asyncio

class ActionExecutor:
    def __init__(self, driver: WebDriver, default_timeout=15):
        self.driver = driver
        self.default_timeout = default_timeout

    async def tap(self, locator_type, locator):
        try:
            element = self.wait_for_element(locator_type, locator)
            element.click()
        except (TimeoutException, NoSuchElementException):
            # Retry after delay
            await asyncio.sleep(1)
            element = self.wait_for_element(locator_type, locator)
            element.click()

    def wait_for_element(self, locator_type, locator, timeout=None):
        timeout = timeout or self.default_timeout
        elapsed = 0
        interval = 0.5
        while elapsed < timeout:
            try:
                elem = self.driver.find_element(locator_type, locator)
                if elem.is_displayed():
                    return elem
            except Exception:
                pass
            asyncio.sleep(interval)
            elapsed += interval
        raise TimeoutException(f"Element {locator} not found after {timeout} seconds")

    async def type_text(self, locator_type, locator, text):
        elem = self.wait_for_element(locator_type, locator)
        elem.clear()
        elem.send_keys(text)
