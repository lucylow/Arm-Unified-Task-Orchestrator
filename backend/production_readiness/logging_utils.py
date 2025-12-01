
import os
from PIL import Image, ImageFilter
from datetime import datetime
import hashlib
import logging
from typing import Optional

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("ProductionReadinessLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def mask_sensitive(image_path: str, output_path: str, blur_radius: int = 8):
    """Blur the entire screenshot (or defined bounding boxes) to mask sensitive info."""
    try:
        img = Image.open(image_path)
        masked = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        masked.save(output_path)
        logger.info(f"Masked screenshot saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to mask screenshot {image_path}: {e}")

def log_event(event_type: str, details: str, level="INFO"):
    """Logs a structured event to a file."""
    ts = datetime.utcnow().isoformat()
    line = f"{ts} [{level}] {event_type}: {details}\n"
    try:
        with open(os.path.join(LOG_DIR, "events.log"), "a") as f:
            f.write(line)
        logger.debug(f"Logged event: {event_type}")
    except Exception as e:
        logger.error(f"Failed to write event to log file: {e}")

def store_secure_log(filename: str, content: str, encryption_key: str = "secure_key_placeholder"):
    """Store encrypted logs using a simple XOR cipher for demonstration."""
    try:
        key = hashlib.sha256(encryption_key.encode()).digest()
        encrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(content.encode())])
        path = os.path.join(LOG_DIR, filename + ".enc")
        with open(path, "wb") as f:
            f.write(encrypted)
        logger.info(f"Secure log stored at {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to store secure log {filename}: {e}")
        return None

def decrypt_secure_log(filepath: str, encryption_key: str = "secure_key_placeholder") -> Optional[str]:
    """Decrypts a secure log file."""
    try:
        with open(filepath, "rb") as f:
            encrypted_content = f.read()
        key = hashlib.sha256(encryption_key.encode()).digest()
        decrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(encrypted_content)]).decode()
        logger.info(f"Secure log decrypted from {filepath}")
        return decrypted
    except Exception as e:
        logger.error(f"Failed to decrypt secure log {filepath}: {e}")
        return None

