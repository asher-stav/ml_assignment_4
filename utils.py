from datetime import datetime
from rich import inspect
from global_config import CONFIG


def log(message):
    if CONFIG.get("debug", True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame = inspect.currentframe()
        caller = frame.f_back.f_code.co_name
        print(f"[{timestamp}] [{caller}] {message}")
