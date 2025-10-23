import os
import logging
import sys
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# --- Color formatting on logs ---

LOG_COLORS = {
    "DEBUG": Fore.GREEN,
    "INFO": Fore.CYAN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Back.RED + Fore.WHITE,
}
RESET_COLOR = "\033[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL if log_color else ""
        record.levelname = f"{log_color}{record.levelname}{reset}"
        if log_color:
            # Color every line in the message
            lines = str(record.msg).splitlines()
            record.msg = "\n".join(
                f"{log_color}{line}{reset}" if line.strip() else line for line in lines
            )
        return super().format(record)


# --- Set up the LOGGER ---

# Get the desired log level
env_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_levels.get(env_log_level, logging.INFO)

# Create a named logger
agent_logger = logging.getLogger("agent")
agent_logger.setLevel(log_level)

# Remove all handlers associated with the logger object
if agent_logger.hasHandlers():
    agent_logger.handlers.clear()

# Create handler and formatter
handler = logging.StreamHandler(sys.stderr)
formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
agent_logger.addHandler(handler)

# Avoid duplicate logs
agent_logger.propagate = False
