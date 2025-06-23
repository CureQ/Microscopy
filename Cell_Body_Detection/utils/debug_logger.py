DEBUG_MODE_ENABLED = True  # Set to False to disable all debug logs

# Define log levels with severity (higher number means higher severity)
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,  # Added for completeness, can be used for very severe errors
}

# Set the current threshold for logging.
# Messages with severity equal to or greater than this level will be printed.
# For example, if set to "INFO", then INFO, WARNING, ERROR, CRITICAL messages will be shown.
# If set to "DEBUG", all messages will be shown.
CURRENT_LOG_LEVEL = "WARNING"  # Default to showing all debug messages initially


def log(message: str, level: str = "INFO"):
    """
    Logs a message to the console if DEBUG_MODE_ENABLED is True and
    the message's level is at or above the CURRENT_LOG_LEVEL.

    Args:
        message: The message to log.
        level: The log level string (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    if DEBUG_MODE_ENABLED:
        message_level_severity = LOG_LEVELS.get(
            level.upper(), 0
        )  # Default to 0 if level is unknown
        current_threshold_severity = LOG_LEVELS.get(
            CURRENT_LOG_LEVEL.upper(), LOG_LEVELS["INFO"]
        )  # Default threshold to INFO

        if message_level_severity >= current_threshold_severity:
            print(f"[{level.upper()}] {message}")


def set_log_level(level_name: str):
    """
    Sets the current logging level threshold.
    Args:
        level_name: The name of the log level to set (e.g., "DEBUG", "INFO").
                    Must be a key in LOG_LEVELS.
    """
    global CURRENT_LOG_LEVEL
    if level_name.upper() in LOG_LEVELS:
        CURRENT_LOG_LEVEL = level_name.upper()
        # Optional: log that the level changed, but be careful not to cause recursion if log level is too low
        # print(f"[SYSTEM] Log level set to {CURRENT_LOG_LEVEL}")
    else:
        print(
            f"[SYSTEM WARNING] Invalid log level: {level_name}. Keeping current level: {CURRENT_LOG_LEVEL}"
        )


# Example of how to change the log level from another part of the application:
# from ._debug_logger import set_log_level
# set_log_level("INFO")
