import os
import sys
import json

# ==============================
# Logging Management
# ==============================

# Logger class for logging to both console and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def _atomic_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=4)
    os.replace(tmp_path, path)
    
def make_log_func(multithread, log_lines=None):
    """
    Returns a log function:
      - multithread=True: appends to log_lines list (no terminal output)
      - multithread=False: acts like print() (terminal output captured by Logger)
    """
    if multithread:
        def log(msg):
            log_lines.append(str(msg))
        return log
    else:
        return print
