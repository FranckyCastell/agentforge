#!/usr/bin/env python3
"""AgentForge entry point.

Usage:
    python orchestrator.py              # interactive CLI
    python orchestrator.py --json-events  # JSON events over stdin/stdout
"""

from core import main

if __name__ == "__main__":
    main()
