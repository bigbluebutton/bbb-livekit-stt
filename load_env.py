"""Load .env into the environment.

main.py imports this first. config builds its singletons from the environment
at import. prometheus_client picks its value class from PROMETHEUS_MULTIPROC_DIR
on its first import. Both must see the values from .env.
"""

from dotenv import load_dotenv

load_dotenv()
