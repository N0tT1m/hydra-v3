"""Allow running hydra_worker as a module: python -m hydra_worker"""

from hydra_worker.cli import main

if __name__ == "__main__":
    main()
