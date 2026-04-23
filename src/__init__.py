"""
src/__init__.py
===============
src パッケージの公開 API。
sample.py では `from src import RobotSimulation` のみでアクセス可能。
"""

from src.simulation import RobotSimulation

__all__ = ["RobotSimulation"]
