"""SPEPA core 包：核心模块。"""
from spepa.core.state_bus import StateBus
from spepa.core.predictive_model import PredictiveModel, PredictiveLayer
from spepa.core.free_energy import FreeEnergyCalculator
from spepa.core.drive_engine import DriveEngine, Homeostasis
from spepa.core.gap_detector import GapDetector, GapInfo, GAP_LEVELS
from spepa.core.goal_hierarchy import GoalHierarchy, Goal
from spepa.core.active_inference import ActiveInference, Policy
from spepa.core.metacognition import Metacognition, ConfidenceCalibrator
from spepa.core.memory_system import MemorySystem, FastMemory, SlowMemory, MemoryItem
from spepa.core.optimizer import Optimizer
from spepa.core.safety_guard import SafetyGuard
from spepa.core.self_model import SelfModel

__all__ = [
    "StateBus",
    "PredictiveModel",
    "PredictiveLayer",
    "FreeEnergyCalculator",
    "DriveEngine",
    "Homeostasis",
    "GapDetector",
    "GapInfo",
    "GAP_LEVELS",
    "GoalHierarchy",
    "Goal",
    "ActiveInference",
    "Policy",
    "Metacognition",
    "ConfidenceCalibrator",
    "MemorySystem",
    "FastMemory",
    "SlowMemory",
    "MemoryItem",
    "Optimizer",
    "SafetyGuard",
    "SelfModel",
]
