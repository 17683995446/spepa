"""SPEPA 资源自调节与背压模块。"""
from spepa.resource.controller import ResourceController, ThrottleSignal
from spepa.resource.backpressure import Backpressure, BackpressureLevel
from spepa.resource.degradation import DegradationMatrix, DegradationLevel
