"""Natural-language agent facade for the foreign-object detection system."""

from .intents import Intent, IntentMatch, RuleBasedIntentRecognizer
from .integrations import existing_web_alarm_control
from .recognizers import (
    HybridIntentRecognizer,
    IntentRecognizer,
    RecognitionMode,
)
from .service import AgentService
from .tools import AgentTools, ImageDetectionOutcome, VideoDetectionOutcome

__all__ = [
    "AgentService",
    "AgentTools",
    "Intent",
    "IntentMatch",
    "IntentRecognizer",
    "HybridIntentRecognizer",
    "RecognitionMode",
    "RuleBasedIntentRecognizer",
    "ImageDetectionOutcome",
    "VideoDetectionOutcome",
    "existing_web_alarm_control",
]
