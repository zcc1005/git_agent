"""Natural-language agent facade for the foreign-object detection system."""

from .intents import Intent, IntentMatch, RuleBasedIntentRecognizer
from .integrations import existing_web_alarm_control
from .recognizers import (
    HybridIntentRecognizer,
    IntentRecognizer,
    RecognitionMode,
)
from .service import AgentService
from .planners import SkillPlan, SkillPlanner, SkillPlanningError, SkillPlanStep
from .skills import RuntimeSkill, SkillRegistry, SkillResult, SkillSpec
from .tools import AgentTools, ImageDetectionOutcome, VideoDetectionOutcome
from .archive import ArchiveRangeResult, HistoricalStreamArchiveManager
from .streaming import (
    RtspStreamCapture,
    RtspStreamProbe,
    StreamCaptureResult,
    StreamProbeResult,
)
from .video_sources import (
    LongVideoSource,
    LongVideoSourceRegistry,
    RtspStreamSettings,
    VideoResolution,
    VideoSegment,
    VideoZone,
    load_video_source_registry,
)

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
    "ArchiveRangeResult",
    "HistoricalStreamArchiveManager",
    "existing_web_alarm_control",
    "RuntimeSkill",
    "SkillRegistry",
    "SkillResult",
    "SkillSpec",
    "SkillPlan",
    "SkillPlanner",
    "SkillPlanningError",
    "SkillPlanStep",
    "LongVideoSource",
    "LongVideoSourceRegistry",
    "RtspStreamSettings",
    "VideoResolution",
    "VideoSegment",
    "VideoZone",
    "load_video_source_registry",
    "RtspStreamProbe",
    "StreamProbeResult",
    "RtspStreamCapture",
    "StreamCaptureResult",
]
