from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from .intents import Intent, IntentMatch, RuleBasedIntentRecognizer


class RecognitionMode(str, Enum):
    """Intent-recognition execution mode.

    ``rules`` is fully deterministic. ``hybrid`` uses rules first and calls the
    injected model only for unresolved text. ``model`` lets the model classify
    all text while retaining the explicit-action guard for alarm controls.
    """

    RULES = "rules"
    HYBRID = "hybrid"
    MODEL = "model"


@runtime_checkable
class IntentRecognizer(Protocol):
    def recognize(
        self,
        text: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> IntentMatch:
        """Return one of the closed-set Intent values and optional slots."""


MODEL_BLOCKED_ACTIONS = {Intent.CONFIRM_ALARM, Intent.CANCEL_ALARM}


class HybridIntentRecognizer:
    """Rule-first adapter with an optional future model recognizer.

    The model is deliberately isolated behind ``IntentRecognizer``.  It cannot
    call tools or return arbitrary tool names; the existing router remains the
    only component allowed to map a closed-set intent to executable code.
    """

    def __init__(
        self,
        *,
        rule_recognizer: Optional[IntentRecognizer] = None,
        model_recognizer: Optional[IntentRecognizer] = None,
        mode: RecognitionMode | str = RecognitionMode.HYBRID,
        model_confidence_threshold: float = 0.75,
    ) -> None:
        self.rule_recognizer = rule_recognizer or RuleBasedIntentRecognizer()
        self.model_recognizer = model_recognizer
        self.mode = RecognitionMode(mode)
        if not 0.0 <= model_confidence_threshold <= 1.0:
            raise ValueError("model_confidence_threshold 必须在 0 到 1 之间")
        if self.mode == RecognitionMode.MODEL and self.model_recognizer is None:
            raise ValueError("model 模式必须提供 model_recognizer")
        self.model_confidence_threshold = model_confidence_threshold

    def recognize(
        self,
        text: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> IntentMatch:
        rule_match = self._validate(
            self.rule_recognizer.recognize(text, context=context), "rule_recognizer"
        )
        if self.mode == RecognitionMode.RULES:
            return rule_match

        if (
            self.mode == RecognitionMode.HYBRID
            and rule_match.intent != Intent.UNKNOWN
            and rule_match.confidence >= self.model_confidence_threshold
        ):
            return replace(
                rule_match,
                source="hybrid_rules",
                metadata={**rule_match.metadata, "mode": self.mode.value},
            )

        if self.model_recognizer is None:
            return replace(
                rule_match,
                source="hybrid_rules",
                metadata={
                    **rule_match.metadata,
                    "mode": self.mode.value,
                    "model_available": False,
                },
            )

        model_match = self._validate(
            self.model_recognizer.recognize(text, context=context), "model_recognizer"
        )
        base_metadata = {
            "mode": self.mode.value,
            "rule_intent": rule_match.intent.value,
            "rule_confidence": rule_match.confidence,
            "model_intent": model_match.intent.value,
            "model_confidence": model_match.confidence,
        }

        if model_match.intent in MODEL_BLOCKED_ACTIONS and rule_match.intent != model_match.intent:
            return IntentMatch(
                Intent.UNKNOWN,
                model_match.confidence,
                source="hybrid_safety_guard",
                metadata={
                    **base_metadata,
                    "blocked_model_action": model_match.intent.value,
                    "reason": "alarm_action_requires_explicit_rule_match",
                },
            )

        if model_match.confidence < self.model_confidence_threshold:
            return IntentMatch(
                Intent.UNKNOWN,
                model_match.confidence,
                source="hybrid_low_confidence",
                metadata={**base_metadata, "reason": "model_confidence_below_threshold"},
            )

        return replace(
            model_match,
            slots={**rule_match.slots, **model_match.slots},
            source="hybrid_model",
            metadata={**model_match.metadata, **base_metadata},
        )

    @staticmethod
    def _validate(match: IntentMatch, component: str) -> IntentMatch:
        if not isinstance(match, IntentMatch):
            raise TypeError(f"{component} 必须返回 IntentMatch")
        if not 0.0 <= match.confidence <= 1.0:
            raise ValueError(f"{component} 返回的 confidence 必须在 0 到 1 之间")
        return match
