from __future__ import annotations

from storage import AlarmRecord


def existing_web_alarm_control(action: str, alarm: AlarmRecord) -> None:
    """Adapter for the alarm-control functions currently living in web_app.py.

    Importing is delayed until an alarm action is executed.  This avoids a
    circular import when the future Flask endpoint constructs AgentService and
    also prevents ordinary history/report queries from loading the Web/CV stack.
    """

    from web_app import (
        restore_active_alarm_report,
        write_alarm_control_command,
        write_cancelled_alarm_report,
    )

    if action == "confirm":
        restore_active_alarm_report()
        write_alarm_control_command("yes")
    elif action == "cancel":
        write_cancelled_alarm_report()
        write_alarm_control_command("no")
    else:
        raise ValueError(f"不支持的报警控制动作：{action}")
