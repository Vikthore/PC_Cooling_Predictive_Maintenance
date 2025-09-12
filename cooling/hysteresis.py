# --- HYSTERESIS HELPERS ---

from typing import List, Dict, Tuple


def apply_hysteresis(
    status_raw: List[str],
    severity: List[float],
    t_start: List[float],
    t_end: List[float],
    *,
    red_enter_win: int = 3,
    red_exit_green_win: int = 5,
    yellow_enter_win: int = 2,
    yellow_exit_green_win: int = 3,
    escalate_yellow_to_red_win: int = 6,
    sticky_red: bool = True,
    cooloff_win: int = 6,
) -> Tuple[List[str], List[Dict]]:
    """
    Returns:
      status_hyst: stabilized status per window
      events: list of {"start","end","peak"} for each red episode (notify once)
    """
    state = "🟢 Healthy"
    status_hyst = []
    events: List[Dict] = []

    # counters
    red_enter_ctr = 0
    red_clear_ctr = 0
    yellow_enter_ctr = 0
    yellow_clear_ctr = 0
    yellow_streak = 0
    since_last_notify = cooloff_win

    current_event = None

    for i, (raw, sev, ts, te) in enumerate(zip(status_raw, severity, t_start, t_end)):
        # track streaks
        is_red = raw == "🔴 Faulty"
        is_yellow = raw == "🟡 Warning"
        is_green = raw == "🟢 Healthy"

        # --- escalation logic ---
        if state != "🔴 Faulty":
            if is_red:
                red_enter_ctr += 1
            else:
                red_enter_ctr = 0

            if is_yellow:
                yellow_enter_ctr += 1
                yellow_streak += 1
            else:
                yellow_enter_ctr = 0
                if is_green:
                    yellow_streak = 0
        else:
            # already red
            red_enter_ctr = 0
            yellow_enter_ctr = 0
            yellow_streak = 0

        # --- de-escalation logic ---
        if state == "🔴 Faulty":
            if is_green:
                red_clear_ctr += 1
            elif sticky_red and is_yellow:
                red_clear_ctr = 0  # ignore yellow while red if sticky
            else:
                red_clear_ctr = 0
        elif state == "🟡 Warning":
            if is_green:
                yellow_clear_ctr += 1
            else:
                yellow_clear_ctr = 0
        else:
            red_clear_ctr = 0
            yellow_clear_ctr = 0

        # --- transitions ---
        transitioned = False

        # 1) escalate from non-red to red
        if state != "🔴 Faulty" and (
            red_enter_ctr >= red_enter_win
            or yellow_streak >= escalate_yellow_to_red_win
        ):
            state = "🔴 Faulty"
            transitioned = True
            red_enter_ctr = yellow_streak = 0
            # start/latch event
            current_event = {"start": ts, "peak": float(sev)}
            # rate-limited notify marker via events list (consumer decides how)
            if since_last_notify >= cooloff_win:
                current_event["notify"] = True
                since_last_notify = 0

        # 2) clear red back to green (never to yellow if sticky)
        elif state == "🔴 Faulty" and red_clear_ctr >= red_exit_green_win:
            state = "🟢 Healthy"
            transitioned = True
            red_clear_ctr = 0
            if current_event:
                current_event["end"] = te
                events.append(current_event)
                current_event = None

        # 3) non-red transitions: green <-> yellow with dwell
        elif state == "🟢 Healthy" and yellow_enter_ctr >= yellow_enter_win:
            state = "🟡 Warning"
            transitioned = True
            yellow_enter_ctr = 0
        elif state == "🟡 Warning" and yellow_clear_ctr >= yellow_exit_green_win:
            state = "🟢 Healthy"
            transitioned = True
            yellow_clear_ctr = 0

        # update outputs and bookkeeping
        status_hyst.append(state)
        since_last_notify += 1
        if current_event:
            current_event["peak"] = max(current_event["peak"], float(sev))

    return status_hyst, events
