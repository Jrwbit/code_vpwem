#!/usr/bin/env python3
"""Execute the reviewed arm-stow/ordered-waist entry via supervised HMI only.

Requires a route from search_waist_entry.py and an independently verified entry
receipt. The caller must have obtained the operator's field/fault confirmation.
This never changes calibration, safety gates, or detection/insertion settings.
"""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("route", type=Path)
    ap.add_argument("receipt", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--field-confirmed", action="store_true", required=True)
    ap.add_argument("--faults-checked", action="store_true", required=True)
    args = ap.parse_args()
    route = json.loads(args.route.read_text())
    receipt = json.loads(args.receipt.read_text())
    if not (receipt["independent_process_verified"] and receipt["kinematic_signature_matches"]
            and receipt["start_feedback_matches"] and receipt["minimum_clearance_mm"] >= 15):
        raise ValueError("Missing independently checked entry route")
    sol = route["solution"]
    waist_route = sol["waist_route_if_sequential"]
    if waist_route[0] != route["start_waist_rad"] or waist_route[-1] != route["target_waist_rad"]:
        raise ValueError("Waist route endpoints differ")
    args.output.mkdir(parents=True, exist_ok=False)
    csrf = None
    active_operation = None

    def request(path, body=None):
        headers = {"Content-Type": "application/json"}
        if csrf:
            headers["X-CSRF-Token"] = csrf
        req = urllib.request.Request("http://127.0.0.1:50168" + path,
            data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from None

    def state_vectors(state):
        j = state["real"]["joints"]
        return {"waist": [j[f"{i}-Joint"] for i in range(1, 5)],
                "left_arm": [j[f"AR5-5_07L_joint_{i}"] for i in range(1, 8)],
                "right_arm": [j[f"AR5-5_07R_joint_{i}"] for i in range(1, 8)],
                "head": [j[f"{i}-Joint"] for i in (5, 6)]}

    def delta(a, b):
        return max(abs(x-y) for x, y in zip(a, b))

    expected = {"waist": route["start_waist_rad"], "left_arm": route["start_left_rad"],
                **route["static_pose"]}

    def guard(state, tolerance):
        if state["motion"]["busy"] or state["motion"]["latched"] or state["plan"]["active"]:
            raise ValueError("Controller busy or latched")
        for part in state["real"]["components"].values():
            if (part["ok"] is not True or not 0 <= part["age_s"] < 1
                    or (part["power"], part["mode"], part["op"]) != ("off", "manual", "idle")):
                raise ValueError("Fresh off/manual/idle feedback required")
        actual = state_vectors(state)
        if any(delta(actual[name], values) > tolerance for name, values in expected.items()):
            raise ValueError("Actual pose differs from checked route")

    events = args.output / "events.jsonl"
    def record(event):
        with events.open("a") as f:
            f.write(json.dumps({"time": time.time(), **event}) + "\n")
        print(json.dumps(event), flush=True)

    try:
        state = request("/state")
        csrf = state["motion"]["csrf_token"]
        guard(state, math.radians(.1))
        stages = [("left_arm", sol["left_stow_rad"])] + [("waist", w) for w in waist_route[1:]]
        for stage, (group, final) in enumerate(stages):
            initial = expected[group][:]
            step_deg = 10 if group == "left_arm" else 8
            speed = 5 if group == "left_arm" else 2
            if delta(initial, final) < math.radians(.1):
                continue
            pieces = max(1, math.ceil(delta(initial, final) / math.radians(step_deg)))
            for piece in range(1, pieces+1):
                target = [a + (b-a)*piece/pieces for a, b in zip(initial, final)]
                state = request("/state")
                csrf = state["motion"]["csrf_token"]
                guard(state, math.radians(1))
                names = ([f"{i}-Joint" for i in range(1, 5)] if group == "waist"
                         else [f"AR5-5_07L_joint_{i}" for i in range(1, 8)])
                sync = request("/sim/joints", {"joints": dict(zip(names, target))})
                if sync.get("ok") is not True:
                    raise RuntimeError(str(sync))
                proposal = request(f"/send_preview?group={group}&max_step_deg={step_deg+1}&joint_speed_percent={speed}")["proposal"]
                if proposal.get("ok") is not True:
                    raise RuntimeError(str(proposal))
                record({"event": "starting", "stage": stage, "piece": piece, "pieces": pieces,
                        "group": group, "target_model_rad": target, "speed_percent": speed})
                sent = request("/send", {"group": group, "token": proposal["token"],
                    "confirmation": True, "ack": "HELIOS_WEB_ACK::"+group,
                    "field_confirmed": True, "faults_checked": True, "max_step_deg": step_deg+1})
                if sent.get("ok") is not True:
                    raise RuntimeError(str(sent))
                active_operation = sent["operation_id"]
                deadline = time.monotonic()+100
                while True:
                    request("/motion/heartbeat", {"operation_id": active_operation})
                    state = request("/state")
                    motion = state["motion"]
                    if motion["operation_id"] != active_operation:
                        raise RuntimeError("Operation ownership changed")
                    if not motion["busy"]:
                        result = motion["last_result"]
                        record({"event": "native_result", "stage": stage, "piece": piece, "result": result})
                        if not (result.get("ok") and result.get("status") == "completed"
                                and result.get("safe_exit_confirmed") is True):
                            raise RuntimeError("Native move did not complete safely")
                        active_operation = None
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Supervised move deadline exceeded")
                    time.sleep(.25)
                expected[group] = target
                guard(state, math.radians(1))
                time.sleep(3)
        final_state = request("/state")
        guard(final_state, math.radians(1))
        record({"event": "entry_completed", "actual_model_rad": state_vectors(final_state),
                "controller_writes": True, "detection_completed": False})
    except BaseException:
        if active_operation:
            request("/stop", {"operation_id": active_operation})
        raise


if __name__ == "__main__":
    main()
