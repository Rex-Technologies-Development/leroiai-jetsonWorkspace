# V5 Action Protocol

This document is the contract between the Jetson ROS 2 stack and the VEX V5 brain.
Use it to develop the V5-side firmware.

It covers:

1. The serial wire format (Jetson → V5 commands, V5 → Jetson messages).
2. The action command JSON the V5 receives.
3. Every action id, its meaning, and expected V5 behavior.
4. The ACK the V5 must send back, and the block-state rules the Jetson enforces.

Source of truth on the Jetson side:

- Commands / gating: `src/leroi_decision/leroi_decision/decision_node.py`
- Serial framing / parsing: `src/leroi_serial_messenger/leroi_serial_messenger/serial_messenger_node.py`

---

## 1. Coordinate system & units

- Distances are **centimeters**. The VEX field is 365.76 cm × 365.76 cm (12 ft).
- Origin `(0, 0)` is the **bottom-left** of the field map. `+X` = right, `+Y` = up / forward.
- `heading` is in **degrees**.
- Each waypoint is `{ "x": <cm>, "y": <cm>, "heading": <deg> }`.

---

## 2. Serial wire format

### 2.1 Jetson → V5 (commands): binary-framed

Every payload the Jetson sends is wrapped in this frame:

| Field        | Type        | Bytes | Notes                                             |
|--------------|-------------|-------|---------------------------------------------------|
| Sync         | bytes       | 4     | `0xAA 0x55 0xCC 0x33` (constant, marks frame start) |
| Payload len  | int16 LE    | 2     | Length of payload in bytes, **including** trailing `\n` |
| Payload type | int16 LE    | 2     | See table below                                    |
| CRC32        | uint32 LE   | 4     | Standard CRC-32 (IEEE 802.3, same as `zlib.crc32`) over the payload bytes |
| Payload      | UTF-8 bytes | N     | JSON text, newline-terminated                      |

- Total header is 12 bytes, then `N` payload bytes.
- CRC and length are computed **after** the trailing `\n` is appended, so they include it.
- Max payload is 1024 bytes (`max_payload_bytes`).

**Payload types:**

| Type | Meaning           | Source topic                | Payload                                  |
|------|-------------------|-----------------------------|------------------------------------------|
| 1    | Raw passthrough   | `/serial_messenger/tx`      | arbitrary text                           |
| 5    | **ACTION command**| `/robot/command`            | action JSON (section 3) — the important one |
| 10   | Heartbeat         | internal timer (~5 s)       | `{"type":"HEARTBEAT","stamp_ns":...,"uptime_sec":...}` |

The V5 should:
- Scan the byte stream for the 4-byte sync pattern.
- Read length + type + CRC, then read `length` payload bytes.
- Verify CRC32 before acting. If it fails, drop the frame.
- For type 5, parse the JSON and execute the action.
- Heartbeats (type 10) can be used as a "Jetson is alive" watchdog; no ACK required.

### 2.2 V5 → Jetson (telemetry & ACKs): line-based CSV

The V5 sends **newline-terminated ASCII lines**. Each line the Jetson cares about
starts with the token `soutV5,` (the parser searches for `soutV5,` anywhere in the
line, so leading log noise is tolerated). Fields are comma-separated:

```
soutV5,<MESSAGE_TYPE>,<field2>,<field3>,...\n
```

Two message types are parsed specially:

**Odometry** (robot pose, send continuously, e.g. 20–50 Hz):

```
soutV5,ODOM,<x>,<y>,<heading>,<frame>\n
```

- `x`, `y` in cm; `heading` in degrees; `frame` is an integer counter.
- Published by the Jetson on `/v5/odom` and used to draw the robot on the map.

**Action acknowledgement** (see section 5):

```
soutV5,ACK,<command_id>,<status>[,<action_id>][,<key>=<value> ...]\n
```

Any other `soutV5,<TYPE>,...` line is passed through untyped (still published on
`/serial_messenger/rx_parsed`) — useful for debug/log lines.

---

## 3. Action command JSON (payload type 5)

This is the exact JSON object the V5 receives in a type-5 frame:

```json
{
  "type": "ACTION",
  "id": 7,
  "action_id": 2,
  "action_name": "COLLECT_BLOCKS",
  "points": [
    { "x": 250.0, "y": 200.0, "heading": 90.0 },
    { "x": 270.0, "y": 220.0, "heading": 45.0 }
  ],
  "params": {}
}
```

| Field         | Type    | Always present? | Meaning                                                        |
|---------------|---------|-----------------|----------------------------------------------------------------|
| `type`        | string  | yes             | Always `"ACTION"`.                                              |
| `id`          | int     | yes             | Unique command id. **Echo this back as `command_id` in the ACK.** |
| `action_id`   | int     | yes             | Which action to run (0–22, section 4).                          |
| `action_name` | string  | yes             | Human-readable name (for logging/debug; same mapping as table). |
| `points`      | array   | yes (may be `[]`)| Ordered waypoints `{x, y, heading}`. Meaning depends on action. |
| `params`      | object  | yes (may be `{}`)| Free-form extras for future use (speeds, flags, etc.).         |

Notes:
- `points` is **always an array** (possibly empty). `x`, `y`, `heading` are always
  present and finite floats — the Jetson normalizes them before sending.
- Only **one command is in flight at a time**. The Jetson will not send the next
  action until it receives the ACK for the current one (except STOP — see section 5).

---

## 4. Action catalog

`action_id` → behavior. The mapping is defined in `ACTION_NAMES` in `decision_node.py`.

| id | name                                    | uses `points`? | requires blocks? | summary |
|----|-----------------------------------------|----------------|------------------|---------|
| 0  | `STOP`                                  | no             | no               | Immediately stop all motion and mechanisms. |
| 1  | `MOVE_CHASSIS_ONLY`                      | **yes**        | no               | Drive the chassis through the given waypoint(s). No intake/scoring. |
| 2  | `COLLECT_BLOCKS`                         | **yes**        | no               | Prime intake, drive to the waypoints, and collect blocks. |
| 3  | `SCORE_LEFT_LONG_GOAL_ALLIANCE`         | no             | **yes**          | Auto-navigate to the left long goal (alliance side) and score. |
| 4  | `SCORE_LEFT_LONG_GOAL_OPPONENT`         | no             | **yes**          | Auto-navigate to the left long goal (opponent side) and score. |
| 5  | `SCORE_RIGHT_LONG_GOAL_ALLIANCE`        | no             | **yes**          | Auto-navigate to the right long goal (alliance side) and score. |
| 6  | `SCORE_RIGHT_LONG_GOAL_OPPONENT`        | no             | **yes**          | Auto-navigate to the right long goal (opponent side) and score. |
| 7  | `SCORE_MID_LEFT`                         | no             | **yes**          | Auto-navigate to the mid-left goal and score. |
| 8  | `SCORE_MID_RIGHT`                        | no             | **yes**          | Auto-navigate to the mid-right goal and score. |
| 9  | `SCORE_LOW_LEFT`                         | no             | **yes**          | Auto-navigate to the low-left goal and score. |
| 10 | `SCORE_LOW_RIGHT`                        | no             | **yes**          | Auto-navigate to the low-right goal and score. |
| 11 | `DESCORE_LEFT_LONG_GOAL_ALLIANCE_FIELD` | no             | no               | Remove blocks from left long goal (alliance side), field approach. |
| 12 | `DESCORE_LEFT_LONG_GOAL_ALLIANCE_WALL`  | no             | no               | Remove blocks from left long goal (alliance side), wall approach. |
| 13 | `DESCORE_LEFT_LONG_GOAL_OPPONENT_FIELD` | no             | no               | Remove blocks from left long goal (opponent side), field approach. |
| 14 | `DESCORE_LEFT_LONG_GOAL_OPPONENT_WALL`  | no             | no               | Remove blocks from left long goal (opponent side), wall approach. |
| 15 | `DESCORE_RIGHT_LONG_GOAL_ALLIANCE_FIELD`| no             | no               | Remove blocks from right long goal (alliance side), field approach. |
| 16 | `DESCORE_RIGHT_LONG_GOAL_ALLIANCE_WALL` | no             | no               | Remove blocks from right long goal (alliance side), wall approach. |
| 17 | `DESCORE_RIGHT_LONG_GOAL_OPPONENT_FIELD` | no             | no               | Remove blocks from right long goal (opponent side), field approach. |
| 18 | `DESCORE_RIGHT_LONG_GOAL_OPPONENT_WALL` | no             | no               | Remove blocks from right long goal (opponent side), wall approach. |
| 19 | `RAM_RIGHT_LONG_GOAL_OPPONENT`          | no             | no               | Ram the right long goal (opponent side). |
| 20 | `RAM_RIGHT_LONG_GOAL_ALLIANCE`          | no             | no               | Ram the right long goal (alliance side). |
| 21 | `RAM_LEFT_LONG_GOAL_OPPONENT`           | no             | no               | Ram the left long goal (opponent side). |
| 22 | `RAM_LEFT_LONG_GOAL_ALLIANCE`           | no             | no               | Ram the left long goal (alliance side). |

### Behavior details

- **`STOP` (0)** — Highest priority. The Jetson is allowed to send STOP even while
  waiting for another command's ACK, so the V5 must always be able to accept and
  act on it. Stop drive + mechanisms, abort the current action.
- **`MOVE_CHASSIS_ONLY` (1)** — Pure navigation. Drive through `points` in order to
  the final pose. No intake, no scoring. Used to reposition.
- **`COLLECT_BLOCKS` (2)** — Prime the intake mechanism, then drive to the `points`
  (block locations of interest) and collect. **This action requires `points`** — for
  now they are supplied manually; later they'll come from the perception package.
  On completion the robot is considered to be **holding blocks**.
- **`SCORE_*` (3–10)** — The robot **auto-navigates** to the target goal; no `points`
  are sent. These are **gated**: the Jetson rejects a score command unless a
  `COLLECT_BLOCKS` has completed first (see section 6). On completion the robot is
  considered **empty** again.
- **`DESCORE_*` (11–18)** — Remove opponent/alliance blocks from a goal. Auto-navigate;
  no `points`. The name encodes goal side (LEFT/RIGHT long goal), which alliance the
  goal belongs to (ALLIANCE/OPPONENT), and the approach side (FIELD/WALL).
- **`RAM_*` (19–22)** — Drive hard into the named goal. Auto-navigate; no `points`.

> The Jetson does **not** enforce that `points` is non-empty for action 2, nor that
> it is empty for 3–22. It forwards whatever it's given. Empty `points` on a collect
> means the robot has nowhere to drive — handle that defensively on the V5.

---

## 5. ACK protocol (V5 → Jetson)

After the V5 **finishes** an action, it sends exactly **one** ACK line:

```
soutV5,ACK,<command_id>,<status>[,<action_id>][,holding_blocks=<true|false>]\n
```

| Field           | Required | Notes                                                             |
|-----------------|----------|-------------------------------------------------------------------|
| `command_id`    | **yes**  | Must equal the `id` from the command being acknowledged.          |
| `status`        | **yes**  | A word, e.g. `DONE`. Uppercased by the Jetson. See below.         |
| `action_id`     | optional | The action that finished. Recommended for clarity; the Jetson infers it from the pending command if omitted. |
| `holding_blocks`| optional | `holding_blocks=true` / `holding_blocks=false`. The robot's own report of whether it currently holds blocks. **Authoritative** — overrides the Jetson's inference. |

Examples:

```
soutV5,ACK,7,DONE,2,holding_blocks=true      # collect (id 7) finished, now holding blocks
soutV5,ACK,8,DONE,5,holding_blocks=false     # score (id 8) finished, now empty
soutV5,ACK,9,DONE                            # generic "command 9 finished"
```

### Important behavioral rules for the V5

1. **One ACK per command, sent when the action is complete.** The Jetson treats *any*
   ACK whose `command_id` matches the in-flight command as "that action finished" and
   then sends the next command. Do **not** send progress/intermediate ACKs reusing the
   same `command_id`, or the Jetson will advance early.
2. There is currently **no success/failure distinction** consumed by the Jetson — an
   ACK means "done." You may still send a meaningful `status` (`DONE`, `FAILED`,
   `REJECTED`, `CANCELLED`, `TIMEOUT`) for logging; only `DONE`-style completion is
   acted on today. (If real failure handling is added later, it hooks in at the ACK
   handler in `decision_node.py`.)
3. **`holding_blocks` is the most reliable signal.** If the V5 reports it on every
   relevant ACK, the Jetson's block-state always matches reality, regardless of the
   inference rules in section 6.
4. If the Jetson sends no ACK within `command_timeout_sec` (default **15 s**), it
   gives up waiting and clears its pending state. Long actions should either finish
   within that window or the timeout should be raised on the Jetson side.

---

## 6. Block-state gating (Jetson side, for context)

The Jetson tracks a single boolean, `blocks_collected` (does the robot have blocks?),
to decide whether scoring is allowed. The V5 doesn't have to track this, but should
understand it:

- It becomes **true** when a `COLLECT_BLOCKS` (action 2) ACK is received.
- It becomes **false** when a `SCORE_*` (3–10) ACK is received.
- A `holding_blocks=...` field on any ACK overrides the above with the robot's report.
- **Scoring (3–10) is rejected** by the Jetson unless `blocks_collected` is true — i.e.
  you must complete a collect before a score will ever be forwarded to the V5.
- Descore (11–18) and ram (19–22) are **not** gated and do not change the flag.

So the normal cycle the V5 will see is: `COLLECT_BLOCKS` → (ACK) → `SCORE_*` → (ACK) →
`COLLECT_BLOCKS` → … Scoring twice without an intervening collect will never reach the
V5.

---

## 7. Quick reference

**Command in (type-5 frame payload):**
```json
{"type":"ACTION","id":7,"action_id":2,"action_name":"COLLECT_BLOCKS","points":[{"x":250.0,"y":200.0,"heading":90.0}],"params":{}}
```

**ACK out:**
```
soutV5,ACK,7,DONE,2,holding_blocks=true\n
```

**Odometry out (continuous):**
```
soutV5,ODOM,182.4,210.0,90.0,1234\n
```
