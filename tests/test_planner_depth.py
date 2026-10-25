"""The Host depth receiver (dixiatielu/sim2real_scalecar#8): ``sim2real.planner.depth``.

Seams under test (the SSH client and the Onboard Computer's capture program are the
system boundaries):
- ``decode_packet``: SCDEPTH2 framing, CRC, bounds, and the 0.2 s age flag;
- ``probe_clock``: the NTP-style round-trip clock sync and its 10 ms requirement;
- ``DepthSource`` end to end, with a fake ``ssh`` on PATH that runs a fake capture
  program speaking the wire protocol of ``scripts/g1/depth_stream/depth_stream.cpp``:
  the SSH command line, Host-monotonic timestamps, dropping old frames, reconnecting.

    env -u PYTHONPATH HF_HUB_OFFLINE=1 UV_FROZEN=1 uv run --with pytest python -m pytest tests/test_planner_depth.py
"""

from __future__ import annotations

import hashlib
import io
import json
import socket
import struct
import sys
import textwrap
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

from sim2real.planner import depth
from sim2real.planner.depth import HEADER, decode_packet, probe_clock

REPO_ROOT = Path(__file__).resolve().parents[1]


def z16_image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(150, 4000, size=(480, 640), dtype=np.uint16)


def packet(values=None, stamp=10.0, **changes) -> bytes:
    values = z16_image() if values is None else values
    raw = values.astype("<u2").tobytes()
    payload = zlib.compress(raw, 1)
    fields = dict(magic=b"SCDEPTH2", version=2, size=HEADER.size, session=2, frame=1, device_ms=10000.0,
                  acquired=stamp, bracket=0.00001, width=640, height=480, domain=2, length=len(payload),
                  crc=zlib.crc32(raw))
    fields.update(changes)
    return HEADER.pack(*fields.values()) + payload


# ---------------------------------------------------------------------------
# The Onboard Computer's capture program: verbatim copies
# ---------------------------------------------------------------------------

DEPTH_STREAM = REPO_ROOT / "scripts" / "g1" / "depth_stream"
SCALECAR = Path("/home/galbot/projects/Scalecar_v2")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_capture_program_files_match_their_recorded_sha256():
    files = json.loads((DEPTH_STREAM / "SOURCES.json").read_text())["files"]
    assert {"depth_stream.cpp", "depth_clock.hpp"} <= set(files)
    for name, entry in files.items():
        assert sha256(DEPTH_STREAM / name) == entry["sha256"], name


@pytest.mark.skipif(not SCALECAR.is_dir(), reason="Scalecar_v2 checkout not present")
def test_the_capture_program_files_are_byte_identical_to_scalecar():
    for name, entry in json.loads((DEPTH_STREAM / "SOURCES.json").read_text())["files"].items():
        assert sha256(SCALECAR / entry["source"]) == entry["sha256"], name


# ---------------------------------------------------------------------------
# Framing and clock sync
# ---------------------------------------------------------------------------


def test_a_frame_is_stamped_on_the_host_clock_minus_the_10_ms_margin():
    image = z16_image(3)
    decoded = decode_packet(io.BytesIO(packet(image, stamp=10.0)), 3.0, 13.03)

    assert decoded["acquired"] == pytest.approx(12.99)  # 10.0 on the G1 clock + 3.0 offset - 0.01
    assert decoded["received"] == 13.03 and not decoded["delayed"] and not decoded["clock_rejected"]
    assert (decoded["session"], decoded["frame"]) == (2, 1)
    np.testing.assert_array_equal(decoded["z16"], image)


@pytest.mark.parametrize(
    "changes",
    [{"magic": b"SCDEPTH1"}, {"version": 1}, {"size": 80}, {"width": 160}, {"domain": 7}, {"bracket": 0.02},
     {"crc": 1}, {"length": 9_999_999}],
)
def test_malformed_frames_are_rejected(changes):
    with pytest.raises(ValueError):
        decode_packet(io.BytesIO(packet(**changes)), 3.0, 13.03)


def test_a_truncated_frame_is_a_closed_stream():
    with pytest.raises(EOFError):
        decode_packet(io.BytesIO(packet()[:-5]), 3.0, 13.03)


def test_a_payload_that_inflates_past_one_frame_is_rejected():
    payload = zlib.compress(b"\0" * (640 * 480 * 8))
    fields = list(HEADER.unpack(packet()[: HEADER.size]))
    fields[-2] = len(payload)
    with pytest.raises(ValueError, match="compressed"):
        decode_packet(io.BytesIO(HEADER.pack(*fields) + payload), 0.0, 10.01)


def test_frames_older_than_0_2_s_on_arrival_and_frames_from_the_future_are_flagged():
    assert not decode_packet(io.BytesIO(packet(stamp=10.0)), 0.0, 10.18)["delayed"]  # 0.19 s incl. the margin
    assert decode_packet(io.BytesIO(packet(stamp=10.0)), 0.0, 10.21)["delayed"]
    assert decode_packet(io.BytesIO(packet(stamp=10.0101742)), 0.0, 10.0)["clock_rejected"]
    assert not decode_packet(io.BytesIO(packet(stamp=10.02)), 0.0, 10.05)["clock_rejected"]


class ProbeSocket:
    """The capture program's side of the clock probe: its clock is the Host's plus ``offset``;
    each reply takes ``delays[i]`` seconds on the wire (split evenly both ways)."""

    def __init__(self, offset: float, delays) -> None:
        self.offset, self.delays = offset, list(delays)
        self.replies = io.BytesIO()

    def sendall(self, data: bytes) -> None:
        assert data == b"C"
        delay = self.delays.pop(0)
        time.sleep(delay / 2)
        remote = time.monotonic() + self.offset
        self.replies = io.BytesIO(struct.pack("<dd", remote, remote + 1e-6))
        time.sleep(delay / 2)

    def read(self, size: int) -> bytes:
        return self.replies.read(size)


def test_the_clock_sync_keeps_the_probe_with_the_shortest_round_trip():
    sock = ProbeSocket(offset=-5000.0, delays=[0.030, 0.004, 0.020])
    clock = probe_clock(sock, sock, count=3)

    assert clock["uncertainty_s"] == pytest.approx(0.002, abs=1e-3)
    assert clock["offset_s"] == pytest.approx(5000.0, abs=clock["uncertainty_s"] + 1e-3)
    assert len(clock["samples"]) == 3


def test_the_clock_sync_refuses_an_uncertainty_above_10_ms():
    sock = ProbeSocket(offset=0.0, delays=[0.025] * 3)
    with pytest.raises(ValueError, match="10 ms"):
        probe_clock(sock, sock, count=3)


# ---------------------------------------------------------------------------
# DepthSource end to end: a fake ``ssh`` that runs a fake capture program
# ---------------------------------------------------------------------------

FAKE_SSH = r'''
"""Fake ``ssh -T ... -L 127.0.0.1:LP:127.0.0.1:RP host 'exe --port RP'``: plays the tunnel and
the Onboard Computer's capture program (depth_stream.cpp's wire protocol) on 127.0.0.1:LP."""
import json, os, socket, struct, sys, time, zlib

HEADER = struct.Struct("<8sIIQQdddIIIII")
root = os.environ["FAKE_DEPTH_DIR"]
with open(os.path.join(root, "ssh_calls.jsonl"), "a") as calls:
    calls.write(json.dumps(sys.argv[1:]) + "\n")
attempt = sum(1 for _ in open(os.path.join(root, "ssh_calls.jsonl"))) - 1
scenario = json.load(open(os.path.join(root, "scenario.json")))
plan = scenario[min(attempt, len(scenario) - 1)]
argv = sys.argv[1:]
forward = argv[argv.index("-L") + 1].split(":")
local_port, remote_port = int(forward[1]), int(forward[3])
offset = plan.get("remote_offset", 5000.0)

def remote_now():
    return time.monotonic() + offset

if "exit" in plan:
    print(plan["exit"], file=sys.stderr, flush=True)
    sys.exit(1)
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", local_port))
listener.listen(1)
print(f"Depth listener ready on 127.0.0.1:{remote_port}", file=sys.stderr, flush=True)
listener.settimeout(15)
client, _ = listener.accept()

def recv1():
    data = client.recv(1)
    if not data:
        sys.exit(0)
    return data

def clock_reply(delay=0.0):
    time.sleep(delay)
    remote_in = remote_now()
    client.sendall(struct.pack("<dd", remote_in, remote_now()))

request = recv1()
while request == b"C":
    clock_reply(plan.get("probe_delay", 0.0))
    request = recv1()
assert request == b"S", request
for frame in plan.get("frames", []):
    time.sleep(frame.get("gap", 0.1))
    raw = struct.pack("<H", 1000 + frame["frame"]) * (640 * 480)
    payload = zlib.compress(raw, 1)
    crc = zlib.crc32(raw) ^ (0 if frame.get("crc_ok", True) else 1)
    header = HEADER.pack(b"SCDEPTH2", 2, HEADER.size, plan["session"], frame["frame"], 1000.0 * frame["frame"],
                         remote_now() - frame.get("age", 0.03), 0.0001, 640, 480, 2, len(payload), crc)
    client.sendall(header + payload)
    if recv1() != b"C":
        sys.exit(2)
    clock_reply()
if plan.get("then") == "close":
    time.sleep(0.2)
    client.close()
    sys.exit(0)
while client.recv(1):  # hold: stay connected, send nothing, until the receiver hangs up
    pass
'''


@pytest.fixture
def fake_onboard(tmp_path, monkeypatch):
    """Returns ``start(scenario) -> DepthSource``; one scenario entry per SSH connection attempt."""
    bin_dir, root, logs = tmp_path / "bin", tmp_path / "onboard", tmp_path / "logs"
    for path in (bin_dir, root, logs):
        path.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_SSH))
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("FAKE_DEPTH_DIR", str(root))
    sources = []

    def start(scenario, **options):
        (root / "scenario.json").write_text(json.dumps(scenario))
        source = depth.DepthSource(depth.depth_options({"local_port": free_port(), **options}), logs)
        sources.append(source)
        return source

    start.calls = lambda: [json.loads(line) for line in (root / "ssh_calls.jsonl").read_text().splitlines()]
    start.logs = logs
    yield start
    for source in sources:
        source.close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def collect(source, until, timeout_s: float = 20.0) -> list[dict]:
    """Every distinct packet ``poll`` returns (frames and errors, in order) until ``until(events)``."""
    events, last, deadline = [], None, time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        packet = source.poll()
        if packet is not None and packet is not last:
            last = packet
            key = packet.get("error") or (packet["session"], packet["frame"])
            if not events or events[-1]["key"] != key:
                events.append(dict(packet, key=key, polled_at=time.monotonic()))
                if until(events):
                    return events
        time.sleep(0.002)
    raise AssertionError(f"timed out; events so far: {[e['key'] for e in events]}")


def frames_of(events):
    return [e["key"] for e in events if "error" not in e]


def test_frames_arrive_on_the_host_clock_and_frames_older_than_0_2_s_are_dropped(fake_onboard):
    local_port = free_port()
    source = fake_onboard([dict(session=7, frames=[dict(frame=1, age=0.03), dict(frame=2, age=0.35),
                                                   dict(frame=3, age=0.05)], then="hold")],
                          ssh_host="unitree@192.168.123.164", remote_executable="/opt/depth/scalecar-depth",
                          port=18741, local_port=local_port)
    events = collect(source, lambda ev: (7, 3) in frames_of(ev))

    assert frames_of(events) == [(7, 1), (7, 3)]  # frame 2 was 0.35 s old on arrival
    for event in events:
        frame = event["frame"]
        assert event["depth_mm"].shape == (120, 160) and event["depth_mm"].dtype == np.uint16
        assert (event["depth_mm"] == 1000 + frame).all()
        # The capture program's clock runs 5000 s ahead; its stamps land on the Host's
        # monotonic clock, 10 ms early (the safety margin), up to the transport time.
        age = {1: 0.03, 3: 0.05}[frame]
        assert age + 0.01 - 0.002 <= event["received"] - event["acquired"] <= age + 0.01 + 0.02
        assert event["received"] <= event["polled_at"]
    call = fake_onboard.calls()[0]
    assert call == ["-T", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=5",
                    "-L", f"127.0.0.1:{local_port}:127.0.0.1:18741",
                    "unitree@192.168.123.164", "/opt/depth/scalecar-depth --port 18741"]
    clock = json.loads((fake_onboard.logs / "depth_clock.json").read_text())
    assert clock["uncertainty_s"] <= 0.01 and clock["offset_s"] == pytest.approx(-5000.0, abs=0.01)

    source.close()
    summary = json.loads((fake_onboard.logs / "depth_summary.json").read_text())
    assert summary["frames"] == 2 and summary["delayed"] == 1


def test_after_the_stream_closes_the_receiver_reports_it_and_reconnects_with_a_fresh_clock_sync(fake_onboard):
    source = fake_onboard([
        dict(session=1, frames=[dict(frame=1), dict(frame=2)], then="close"),
        dict(session=2, frames=[dict(frame=1), dict(frame=2)], then="hold", remote_offset=7000.0),
    ])
    events = collect(source, lambda ev: (2, 2) in frames_of(ev))

    keys = [e["key"] for e in events]
    error = next(i for i, e in enumerate(events) if "error" in e)
    assert keys[:error] == [(1, 1), (1, 2)] and keys[error + 1:] == [(2, 1), (2, 2)]
    assert "closed" in events[error]["error"]
    assert len(fake_onboard.calls()) == 2
    # The second connection measured its own clock offset.
    assert events[-1]["received"] - events[-1]["acquired"] == pytest.approx(0.04, abs=0.02)


def test_a_frame_with_a_bad_crc_drops_the_connection_and_the_receiver_reconnects(fake_onboard):
    source = fake_onboard([
        dict(session=1, frames=[dict(frame=1), dict(frame=2, crc_ok=False)], then="hold"),
        dict(session=2, frames=[dict(frame=1)], then="hold"),
    ])
    events = collect(source, lambda ev: (2, 1) in frames_of(ev))

    errors = [e["error"] for e in events if "error" in e]
    assert frames_of(events) == [(1, 1), (2, 1)] and len(errors) == 1 and "CRC" in errors[0]


def test_a_clock_sync_worse_than_10_ms_refuses_the_connection_until_a_retry_meets_it(fake_onboard):
    source = fake_onboard([
        dict(session=1, probe_delay=0.03, frames=[dict(frame=1)], then="hold"),
        dict(session=2, frames=[dict(frame=1)], then="hold"),
    ])
    events = collect(source, lambda ev: (2, 1) in frames_of(ev))

    assert "error" in events[0] and "10 ms" in events[0]["error"]
    assert frames_of(events) == [(2, 1)]


def test_a_capture_program_that_fails_to_start_is_reported_and_retried(fake_onboard):
    source = fake_onboard([
        dict(exit="Exactly one RealSense camera required"),
        dict(session=3, frames=[dict(frame=1)], then="hold"),
    ])
    events = collect(source, lambda ev: (3, 1) in frames_of(ev))

    assert "did not become ready" in events[0]["error"]
    assert "Exactly one RealSense camera required" in (fake_onboard.logs / "depth_source.log").read_text()


def test_the_transport_needs_no_mount_status_and_refuses_unknown_options():
    options = depth.depth_options()
    assert options["port"] == 18741 and options["local_port"] == 18742
    assert options["ssh_host"] and options["remote_executable"].startswith("/")
    with pytest.raises(ValueError, match="depth accepts only"):
        depth.depth_options({"mount_status": "calibrated"})
    with pytest.raises(ValueError, match="absolute"):
        depth.depth_options({"remote_executable": "scalecar-depth"})
    with pytest.raises(ValueError, match="SSH alias"):
        depth.depth_options({"ssh_host": "g1; rm -rf /"})
