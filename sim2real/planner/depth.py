"""The Host depth receiver: SSH tunnel, bounded decoding, clock sync and latest-frame IPC.

Ported from Scalecar_v2 ``hardware/depth.py`` (SOURCES.json). It starts the Onboard
Computer's capture program (``scripts/g1/depth_stream``) through ``ssh -T -L`` (key login,
``BatchMode``), maps its clock onto the Host's ``time.monotonic`` by NTP-style round trips
(uncertainty at most 10 ms), decodes ``SCDEPTH2`` frames (CRC checked), drops frames older
than 0.2 s on arrival, downsamples them to 120x160 and reconnects after any error.
Differences from the source:

- no ``mount_status`` option: that gate belongs to Scalecar's commissioning, dropped by ADR-0002;
- no TraceWriter recording of every frame; the log directory keeps the capture program's
  stderr (``depth_source.log``), the clock mapping (``depth_clock.json``) and the summary
  (``depth_summary.json``);
- ``DEFAULT_OPTIONS`` / ``depth_options`` fill in the transport settings.
"""

from __future__ import annotations

import json
from collections import deque
import multiprocessing as mp
import os
from pathlib import Path
import re
import shlex
import socket
import struct
import subprocess
import time
from typing import Any, BinaryIO
import zlib

import numpy as np

from .mailbox import LatestMailbox

HEADER = struct.Struct("<8sIIQQdddIIIII")
RAW_BYTES = 640 * 480 * 2
MAX_PAYLOAD = RAW_BYTES + 4096
DEFAULT_OPTIONS = dict(
    ssh_host="unitree@192.168.123.164",
    remote_executable="/home/unitree/sim2real_depth_stream/build/scalecar-depth",
    port=18741,
    local_port=18742,
    cpu=None,
)


def read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read bounded fixed-length data, rejecting truncated frames."""
    if not 0 <= size <= MAX_PAYLOAD:
        raise ValueError("Depth packet length outside bounds")
    parts = bytearray()
    while len(parts) < size:
        value = stream.read(size - len(parts))
        if not value:
            raise EOFError("Depth stream closed during packet")
        parts.extend(value)
    return bytes(parts)


class SocketReader:
    """Poll a socket without losing a partially read packet on a timeout."""

    def __init__(self, sock, stop, idle_seconds=30.0):
        self.sock, self.stop, self.idle_seconds = sock, stop, idle_seconds
        self.last_progress = time.monotonic()

    def read(self, size):
        while not self.stop.is_set():
            try:
                value = self.sock.recv(size)
                if value:
                    self.last_progress = time.monotonic()
                return value
            except (socket.timeout, InterruptedError):
                if time.monotonic() - self.last_progress > self.idle_seconds:
                    raise TimeoutError("Depth transport idle; reconnecting")
        raise InterruptedError("Depth source closing")

    def close(self):
        pass


def decode_packet(stream: BinaryIO, offset: float, received: float | None = None) -> dict:
    """Decode v2 Z16 mm; invalid clock samples are marked for recoverable dropping."""
    fields = HEADER.unpack(read_exact(stream, HEADER.size))
    magic, version, size, session, frame, device_ms, acquired, bracket, w, h, domain, n, crc = (
        fields
    )
    if (
        magic != b"SCDEPTH2"
        or version != 2
        or size != HEADER.size
        or (w, h) != (640, 480)
        or domain not in (0, 1, 2)
        or not 0 < n <= MAX_PAYLOAD
        or not np.isfinite([device_ms, acquired, bracket]).all()
        or not 0 <= bracket <= 0.01
    ):
        raise ValueError("Invalid depth header/clock/format")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(read_exact(stream, n), RAW_BYTES + 1)
    if (
        len(raw) != RAW_BYTES
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or zlib.crc32(raw) != crc
    ):
        raise ValueError("Invalid compressed depth length or CRC")
    mapped_acquisition = acquired + offset
    # Old SDK global-clock mapping can lead by a few milliseconds. Keep the
    # exact mapped acquisition in evidence and use a conservative 10 ms bound
    # for sampling/freshness rather than substituting network receipt time.
    stamp = mapped_acquisition - 0.01
    received = time.monotonic() if received is None else received
    return dict(
        session=session,
        frame=frame,
        device_ms=device_ms,
        domain=domain,
        acquired=stamp,
        received=received,
        delayed=received - stamp > 0.2,
        clock_rejected=received < stamp,
        mapping_bracket_s=bracket,
        mapped_acquisition=mapped_acquisition,
        timestamp_safety_margin_s=0.01,
        z16=np.frombuffer(raw, dtype="<u2").reshape(480, 640).copy(),
    )


def probe_clock(
    sock: socket.socket, stream: BinaryIO, count: int = 12, *, require_precision: bool = True
) -> dict:
    """NTP-style monotonic mapping; smallest RTT bounds transport uncertainty."""
    samples = []
    for _ in range(count):
        before = time.monotonic()
        sock.sendall(b"C")
        remote_in, remote_out = struct.unpack("<dd", read_exact(stream, 16))
        after = time.monotonic()
        delay = (after - before) - (remote_out - remote_in)
        if not np.isfinite([remote_in, remote_out]).all() or remote_out < remote_in or delay < 0:
            raise ValueError("Invalid clock probe")
        samples.append(
            dict(
                offset_s=((before - remote_in) + (after - remote_out)) / 2,
                uncertainty_s=delay / 2,
                host_time=after,
            )
        )
    best = min(samples, key=lambda v: v["uncertainty_s"])
    if require_precision and best["uncertainty_s"] > 0.01:
        raise ValueError("G1 clock mapping uncertainty exceeds 10 ms")
    return best | dict(samples=samples, method="minimum_round_trip_monotonic")


def validate_depth(options: dict[str, Any]) -> dict[str, Any]:
    """Explicit transport configuration; no shell commands accepted from JSON."""
    allowed = {"ssh_host", "remote_executable", "port", "local_port", "cpu"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError(f"depth accepts only {sorted(allowed)}")
    result = dict(options)
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", result.get("ssh_host", "")):
        raise ValueError("depth.ssh_host must be an SSH alias or hostname")
    if not str(result.get("remote_executable", "")).startswith("/"):
        raise ValueError("depth.remote_executable must be absolute on G1")
    for key in ("port", "local_port"):
        if type(result.get(key)) is not int or not 1024 <= result[key] <= 65535:
            raise ValueError(f"depth.{key} must be a TCP port in [1024,65535]")
    if result.get("cpu") is not None and type(result["cpu"]) is not int:
        raise ValueError("depth.cpu must be an integer or null")
    return result


def depth_options(options: dict[str, Any] | None = None) -> dict[str, Any]:
    """``DEFAULT_OPTIONS`` overridden by ``options``, validated."""
    return validate_depth({**DEFAULT_OPTIONS, **(options or {})})


def _capture_once(options: dict, mailbox: LatestMailbox, stop: Any, output: str) -> dict:
    from .scalecar_dp.dp.depth_preprocess import downsample_z16

    if options.get("cpu") is not None:
        os.sched_setaffinity(0, {options["cpu"]})
    process = sock = stream = None
    report = dict(frames=0, dropped=0, delayed=0, error=None, clock=None)
    intervals = deque(maxlen=108000)  # One hour; raw per-frame timestamps remain recorded.
    try:
        with open(Path(output) / "depth_source.log", "ab") as errors:
            log_offset = errors.tell()
            command = [options["remote_executable"], "--port", str(options["port"])]
            process = subprocess.Popen(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ConnectTimeout=5",
                    "-L",
                    f"127.0.0.1:{options['local_port']}:127.0.0.1:{options['port']}",
                    options["ssh_host"],
                    shlex.join(command),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errors,
            )
            deadline = time.monotonic() + 10
            # A forwarded local TCP connect may succeed before the remote listener
            # exists. Wait for its explicit readiness message before opening a channel.
            while (
                not stop.is_set()
                and "Depth listener ready"
                not in (Path(output) / "depth_source.log").read_text()[log_offset:]
            ):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("G1 depth source did not become ready; see depth_source.log")
                stop.wait(0.05)
            while not stop.is_set():
                try:
                    sock = socket.create_connection(("127.0.0.1", options["local_port"]), timeout=2)
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(
                            "Cannot connect timestamped G1 depth source; see depth_source.log"
                        )
                    stop.wait(0.1)
            if stop.is_set():
                return
            sock.settimeout(0.25)
            stream = SocketReader(sock, stop)
            report["clock"] = probe_clock(sock, stream)
            try:
                (Path(output) / "depth_clock.json").write_text(
                    json.dumps(report["clock"], indent=2) + "\n"
                )
            except OSError as error:
                report["clock_log_error"] = str(error)
            sock.settimeout(0.25)
            sock.sendall(b"S")
            last = None
            while not stop.is_set():
                packet = decode_packet(stream, report["clock"]["offset_s"])
                # v2 requests a fresh clock pair after every frame, so clock
                # drift cannot accumulate during a long physical session.
                clock = probe_clock(sock, stream, count=1, require_precision=False)
                if clock["uncertainty_s"] <= 0.01:
                    report["clock"] = clock
                else:
                    report["delayed_clock_probes"] = report.get("delayed_clock_probes", 0) + 1
                if packet["clock_rejected"]:
                    report["clock_rejected"] = report.get("clock_rejected", 0) + 1
                    continue
                if packet["delayed"]:
                    report["delayed"] += 1
                    continue
                if last is not None:
                    if packet["session"] != last["session"]:
                        report["session_changes"] = report.get("session_changes", 0) + 1
                        last = None
                    if last is not None and (
                        packet["frame"] <= last["frame"]
                        or packet["acquired"] <= last["acquired"]
                    ):
                        report["regressed_frames"] = report.get("regressed_frames", 0) + 1
                        continue
                    if last is not None:
                        intervals.append((packet["acquired"] - last["acquired"]) * 1000)
                last = packet
                frame = downsample_z16(packet.pop("z16"))
                packet["depth_mm"] = frame
                packet["clock_uncertainty_s"] = report["clock"]["uncertainty_s"]
                # Latest-frame delivery is independent of disk backlog.
                report["frames"] += 1
                report["dropped"] += not mailbox.publish(packet)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        mailbox.publish(dict(error=report["error"]))
    finally:
        if stream is not None:
            stream.close()
        if sock is not None:
            sock.close()
        if process is not None:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        report["intervals_ms"] = list(intervals)
        return report


def _capture(options, mailbox, stop, output):
    """Reconnect after EOF/errors; each new stream receives its own clock mapping."""
    total = dict(
        frames=0,
        dropped=0,
        delayed=0,
        reconnects=0,
        error=None,
        last_error=None,
        intervals_ms=[],
    )
    while not stop.is_set():
        report = _capture_once(options, mailbox, stop, output)
        for key in ("frames", "dropped", "delayed"):
            total[key] += report.get(key, 0)
        total["intervals_ms"] = (total["intervals_ms"] + report["intervals_ms"])[-108000:]
        total["clock"] = report["clock"]
        if report["error"] and not stop.is_set():
            total["last_error"] = report["error"]
            total["reconnects"] += 1
            stop.wait(1.0)
    try:
        (Path(output) / "depth_summary.json").write_text(json.dumps(total, indent=2) + "\n")
    except OSError as error:
        mailbox.publish(dict(error=f"Depth summary write failed: {error}"))


class DepthSource:
    """Own a camera child; callers poll a latest-only slot without blocking."""

    def __init__(self, options: dict, output: Path):
        context = mp.get_context("spawn")
        self.mailbox = LatestMailbox(context)
        self.stop = context.Event()
        self.process = context.Process(
            target=_capture,
            args=(validate_depth(options), self.mailbox, self.stop, str(output)),
            daemon=False,
        )
        self.version = 0
        self.latest = None
        self.process.start()

    def poll(self) -> dict | None:
        self.version, packet = self.mailbox.read(self.version)
        if packet is not None:
            self.latest = packet
        if not self.process.is_alive() and (self.latest is None or "error" not in self.latest):
            self.latest = dict(error="Depth capture process exited")
        return self.latest

    def close(self) -> None:
        self.stop.set()
        self.process.join(timeout=12)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
