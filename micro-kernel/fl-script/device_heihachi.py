# heihachi FL MIDI Controller Script
#
# The only real two-way channel into FL Studio: FL's scripting sandbox
# exposes transport, mixer track volume/pan, and pattern/channel selection.
# It does not expose plugin insertion, piano-roll editing, automation
# curves, or rendering -- nothing in this script or the daemon it talks to
# changes that.
#
# Install: copy this file into
#   Documents/Image-Line/FL Studio/Settings/Hardware/heihachi/device_heihachi.py
# then select "heihachi" as a MIDI input/output device in FL's MIDI settings
# (Options > MIDI Settings), input and output both pointed at it, so FL
# loads the script.
#
# Protocol: line-delimited JSON over a loopback TCP socket, daemon-side
# listener at the address printed in the `heihachi serve` banner as
# `fl-link` (default 127.0.0.1:7750). The daemon is the server; this script
# is the client, matching FL's scripting model where a device script is
# initialised once and then polled from `OnIdle`, never given control of
# its own thread.
#
# Command frames (daemon -> script), one JSON object per line:
#   {"cmd":"transport","action":"start"}
#   {"cmd":"transport","action":"stop"}
#   {"cmd":"transport","action":"record"}
#   {"cmd":"mixer_set","track":3,"param":"volume","value":0.8}
#   {"cmd":"mixer_set","track":3,"param":"pan","value":-0.2}
#   {"cmd":"pattern_jump","index":5}

import json
import socket

import transport
import mixer
import patterns

HOST = "127.0.0.1"
PORT = 7750

_sock = None
_buffer = b""


def OnInit():
    _connect()


def _connect():
    global _sock
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        s.connect_ex((HOST, PORT))
        _sock = s
    except OSError:
        _sock = None


def OnIdle():
    # FL scripts have no free-running loop; OnIdle is polled frequently
    # enough that a small non-blocking read here is effectively a socket
    # event loop without owning a thread.
    global _sock, _buffer
    if _sock is None:
        _connect()
        return

    try:
        chunk = _sock.recv(4096)
    except BlockingIOError:
        return
    except OSError:
        # peer dropped; the daemon's FlLink already treats this as
        # "not connected" on its next send, so just try to reconnect
        _sock = None
        return

    if not chunk:
        _sock = None
        return

    _buffer += chunk
    while b"\n" in _buffer:
        line, _buffer = _buffer.split(b"\n", 1)
        if line.strip():
            _dispatch(line)


def _dispatch(line):
    try:
        cmd = json.loads(line)
    except ValueError:
        return

    kind = cmd.get("cmd")
    if kind == "transport":
        _do_transport(cmd.get("action"))
    elif kind == "mixer_set":
        _do_mixer_set(cmd.get("track"), cmd.get("param"), cmd.get("value"))
    elif kind == "pattern_jump":
        _do_pattern_jump(cmd.get("index"))
    # An unrecognised command is simply not dispatched -- FL's API has no
    # concept of a command failing that the daemon needs to hear back, and
    # the daemon's own act already recorded what it requested regardless of
    # whether this script understood it.


def _do_transport(action):
    if action == "start":
        transport.start()
    elif action == "stop":
        transport.stop()
    elif action == "record":
        transport.record()


def _do_mixer_set(track, param, value):
    if track is None or value is None:
        return
    if param == "volume":
        mixer.setTrackVolume(track, value)
    elif param == "pan":
        mixer.setTrackPan(track, value)


def _do_pattern_jump(index):
    if index is None:
        return
    patterns.jumpToPattern(index)


def OnDeInit():
    global _sock
    if _sock is not None:
        try:
            _sock.close()
        except OSError:
            pass
        _sock = None
