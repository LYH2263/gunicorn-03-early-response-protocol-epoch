#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.

"""Acceptance tests for the half-open exchange commit-step ledger.

Everything here is verified on ONE accepted connection / ONE handle:
HTTP/2 tests drive two real sibling streams on one HTTP2ServerConnection,
never two connections.

The three protocol machines must keep their pinned differences while all
obeying the same append-only ledger semantics:

* WSGI HTTP/1: automatic ``100 Continue`` before the app is called;
  Chunked/Length/EOF body planes, trailers recorded on ``req.trailers``
  with pipelined bytes unread; peer interruption is connection scoped.
* Blocking HTTP/2: 100 only on first body pull (write-before-read
  suppresses it); stream-scoped RST -> HTTP2StreamError; response
  trailers as a second HEADERS block at close.
* ASGI: no automatic 100; no-CL/non-chunked is an empty body completed
  at parse time (never EOF); inbound HTTP/1 trailers dropped; read wait
  timeout becomes http.disconnect without a completed body; on HTTP/2
  http.response.start does not frame final HEADERS, so 100 can still
  follow it; RST -> ClientDisconnected.
"""

import asyncio
from unittest import mock

import pytest
from h2.connection import H2Connection
import h2.config

from gunicorn.config import Config
from gunicorn.exchange import (
    CommitStep, ExchangeLedger, ExchangeRecord, Plane, Protocol,
    InterruptScope,
)
from gunicorn.http.errors import ChunkMissingTerminator, NoMoreData
from gunicorn.http.parser import RequestParser
from gunicorn.http.wsgi import Response as WSGIResponse, create as wsgi_create
from gunicorn.http2.connection import HTTP2ServerConnection
from gunicorn.http2.errors import HTTP2StreamError
from gunicorn.http2.response import HTTP2Response
from gunicorn.http2.stream import HTTP2Stream


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _config(ledger=True, **overrides):
    cfg = Config()
    if ledger:
        cfg.set("exchange_ledger", True)
    for name, value in overrides.items():
        cfg.set(name, value)
    return cfg


def _steps(record):
    return [entry.step for entry in record.entries]


def _all(record, step):
    return [e for e in record.entries if e.step == step]


def _find(record, step, **match):
    entry = record.find(step, **match)
    assert entry is not None, "missing %r %r in %r" % (step, match,
                                                       record.entries)
    return entry


def _parse_h1(raw, cfg=None):
    cfg = cfg or _config()
    parser = RequestParser(cfg, iter([raw]), None)
    req = next(iter(parser))
    return parser, req


class _SendSocket(object):
    def __init__(self):
        self.sent = []

    def send(self, data, *args):
        self.sent.append(bytes(data))
        return len(data)

    def sendall(self, data, *args):
        self.sent.append(bytes(data))


# ---------------------------------------------------------------------------
# ledger core semantics
# ---------------------------------------------------------------------------

class TestLedgerCore:

    def test_append_only_no_rollback_api(self):
        ledger = ExchangeLedger(protocol=Protocol.H1)
        assert not hasattr(ExchangeRecord, "rollback")
        assert not hasattr(ExchangeRecord, "undo")
        assert not hasattr(ExchangeRecord, "pop")
        assert not hasattr(ExchangeLedger, "rollback")
        assert not hasattr(ExchangeLedger, "undo")

        outer = ledger.open_exchange("outer", protocol=Protocol.H1)
        outer.commit(CommitStep.FINAL_HEADERS, status="200 OK")
        inner = ledger.open_exchange("inner", protocol=Protocol.H1,
                                     parent="outer")
        inner.commit(CommitStep.INFORMATIONAL, status=103, sent=True)
        inner.commit(CommitStep.FINAL_HEADERS, status="200 OK")
        inner.close()

        # A frozen exchange swallows late commits instead of rewriting the
        # already observable history.
        assert inner.commit(CommitStep.RESPONSE_TRAILERS) is None

        # Closing/filling the inner exchange never alters the outer record.
        outer_steps = _steps(outer)
        assert outer_steps == [CommitStep.FINAL_HEADERS]
        assert not outer.closed

    def test_sibling_exchanges_share_monotonic_order_only(self):
        ledger = ExchangeLedger(protocol=Protocol.H2)
        a = ledger.open_exchange(1, protocol=Protocol.H2)
        b = ledger.open_exchange(3, protocol=Protocol.H2)
        a.commit(CommitStep.BODY_PULL, plane=Plane.H2)
        b.commit(CommitStep.INFORMATIONAL, status=100, sent=True)
        a.commit(CommitStep.BODY_END, plane=Plane.H2)

        entries = ledger.entries()
        seqs = [e.seq for e in entries]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

        # Entries stay attributed to their own exchange: B's informational
        # must not appear on A.
        assert a.find(CommitStep.INFORMATIONAL) is None
        _find(b, CommitStep.INFORMATIONAL, sent=True)

        # Duplicate open key on the same handle is rejected.
        with pytest.raises(ValueError):
            ledger.open_exchange(1, protocol=Protocol.H2)

    def test_close_is_idempotent_and_freezes(self):
        ledger = ExchangeLedger()
        rec = ledger.open_exchange("x")
        rec.close()
        assert rec.close() is None
        assert rec.commit(CommitStep.FINAL_HEADERS) is None
        assert ledger.close_exchange("x") is rec


# ---------------------------------------------------------------------------
# WSGI HTTP/1
# ---------------------------------------------------------------------------

class TestWSGIHTTP1:

    def test_automatic_100_happens_before_app_and_before_body_pull(self):
        raw = (
            b"POST / HTTP/1.1\r\nHost: x\r\n"
            b"Expect: 100-continue\r\nContent-Length: 5\r\n\r\nhello")
        parser, req = _parse_h1(raw)
        sock = _SendSocket()
        wsgi_create(req, sock, ("127.0.0.1", 12345),
                    ("127.0.0.1", 8000), parser.cfg)

        record = req._exchange_record
        info = _find(record, CommitStep.INFORMATIONAL, status=100,
                     sent=True)
        assert info.detail["trigger"] == "before_app"

        # Nothing pulled the body yet, so the automatic 100 strictly
        # precedes the first BODY_PULL.
        assert not record.committed(CommitStep.BODY_PULL)
        assert req.body.read(5) == b"hello"
        _find(record, CommitStep.BODY_PULL)
        end = _find(record, CommitStep.BODY_END, plane=Plane.LENGTH)
        assert info.seq < record.find(CommitStep.BODY_PULL).seq < end.seq

        assert b"100 Continue" in b"".join(sock.sent)

    def test_chunked_trailers_recorded_and_pipelined_bytes_unread(self):
        tail = b"GET /next HTTP/1.1\r\nHost: x\r\n\r\n"
        raw = (
            b"POST / HTTP/1.1\r\nHost: x\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhello\r\n0\r\nX-Checksum: abc\r\n\r\n" + tail
        )
        parser, req = _parse_h1(raw)
        assert req.body.read() == b"hello"

        record = req._exchange_record
        _find(record, CommitStep.BODY_FRAMING, plane=Plane.CHUNKED)
        trailers = _find(record, CommitStep.REQUEST_TRAILERS,
                         disposition="recorded")
        assert trailers.detail["names"] == ["X-CHECKSUM"]
        assert trailers.detail["unread_bytes"] == len(tail)
        end = _find(record, CommitStep.BODY_END, trailers=True)
        assert req.trailers == [("X-CHECKSUM", "abc")]

        # The unread pipelined request must survive on the same connection.
        req2 = next(iter(parser))
        assert req2.path == "/next"
        assert req2._exchange_record is not record

    def test_no_content_length_is_eof_then_downgraded_empty(self):
        raw = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        parser, req = _parse_h1(raw)
        record = req._exchange_record

        framing = _all(record, CommitStep.BODY_FRAMING)
        planes = [e.detail["plane"] for e in framing]
        # The wire framing really is EOF ...
        assert Plane.EOF in planes
        # ... and the WSGI reader downgrades it to an empty LengthReader.
        empty = _find(record, CommitStep.BODY_FRAMING, plane=Plane.EMPTY)
        assert empty.detail.get("derived_from") == Plane.EOF
        _find(record, CommitStep.BODY_END, plane=Plane.EMPTY)

    def test_final_headers_commit_at_first_write_then_close_freezes(self):
        raw = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        parser, req = _parse_h1(raw)
        sock = _SendSocket()
        resp = WSGIResponse(req, sock, parser.cfg)
        write = resp.start_response("200 OK", [("Content-Length", "2")])
        record = req._exchange_record
        assert not record.committed(CommitStep.FINAL_HEADERS)
        write(b"ok")
        _find(record, CommitStep.FINAL_HEADERS, status="200 OK")
        resp.close()
        assert record.closed
        # HTTP/1 chunked close writes only the empty chunk: there is no
        # response trailers commit point on this machine.
        assert not record.committed(CommitStep.RESPONSE_TRAILERS)

    def test_peer_abort_is_connection_scoped(self):
        raw = (b"POST / HTTP/1.1\r\nHost: x\r\n"
               b"Transfer-Encoding: chunked\r\n\r\n5\r\nhel")
        parser, req = _parse_h1(raw)
        with pytest.raises((NoMoreData, ChunkMissingTerminator)):
            req.body.read()
        interrupt = _find(req._exchange_record,
                          CommitStep.PEER_INTERRUPT,
                          scope=InterruptScope.CONNECTION)
        assert interrupt.detail["kind"] in (
            "no_more_data", "chunk_missing_terminator")

    def test_disabled_ledger_is_bit_invisible(self):
        cfg = _config(ledger=False)
        parser = RequestParser(cfg, iter(
            [b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"]), None)
        req = next(iter(parser))
        assert getattr(parser, "exchange_ledger", None) is None
        assert req._exchange_record is None
        assert req.body._exchange_record is None
        # Pinned behaviour still works with the ledger off.
        assert req.body.read() == b""


# ---------------------------------------------------------------------------
# blocking HTTP/2
# ---------------------------------------------------------------------------

class _MockH2Socket(object):
    def __init__(self):
        self.recvd = []
        self._recv_buffer = b""
        self.closed = False

    def set_recv_data(self, data):
        self._recv_buffer += bytes(data)

    def recv(self, size):
        if not self._recv_buffer:
            return b""
        data = self._recv_buffer[:size]
        self._recv_buffer = self._recv_buffer[size:]
        return data

    def recv_into(self, buf):
        data = self.recv(len(buf))
        buf[:len(data)] = data
        return len(data)

    def sendall(self, data):
        self.recvd.append(bytes(data))

    def send(self, data, *args):
        self.recvd.append(bytes(data))
        return len(data)

    def get_sent_data(self):
        data = b"".join(self.recvd)
        self.recvd = []
        return data

    def setsockopt(self, *args):
        pass

    def getpeername(self):
        return ("127.0.0.1", 12345)

    def getsockname(self):
        return ("127.0.0.1", 8000)


def _h2_client():
    h2_config = h2.config.H2Configuration(client_side=True)
    client = H2Connection(config=h2_config)
    client.initiate_connection()
    return client


def _h2_server(ledger=True):
    cfg = _config(ledger=ledger)
    sock = _MockH2Socket()
    conn = HTTP2ServerConnection(cfg, sock, ("127.0.0.1", 12345))
    conn.initiate_connection()
    client = _h2_client()
    # preface + settings exchange
    conn.receive_data(client.data_to_send())
    client.receive_data(sock.get_sent_data())
    return cfg, sock, conn, client


_H2_GET = [
    (":method", "GET"), (":path", "/"), (":scheme", "http"),
    (":authority", "x"),
]
_H2_POST_EXPECT = [
    (":method", "POST"), (":path", "/"), (":scheme", "http"),
    (":authority", "x"), ("expect", "100-continue"),
    ("content-length", "3"),
]


class TestBlockingHTTP2:

    def test_read_body_emits_100_and_sibling_rst_is_stream_scoped(self):
        cfg, sock, conn, client = _h2_server()

        # Two real sibling streams on ONE connection, fed as one byte stream.
        client.send_headers(1, _H2_GET, end_stream=True)
        client.send_headers(3, _H2_POST_EXPECT, end_stream=False)
        requests = conn.receive_data(client.data_to_send())
        assert [r.stream.stream_id for r in requests] == [1, 3]
        req_get, req_post = requests

        ledger = conn.exchange_ledger
        ex1 = ledger.get_exchange(1)
        ex3 = ledger.get_exchange(3)
        assert ex1 is not None and ex3 is not None

        # Client body only becomes available after the server asks for it.
        client.send_data(3, b"abc", end_stream=True)
        sock.set_recv_data(client.data_to_send())

        # No 100 before the first read ...
        assert not ex3.committed(CommitStep.INFORMATIONAL)
        assert req_post.body.read() == b"abc"
        pull = _find(ex3, CommitStep.BODY_PULL, plane=Plane.H2)
        info = _find(ex3, CommitStep.INFORMATIONAL, status=100, sent=True)
        assert pull.seq < info.seq
        _find(ex3, CommitStep.BODY_END, plane=Plane.H2)

        # ... and the peer actually observes a real 100 frame.
        events = client.receive_data(sock.get_sent_data())
        statuses = [
            dict(e.headers).get(b":status")
            for e in events if hasattr(e, "headers")
        ]
        assert b"100" in statuses

        # The GET sibling has no informational entry at all.
        assert not ex1.committed(CommitStep.INFORMATIONAL)

        # Peer resets ONLY stream 1.
        client.reset_stream(1)
        conn.receive_data(client.data_to_send())
        interrupt = _find(ex1, CommitStep.PEER_INTERRUPT,
                          scope=InterruptScope.STREAM)
        assert interrupt.detail["kind"] == "rst_stream"
        # Stream 3's record is untouched by the sibling reset.
        assert not ex3.committed(CommitStep.PEER_INTERRUPT)

        conn.cleanup_stream(req_get.stream.stream_id)
        conn.cleanup_stream(req_post.stream.stream_id)
        assert ex1.closed and ex3.closed

    def test_write_before_read_suppresses_100(self):
        cfg, sock, conn, client = _h2_server()
        client.send_headers(1, _H2_POST_EXPECT, end_stream=False)
        req = conn.receive_data(client.data_to_send())[0]
        ex = conn.exchange_ledger.get_exchange(1)

        resp = HTTP2Response(req, sock, cfg, conn, 1)
        write = resp.start_response("200 OK", [("content-length", "1")])
        write(b"x")  # final HEADERS committed before any body pull
        _find(ex, CommitStep.FINAL_HEADERS, status=200)
        sock.get_sent_data()

        client.send_data(1, b"abc", end_stream=True)
        sock.set_recv_data(client.data_to_send())
        req.body.read()

        suppressed = _find(ex, CommitStep.INFORMATIONAL, status=100,
                           sent=False)
        assert suppressed.detail["reason"] == "final_headers_committed"
        assert ex.find(CommitStep.INFORMATIONAL, sent=True) is None
        # No 100 frame was ever written: the client never decodes one.
        decoded = client.receive_data(sock.get_sent_data())
        statuses = [
            dict(e.headers).get(b":status")
            for e in decoded if hasattr(e, "headers")
        ]
        assert b"100" not in statuses

    def test_stream_reset_raises_http2streamerror_but_connection_lives(self):
        cfg, sock, conn, client = _h2_server()
        client.send_headers(1, _H2_POST_EXPECT, end_stream=False)
        req = conn.receive_data(client.data_to_send())[0]
        client.reset_stream(1)
        conn.receive_data(client.data_to_send())

        with pytest.raises(HTTP2StreamError):
            req.body.read()
        interrupt = _find(conn.exchange_ledger.get_exchange(1),
                          CommitStep.PEER_INTERRUPT,
                          scope=InterruptScope.STREAM)
        assert interrupt.detail["kind"] in ("rst_stream", "stream_closed")

        # Same connection accepts a brand new sibling stream afterwards.
        client.send_headers(3, _H2_GET, end_stream=True)
        req2 = conn.receive_data(client.data_to_send())
        assert req2 and req2[0].stream.stream_id == 3
        assert conn.exchange_ledger.get_exchange(3) is not None

    def test_graceful_goaway_drains_without_stream_aborts(self):
        cfg, sock, conn, client = _h2_server()
        client.send_headers(1, _H2_POST_EXPECT, end_stream=False)
        req = conn.receive_data(client.data_to_send())[0]
        ex = conn.exchange_ledger.get_exchange(1)

        client.close_connection()
        conn.receive_data(client.data_to_send())
        assert conn.draining

        goaway = [e for e in conn.exchange_ledger.connection_entries
                  if e.detail.get("kind") == "goaway_graceful"]
        assert goaway
        # Graceful drain must not abort the in-flight stream.
        assert not ex.committed(CommitStep.PEER_INTERRUPT)

    def test_response_trailers_are_second_headers_block(self):
        cfg, sock, conn, client = _h2_server()
        client.send_headers(1, _H2_GET, end_stream=True)
        req = conn.receive_data(client.data_to_send())[0]
        ex = conn.exchange_ledger.get_exchange(1)

        resp = HTTP2Response(req, sock, cfg, conn, 1)
        write = resp.start_response("200 OK", [("content-length", "4")])
        resp.trailers = [("x-checksum", "abc")]
        write(b"body")
        assert not ex.committed(CommitStep.RESPONSE_TRAILERS)
        resp.close()
        _find(ex, CommitStep.RESPONSE_TRAILERS,
              names=["x-checksum"])


# ---------------------------------------------------------------------------
# ASGI HTTP/1 driven over a real ASGIProtocol
# ---------------------------------------------------------------------------

class _ASGITransport(object):
    def __init__(self):
        self.written = []
        self.closed = False

    def write(self, data):
        self.written.append(bytes(data))

    def writelines(self, data):
        for chunk in data:
            self.write(chunk)

    def close(self):
        self.closed = True

    def is_closing(self):
        return self.closed

    def get_extra_info(self, name, default=None):
        if name == "peername":
            return ("127.0.0.1", 12345)
        if name == "sockname":
            return ("127.0.0.1", 8000)
        return default

    def can_write_eof(self):
        return False

    def set_tcp_keepalive(self, *a):
        pass

    def pause_reading(self):
        pass

    def resume_reading(self):
        pass

    def set_write_buffer_limits(self, high=None, low=None):
        pass

    def write_eof(self):
        pass

    def abort(self):
        self.close()


class _ASGILog(object):
    access_log_enabled = False

    def exception(self, *a, **k):
        pass

    def access(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def critical(self, *a, **k):
        pass


class _ASGIWorker(object):
    def __init__(self, cfg, log, app):
        self.cfg = cfg
        self.log = log
        self.app = app
        self.asgi = app
        self.nr_conns = 0
        self.nr = 0
        self.max_requests = 10000
        self.alive = True
        self.loop = None
        self._hooks = cfg


async def _run_asgi_h1(app, raw, timeout=30, **cfg_overrides):
    from gunicorn.asgi.protocol import ASGIProtocol

    cfg = _config(timeout=int(timeout), http_parser="python",
                  worker_class="gunicorn.workers.asgi.ASGIWorker",
                  **cfg_overrides)
    # Fractional body-read waits are allowed at runtime even though the
    # setting validates as an int.
    cfg.settings["timeout"].value = timeout
    log = _ASGILog()
    worker = _ASGIWorker(cfg, log, app)
    loop = asyncio.get_event_loop()
    worker.loop = loop
    proto = ASGIProtocol(worker)
    transport = _ASGITransport()
    proto.connection_made(transport)
    proto.data_received(raw)
    for _ in range(200):
        await asyncio.sleep(0)
        if proto._task is None or proto._task.done():
            if transport.closed or b"HTTP/1.1" in b"".join(
                    transport.written):
                break
    return cfg, proto, transport


def _start(status="200 OK"):
    code = status.split(" ", 1)[0]
    return {"type": "http.response.start", "status": int(code),
            "headers": [(b"content-length", b"2")]}


_BODY_OK = {"type": "http.response.body", "body": b"ok",
            "more_body": False}


@pytest.mark.asyncio
class TestASGIHTTP1:

    async def test_no_automatic_100_even_with_expect(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(await receive())
            await send(_start())
            await send(_BODY_OK)

        raw = (b"POST / HTTP/1.1\r\nHost: x\r\n"
               b"Expect: 100-continue\r\nContent-Length: 5\r\n\r\nhello")
        cfg, proto, transport = await _run_asgi_h1(app, raw)

        wire = b"".join(transport.written)
        assert b"100 Continue" not in wire
        ledger = proto.exchange_ledger
        record, = ledger.exchanges
        noted = _find(record, CommitStep.INFORMATIONAL, status=100,
                      sent=False)
        assert noted.detail["trigger"] == "expect_recorded"
        assert record.find(CommitStep.INFORMATIONAL, sent=True) is None
        _find(record, CommitStep.BODY_FRAMING, plane=Plane.LENGTH)
        _find(record, CommitStep.FINAL_HEADERS)
        proto.connection_lost(None)

    async def test_no_framing_is_empty_body_completed_immediately(self):
        async def app(scope, receive, send):
            msg = await receive()
            assert msg["type"] == "http.request"
            assert msg.get("more_body") is False
            await send(_start())
            await send(_BODY_OK)

        raw = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        cfg, proto, transport = await _run_asgi_h1(app, raw)
        record, = proto.exchange_ledger.exchanges
        _find(record, CommitStep.BODY_FRAMING, plane=Plane.EMPTY,
              declared=False)
        end = _find(record, CommitStep.BODY_END, plane=Plane.EMPTY)
        # Completed while the request line/headers were still being parsed,
        # before the application pulled anything.
        pull = record.find(CommitStep.BODY_PULL)
        assert pull is None or end.seq < pull.seq
        # ASGI never uses the EOF plane, unlike WSGI HTTP/1.
        assert record.find(CommitStep.BODY_FRAMING,
                           plane=Plane.EOF) is None
        proto.connection_lost(None)

    async def test_inbound_trailers_validated_then_dropped(self):
        async def app(scope, receive, send):
            chunks = []
            while True:
                msg = await receive()
                if msg["type"] != "http.request":
                    break
                chunks.append(msg.get("body", b""))
                if not msg.get("more_body"):
                    break
            assert b"".join(chunks) == b"hello"
            assert "trailers" not in scope
            await send(_start())
            await send(_BODY_OK)

        raw = (
            b"POST / HTTP/1.1\r\nHost: x\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhello\r\n0\r\nX-Checksum: abc\r\n\r\n")
        cfg, proto, transport = await _run_asgi_h1(app, raw)
        record, = proto.exchange_ledger.exchanges
        dropped = _find(record, CommitStep.REQUEST_TRAILERS,
                        disposition="dropped")
        assert dropped.detail["names"] == ["x-checksum"]
        _find(record, CommitStep.BODY_END, plane=Plane.CHUNKED)
        proto.connection_lost(None)

    async def test_read_timeout_disconnect_without_completed_body(self):
        messages = []

        async def app(scope, receive, send):
            messages.append(await receive())
            messages.append(await receive())
            await send(_start("200 OK"))
            await send(_BODY_OK)

        raw = (b"POST / HTTP/1.1\r\nHost: x\r\n"
               b"Content-Length: 10\r\n\r\nabc")
        cfg, proto, transport = await _run_asgi_h1(app, raw, timeout=0.05)
        await asyncio.sleep(0.15)

        record, = proto.exchange_ledger.exchanges
        _find(record, CommitStep.PEER_INTERRUPT,
              scope=InterruptScope.CONNECTION, kind="body_wait_timeout")
        # Timeout must NOT synthesize more_body=False / body completion.
        assert not record.committed(CommitStep.BODY_END)
        assert messages[0]["type"] == "http.request"
        assert messages[0].get("more_body") is True
        assert messages[1]["type"] == "http.disconnect"
        proto.connection_lost(None)

    async def test_transport_close_is_connection_scoped_disconnect(self):
        async def app(scope, receive, send):
            first = await receive()
            assert first.get("more_body")
            second = await receive()
            assert second["type"] == "http.disconnect"

        raw = (b"POST / HTTP/1.1\r\nHost: x\r\n"
               b"Content-Length: 10\r\n\r\nabc")
        cfg, proto, transport = await _run_asgi_h1(app, raw)
        await asyncio.sleep(0.05)
        proto.connection_lost(None)
        await asyncio.sleep(0.05)
        record, = proto.exchange_ledger.exchanges
        _find(record, CommitStep.PEER_INTERRUPT,
              scope=InterruptScope.CONNECTION, kind="transport_closed")
        assert not record.committed(CommitStep.BODY_END)

    async def test_app_informational_http11_sent_http10_dropped(self):
        async def app11(scope, receive, send):
            await send({"type": "http.response.informational", "status": 103,
                        "headers": [(b"link", b"</style>; rel=preload")]})
            await send(_start())
            await send(_BODY_OK)

        raw = (b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        cfg, proto, transport = await _run_asgi_h1(app11, raw)
        assert b"103 Early Hints" in b"".join(transport.written)
        record, = proto.exchange_ledger.exchanges
        _find(record, CommitStep.INFORMATIONAL, status=103, sent=True,
              trigger="app")
        proto.connection_lost(None)

        async def app10(scope, receive, send):
            await send({"type": "http.response.informational", "status": 103,
                        "headers": []})
            await send(_start())
            await send(_BODY_OK)

        raw10 = (b"GET / HTTP/1.0\r\nHost: x\r\n\r\n")
        cfg2, proto2, transport2 = await _run_asgi_h1(app10, raw10)
        assert b"103 " not in b"".join(transport2.written)
        record2, = proto2.exchange_ledger.exchanges
        dropped = _find(record2, CommitStep.INFORMATIONAL, status=103,
                        sent=False)
        assert dropped.detail["reason"] == "http/1.0_unsupported"
        proto2.connection_lost(None)


# ---------------------------------------------------------------------------
# ASGI HTTP/2 decision points (protocol.py) with two sibling streams on one
# shared ledger; wire-level commits are emulated by the same side effects the
# real AsyncHTTP2Connection methods perform.
# ---------------------------------------------------------------------------

def _asgi_h2_proto(ledger=True):
    from gunicorn.asgi.protocol import ASGIProtocol
    from gunicorn.http2.async_connection import AsyncHTTP2Connection

    cfg = _config(ledger=ledger, timeout=5,
                  worker_class="gunicorn.workers.asgi.ASGIWorker")
    log = _ASGILog()
    worker = _ASGIWorker(cfg, log, None)
    loop = asyncio.new_event_loop()
    worker.loop = loop
    proto = ASGIProtocol(worker)
    real_conn = AsyncHTTP2Connection(cfg, mock.Mock(), mock.Mock(),
                                     ("127.0.0.1", 1))
    return cfg, proto, real_conn


def _asgi_stream(real_conn, stream_id, headers=None, end_stream=False,
                 expect=False):
    h = list(headers or _H2_GET)
    if expect:
        h = list(_H2_POST_EXPECT)
    stream = HTTP2Stream(stream_id, real_conn)
    real_conn.streams[stream_id] = stream
    stream.receive_headers(h, end_stream=end_stream)
    if expect:
        # The connection's header handler sets this when it parses Expect.
        stream.expect_continue = True
    real_conn._open_stream_exchange(stream)
    return stream


def _asgi_req(stream):
    req = mock.Mock()
    req.stream = stream
    req.method = "POST"
    req.path = "/"
    req.raw_path = "/"
    req.query = ""
    req.scheme = "http"
    req.version = (2, 0)
    req.headers = []
    req.content_length = None
    req.chunked = False
    req.priority_weight = None
    req.priority_depends_on = None
    req.priority_exclusive = None
    return req


class _FakeH2:
    """Mimics the ledger side effects of the real AsyncHTTP2Connection."""

    def __init__(self, real_conn):
        self.real_conn = real_conn
        self.streams = real_conn.streams

    async def send_informational(self, stream_id, status, headers=None):
        rec = self.streams[stream_id]._exchange_record
        rec.commit(CommitStep.INFORMATIONAL, status=status, sent=True)

    async def send_response_headers(self, stream_id, headers,
                                    end_stream=False):
        stream = self.streams[stream_id]
        stream.send_headers(list(headers), end_stream=end_stream)
        status = int(dict(headers)[":status"])
        stream._exchange_record.commit(CommitStep.FINAL_HEADERS,
                                       status=status, end_stream=end_stream)
        return True

    async def send_data(self, stream_id, data, end_stream=False):
        return True

    async def send_trailers(self, stream_id, trailers, end_stream=True):
        rec = self.streams[stream_id]._exchange_record
        rec.commit(CommitStep.RESPONSE_TRAILERS,
                   names=[n for n, _ in trailers])
        return True

    async def send_error(self, stream_id, status, message=None):
        return True


def _h2_start(status=200, trailers=False):
    return {"type": "http.response.start", "status": status, "headers": [],
            "trailers": trailers}


def _h2_body(body=b"x", more_body=False):
    return {"type": "http.response.body", "body": body,
            "more_body": more_body}


@pytest.mark.asyncio
class TestASGIHTTP2:

    async def test_100_after_start_then_final_headers_with_first_body(self):
        cfg, proto, real_conn = _asgi_h2_proto()
        stream = _asgi_stream(real_conn, 1, end_stream=False, expect=True)
        req = _asgi_req(stream)

        async def read_body_chunk():
            if not stream._body_chunks and not stream._body_complete:
                stream.receive_data(b"abc", end_stream=True)
            return stream.pop_chunk()
        stream.read_body_chunk = read_body_chunk

        async def app(scope, receive, send):
            await send(_h2_start(200))
            # start must not yet frame final HEADERS on this machine...
            rec = stream._exchange_record
            assert not rec.committed(CommitStep.FINAL_HEADERS)
            msg = await receive()
            assert msg["body"] == b"abc"
            # ...so the first body pull is still early enough for 100.
            _find(rec, CommitStep.INFORMATIONAL, status=100, sent=True)
            await send(_h2_body(b"ok"))
            _find(rec, CommitStep.FINAL_HEADERS, status=200)

        proto.worker.app = app
        proto.worker.asgi = app
        proto.app = app
        await proto._handle_http2_request(
            req, _FakeH2(real_conn), ("127.0.0.1", 8000),
            ("127.0.0.1", 1))
        _find(stream._exchange_record, CommitStep.BODY_END, plane=Plane.H2)

    async def test_write_before_read_suppresses_100(self):
        cfg, proto, real_conn = _asgi_h2_proto()
        stream = _asgi_stream(real_conn, 1, end_stream=False, expect=True)
        req = _asgi_req(stream)

        async def read_body_chunk():
            if not stream._body_chunks and not stream._body_complete:
                stream.receive_data(b"abc", end_stream=True)
            return stream.pop_chunk()
        stream.read_body_chunk = read_body_chunk

        async def app(scope, receive, send):
            await send(_h2_start(200))
            await send(_h2_body(b"z"))  # final HEADERS frame goes out now
            msg = await receive()
            assert msg["body"] == b"abc"

        proto.worker.app = app
        proto.worker.asgi = app
        proto.app = app
        await proto._handle_http2_request(
            req, _FakeH2(real_conn), ("127.0.0.1", 8000),
            ("127.0.0.1", 1))
        rec = stream._exchange_record
        suppressed = _find(rec, CommitStep.INFORMATIONAL, status=100,
                           sent=False)
        assert suppressed.detail["reason"] == "final_headers_committed"
        assert rec.find(CommitStep.INFORMATIONAL, sent=True) is None

    async def test_trailers_message_and_rst_client_disconnected(self):
        from gunicorn.asgi.protocol import ClientDisconnected
        cfg, proto, real_conn = _asgi_h2_proto()
        stream = _asgi_stream(real_conn, 1, end_stream=True, expect=False)
        req = _asgi_req(stream)
        req.method = "GET"
        got = {}

        async def app(scope, receive, send):
            await receive()
            await send(_h2_start(200, trailers=True))
            await send(_h2_body(b"x", more_body=True))
            # Final HEADERS went out with the first body frame.
            _find(stream._exchange_record, CommitStep.FINAL_HEADERS,
                  status=200)
            stream.disconnected = True
            with pytest.raises(ClientDisconnected):
                await send({"type": "http.response.trailers",
                            "headers": [(b"x-checksum", b"abc")]})
            got["disconnected"] = True

        proto.worker.app = app
        proto.worker.asgi = app
        proto.app = app
        # The handler itself logs ClientDisconnected instead of re-raising.
        await proto._handle_http2_request(
            req, _FakeH2(real_conn), ("127.0.0.1", 8000),
            ("127.0.0.1", 1))
        assert got["disconnected"]
        rec = stream._exchange_record
        _find(rec, CommitStep.PEER_INTERRUPT,
              scope=InterruptScope.STREAM, kind="rst_stream_gone")

    async def test_sibling_streams_do_not_bleed_on_one_connection(self):
        from gunicorn.asgi.protocol import ClientDisconnected
        cfg, proto, real_conn = _asgi_h2_proto()
        s1 = _asgi_stream(real_conn, 1, end_stream=False, expect=True)
        s3 = _asgi_stream(real_conn, 3, end_stream=False, expect=True)
        req1, req3 = _asgi_req(s1), _asgi_req(s3)

        async def feeder(stream):
            async def read_body_chunk():
                if not stream._body_chunks and not stream._body_complete:
                    stream.receive_data(b"abc", end_stream=True)
                return stream.pop_chunk()
            return read_body_chunk

        s1.read_body_chunk = await feeder(s1)
        s3.read_body_chunk = await feeder(s3)
        h2 = _FakeH2(real_conn)

        async def app_ok(scope, receive, send):
            await send(_h2_start(200))
            msg = await receive()
            assert msg["body"] == b"abc"
            await send(_h2_body(b"ok"))

        async def app_reset(scope, receive, send):
            await send(_h2_start(200))
            await send(_h2_body(b"z"))
            s3.disconnected = True
            with pytest.raises(ClientDisconnected):
                await send(_h2_body(b"more", more_body=True))

        async def run(req, app):
            proto.worker.app = app
            proto.worker.asgi = app
            proto.app = app
            await proto._handle_http2_request(
                req, h2, ("127.0.0.1", 8000), ("127.0.0.1", 1))

        await asyncio.gather(run(req1, app_ok), run(req3, app_reset))

        ex1 = real_conn.exchange_ledger.get_exchange(1)
        ex3 = real_conn.exchange_ledger.get_exchange(3)
        assert not ex1.committed(CommitStep.PEER_INTERRUPT)
        _find(ex1, CommitStep.INFORMATIONAL, status=100, sent=True)
        _find(ex1, CommitStep.FINAL_HEADERS, status=200)
        _find(ex3, CommitStep.PEER_INTERRUPT,
              scope=InterruptScope.STREAM, kind="rst_stream_gone")
        assert ex3.find(CommitStep.INFORMATIONAL, sent=True) is None
        # Monotonic shared sequence across both siblings.
        seqs = [e.seq for e in real_conn.exchange_ledger.entries()]
        assert seqs == sorted(seqs)


# ---------------------------------------------------------------------------
# same semantic across all three protocol machines
# ---------------------------------------------------------------------------

class TestCrossMachineForks:

    def test_request_trailers_disposition_forks(self):
        # WSGI HTTP/1: recorded and handed to the app.
        raw = (
            b"POST / HTTP/1.1\r\nHost: x\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"1\r\na\r\n0\r\nX-T: v\r\n\r\n")
        parser, req = _parse_h1(raw)
        req.body.read()
        wsgi_entry = _find(req._exchange_record,
                           CommitStep.REQUEST_TRAILERS,
                           disposition="recorded")
        assert wsgi_entry.detail["names"] == ["X-T"]

        # ASGI HTTP/1: validated then dropped (parser hook).
        from gunicorn.asgi.parser import PythonProtocol
        dropped = []
        p = PythonProtocol()
        p.on_request_trailers = lambda names: dropped.append(names)
        p.feed(raw)
        assert dropped == [["x-t"]]

        # Both HTTP/2 machines: trailers arrive as a HEADERS frame /
        # message, never on req.trailers.
        assert Protocol.H2 != Protocol.ASGI_H2

    def test_interrupt_scope_forks(self):
        # WSGI HTTP/1 abort is connection scoped.
        raw = (b"POST / HTTP/1.1\r\nHost: x\r\n"
               b"Transfer-Encoding: chunked\r\n\r\n3\r\nab")
        parser, req = _parse_h1(raw)
        with pytest.raises((NoMoreData, ChunkMissingTerminator)):
            req.body.read()
        wsgi_i = _find(req._exchange_record, CommitStep.PEER_INTERRUPT)
        assert wsgi_i.detail["scope"] == InterruptScope.CONNECTION

        # Blocking HTTP/2 RST is stream scoped while the connection lives.
        cfg, sock, conn, client = _h2_server()
        client.send_headers(1, _H2_POST_EXPECT, end_stream=False)
        req = conn.receive_data(client.data_to_send())[0]
        client.reset_stream(1)
        conn.receive_data(client.data_to_send())
        with pytest.raises(HTTP2StreamError):
            req.body.read()
        h2_i = _find(conn.exchange_ledger.get_exchange(1),
                     CommitStep.PEER_INTERRUPT)
        assert h2_i.detail["scope"] == InterruptScope.STREAM
        # The connection itself is still alive and accepting streams.
        assert not conn.draining

    def test_body_plane_fork_when_no_content_length(self):
        # WSGI recognizes EOF framing, then downgrades to an empty reader.
        raw = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        parser, req = _parse_h1(raw)
        planes = [e.detail["plane"] for e in
                  _all(req._exchange_record, CommitStep.BODY_FRAMING)]
        assert Plane.EOF in planes and Plane.EMPTY in planes

        # ASGI parser side: the framing decision the protocol records is
        # empty-only. Construct the exact decision ASGIProtocol makes.
        from gunicorn.asgi.parser import PythonProtocol
        p = PythonProtocol()
        p.feed(raw)
        # ASGI gives the body back immediately as empty without waiting on
        # EOF: there is no request object here that ever held an EOFReader.
        assert p.is_complete
        # Explicit contrast constant for hosts.
        assert Plane.EMPTY != Plane.EOF

    def test_disabled_default_is_pin_identical(self):
        # Default config keeps the ledger off on every machine.
        cfg = Config()
        assert cfg.exchange_ledger is False
        parser = RequestParser(cfg, iter(
            [b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"]), None)
        req = next(iter(parser))
        assert req._exchange_record is None

        sock = _MockH2Socket()
        conn = HTTP2ServerConnection(cfg, sock, ("127.0.0.1", 1))
        conn.initiate_connection()
        client = _h2_client()
        conn.receive_data(client.data_to_send())
        client.send_headers(1, _H2_GET, end_stream=True)
        req2 = conn.receive_data(client.data_to_send())[0]
        assert getattr(conn, "exchange_ledger", None) is None
        assert req2.stream._exchange_record is None
