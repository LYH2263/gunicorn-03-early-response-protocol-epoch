#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.

"""Half-open exchange commit-step ledger.

A *half-open exchange* is one request/response exchange in progress on an
accepted connection.  Several exchanges may coexist on the same connection:
pipelined HTTP/1 exchanges (sequential) and -- most importantly -- multiple
HTTP/2 streams (siblings), and a host may declare nested exchanges on top of
the exchange currently being served (inner exchanges).

A *commit step* is an irreversible protocol fact that is observable by the
peer: a 1xx frame, an application pull of the request body, completion of the
request body on one of the framing planes, request/response trailers, the
final response headers and the peer-interruption exits (connection-wide or
stream-scoped).

The ledger is an append-only, per-connection book of those facts:

* exchanges never share storage: closing an inner or sibling exchange cannot
  rewrite anything an outer exchange already committed;
* entries are immutable and take their sequence number from one connection
  wide monotonic counter, so causal ordering across sibling exchanges on the
  same connection is decidable;
* there is deliberately **no** pop/rollback/undo operation -- bytes or frames
  already written to the wire cannot be retracted by leaving a ledger scope;
* ``commit()`` on an already closed exchange is a no-op (returns ``None``);
  late protocol events therefore never resurrect or rewrite a finished
  exchange.

The three protocol machines (WSGI HTTP/1, blocking HTTP/2 and ASGI) keep
their pinned behaviour differences; they only converge on *this* ledger
semantics.
"""

import itertools
import threading


class CommitStep(object):
    """The commit steps the ledger understands."""

    #: any 1xx response (implicit 100-continue or application informational)
    INFORMATIONAL = "informational"
    #: the application pulled the request body (wsgi.input read / receive)
    BODY_PULL = "body_pull"
    #: request body framing was selected at parse time
    BODY_FRAMING = "body_framing"
    #: request body was completed on one of the framing planes
    BODY_END = "body_end"
    #: request trailers were recorded, dropped or merely framed
    REQUEST_TRAILERS = "request_trailers"
    #: final (non-1xx) response headers hit the wire
    FINAL_HEADERS = "final_headers"
    #: response trailers hit the wire
    RESPONSE_TRAILERS = "response_trailers"
    #: the peer interrupted the exchange
    PEER_INTERRUPT = "peer_interrupt"


class Plane(object):
    """Request-body completion planes (the pin differences)."""

    CHUNKED = "chunked"
    LENGTH = "length"
    EOF = "eof"          # WSGI HTTP/1 with neither Content-Length nor chunked
    EMPTY = "empty"      # ASGI HTTP/1: complete at once, no EOF
    H2 = "h2"            # HTTP/2 DATA frames, END_STREAM


class Protocol(object):
    H1 = "http/1"
    H2 = "h2"
    ASGI_H1 = "asgi-h1"
    ASGI_H2 = "asgi-h2"


class InterruptScope(object):
    CONNECTION = "connection"
    STREAM = "stream"


class CommitEntry(object):
    """One immutable commit-step fact.

    ``seq`` comes from the ledger-wide monotonic counter; entries belonging
    to different exchanges (sibling streams) are therefore order-comparable.
    """

    __slots__ = ("seq", "step", "key", "protocol", "parent", "detail")

    def __init__(self, seq, step, key, protocol, parent, detail):
        self.seq = seq
        self.step = step
        self.key = key
        self.protocol = protocol
        self.parent = parent
        self.detail = detail

    def __repr__(self):
        return "<CommitEntry #%d %s key=%r detail=%r>" % (
            self.seq, self.step, self.key, dict(self.detail))

    def __eq__(self, other):
        if not isinstance(other, CommitEntry):
            return NotImplemented
        return self.seq == other.seq

    def __hash__(self):
        return hash(self.seq)


class ExchangeRecord(object):
    """The commit-step record of one half-open exchange.

    Append-only.  Closing the record freezes it: further ``commit`` calls are
    ignored (they never raise, so protocol code paths keep their pinned
    behaviour), and previously committed entries stay untouched.
    """

    def __init__(self, ledger, key, protocol, parent):
        self._ledger = ledger
        self.key = key
        self.protocol = protocol
        self.parent = parent
        self._entries = []
        self._closed = False

    # -- lifecycle -------------------------------------------------------

    @property
    def closed(self):
        return self._closed

    def close(self):
        """Freeze the record.  Idempotent.  Never removes entries."""
        self._closed = True

    # -- recording -------------------------------------------------------

    def commit(self, step, **detail):
        """Append one immutable entry.

        Returns the :class:`CommitEntry` or ``None`` when the exchange is
        already closed (late events must not write through a finished
        exchange).
        """
        if self._closed:
            return None
        return self._ledger._append(self, step, detail)

    # -- inspection ------------------------------------------------------

    @property
    def entries(self):
        return tuple(self._entries)

    def committed(self, step):
        return any(e.step == step for e in self._entries)

    def count(self, step):
        return sum(1 for e in self._entries if e.step == step)

    def latest(self, step):
        for entry in reversed(self._entries):
            if entry.step == step:
                return entry
        return None

    def find(self, step, **match):
        for entry in self._entries:
            if entry.step != step:
                continue
            if all(entry.detail.get(k) == v for k, v in match.items()):
                return entry
        return None

    def steps(self):
        return tuple(e.step for e in self._entries)

    def __repr__(self):
        return "<ExchangeRecord key=%r protocol=%s closed=%s steps=%r>" % (
            self.key, self.protocol, self._closed, self.steps())


class ExchangeLedger(object):
    """A per-connection book of half-open exchanges.

    One ledger instance belongs to one accepted connection; for HTTP/2 every
    stream of that connection opens its own exchange on the same ledger so
    that sibling-stream causality and isolation are observable together.
    """

    def __init__(self, protocol=None):
        self.protocol = protocol
        self._exchanges = {}
        self._connection_entries = []
        self._counter = itertools.count(1)
        self._lock = threading.RLock()

    # -- internal --------------------------------------------------------

    def _next_seq(self):
        return next(self._counter)

    def _append(self, record, step, detail):
        with self._lock:
            entry = CommitEntry(
                seq=self._next_seq(),
                step=step,
                key=record.key,
                protocol=record.protocol,
                parent=record.parent,
                detail=dict(detail))
            record._entries.append(entry)
            return entry

    # -- exchange lifecycle ---------------------------------------------

    def open_exchange(self, key, protocol=None, parent=None):
        """Declare a half-open exchange under ``key``.

        ``parent`` is the key of the outer exchange this one is nested in;
        sibling streams simply use their own distinct keys.  Re-opening an
        already open key is a programming error (closed keys stay in the
        book forever -- use a fresh key for a new half-open exchange).
        """
        with self._lock:
            existing = self._exchanges.get(key)
            if existing is not None and not existing.closed:
                raise ValueError("exchange %r is already open" % (key,))
            record = ExchangeRecord(
                self, key, protocol or self.protocol, parent)
            self._exchanges[key] = record
            return record

    def get_exchange(self, key):
        return self._exchanges.get(key)

    def close_exchange(self, key):
        record = self._exchanges.get(key)
        if record is not None:
            record.close()
        return record

    def commit_to(self, key, step, **detail):
        """Commit ``step`` to the exchange registered under ``key``.

        No-op (returns ``None``) when there is no such exchange or it is
        already closed.
        """
        record = self._exchanges.get(key)
        if record is None:
            return None
        return record.commit(step, **detail)

    # -- connection-scoped facts (GOAWAY/EOF/transport loss) -------------

    def commit_connection(self, step, **detail):
        with self._lock:
            entry = CommitEntry(
                seq=self._next_seq(),
                step=step,
                key=None,
                protocol=self.protocol,
                parent=None,
                detail=dict(detail))
            self._connection_entries.append(entry)
            return entry

    @property
    def connection_entries(self):
        return tuple(self._connection_entries)

    # -- inspection ------------------------------------------------------

    @property
    def exchanges(self):
        return tuple(self._exchanges.values())

    @property
    def open_exchanges(self):
        return tuple(e for e in self._exchanges.values() if not e.closed)

    @property
    def closed_exchanges(self):
        return tuple(e for e in self._exchanges.values() if e.closed)

    def entries(self):
        """All entries ordered by their connection-wide sequence number."""
        all_entries = []
        all_entries.extend(self._connection_entries)
        for record in self._exchanges.values():
            all_entries.extend(record._entries)
        all_entries.sort(key=lambda e: e.seq)
        return tuple(all_entries)


# ---------------------------------------------------------------------------
# Helpers used by the protocol machines.  They are all no-ops when no ledger
# record is attached to the object, so unenabled connections keep their exact
# pinned behaviour.
# ---------------------------------------------------------------------------

def exchange(obj):
    """Return the :class:`ExchangeRecord` attached to ``obj`` or ``None``."""
    if obj is None:
        return None
    return getattr(obj, "_exchange_record", None)


def record(obj, step, **detail):
    """Commit ``step`` on the exchange attached to ``obj`` (or no-op)."""
    ex = exchange(obj)
    if ex is None:
        return None
    return ex.commit(step, **detail)
