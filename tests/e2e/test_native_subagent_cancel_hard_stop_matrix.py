"""End-to-end regression test: native sub-agent cancellation hard-stops only Claude.

The bug: ``_cancel_subagent_task`` in tool_dispatch.py checks
``entry.wrapper_label == CLAUDE_NATIVE_WRAPPER_VALUE`` to decide whether to
POST ``stop_session`` (hard-stop) vs ``interrupt`` (best-effort signal).  This
means:

1. **Active entry, uniform-stop harness** — cursor/goose/kiro/kimi/hermes/qwen
   all have ``_UNIFORM_STOP`` handlers on the runner that fire on
   ``stop_session``, yet ``sys_cancel_task`` sends ``interrupt`` to them
   instead.  The resident process can keep running after the cancel returns.

2. **Evicted entry, non-claude-native** — ``_cancel_evicted_claude_native_subagent``
   rejects any session whose wrapper label is not ``claude-code-native-ui``,
   returning "no in-flight task" without sending a stop event even when the
   child session is owned by the caller and its process may be alive.

3. **Failed entry, non-claude-native** — ``can_stop_failed_claude`` only opens
   the cleanup gate for claude-native, so a failed cursor-native (or similar)
   entry returns cached status without attempting a stop, leaving the process
   orphaned.

All three are structural / runner-level bugs reproducible with a mock HTTP
transport; no real harness binary is required.

Usage::

    pytest tests/e2e/test_native_subagent_cancel_hard_stop_matrix.py -v
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from omnigent._wrapper_labels import (
    CURSOR_NATIVE_WRAPPER_VALUE,
    WRAPPER_LABEL_KEY,
)
from omnigent.runner import app as runner_app
from omnigent.runner.tool_dispatch import execute_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNIFORM_STOP_WRAPPERS = [
    CURSOR_NATIVE_WRAPPER_VALUE,
    "goose-native-ui",
    "kiro-native-ui",
    "kimi-native-ui",
    "hermes-native-ui",
    "qwen-native-ui",
]
"""Wrapper labels whose harnesses register a ``_UNIFORM_STOP`` handler.

A ``sys_cancel_task`` on any of these should route to ``stop_session``, not
``interrupt``, so the child runner can invoke its hard-stop bridge.
"""


class _LivePane:
    """Stand-in native pane that always answers alive.

    A ``failed`` work status does not say whether the resident harness
    process exited; this models the dangerous case — the entry went terminal
    but the pane is still running — which is exactly when a cancel must be
    able to hard-stop it.
    """

    async def is_alive(self) -> bool:
        return True


class _LivePaneRegistry:
    """Terminal registry stub that reports a live ``main`` pane for any child."""

    def get(self, conversation_id: str, terminal_name: str, session_key: str) -> _LivePane:
        return _LivePane()


def _make_cancel_server(
    child_id: str, events: list[dict[str, Any]], stop_marks_terminal: bool = False
) -> httpx.MockTransport:
    """Return a mock transport that records events sent to *child_id*.

    :param child_id: The child session id to intercept.
    :param events: Mutable list that receives every event body posted.
    :param stop_marks_terminal: When ``True``, the handler also calls
        ``runner_app.mark_subagent_work_terminal`` to simulate the child
        runner marking the entry cancelled on receipt of ``stop_session``.
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == f"/v1/sessions/{child_id}/events":
            body = json.loads(request.content)
            events.append(body)
            if stop_marks_terminal and body.get("type") == "stop_session":
                runner_app.mark_subagent_work_terminal(
                    child_id, status="cancelled", output="[System: sub-agent stopped]"
                )
            return httpx.Response(204)
        return httpx.Response(404, json={"error": str(request.url)})

    return httpx.MockTransport(_handler)


# ---------------------------------------------------------------------------
# Facet 1 — active entry: uniform-stop harnesses must receive stop_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_label", _UNIFORM_STOP_WRAPPERS)
async def test_cancel_active_uniform_stop_harness_sends_stop_session(
    wrapper_label: str,
) -> None:
    """``sys_cancel_task`` must POST ``stop_session`` for harnesses in ``_UNIFORM_STOP``.

    These harnesses register a runner-side hard-stop handler
    (``_UNIFORM_STOP`` in ``interrupt.py``), so the child runner *will*
    honour ``stop_session`` and kill the resident process.  Posting
    ``interrupt`` instead silently leaves the process running because the
    uniform-stop harnesses' ``stop_session`` path is the only one that calls
    ``kill_session``.

    **What fails on the buggy build:** the event sent is ``interrupt`` instead
    of ``stop_session``.  After the fix the event is ``stop_session`` and the
    result reflects confirmed cancellation.
    """
    parent_id = f"conv_parent_cancel_active_{wrapper_label}"
    child_id = f"conv_child_cancel_active_{wrapper_label}"
    runner_app.register_subagent_work(
        parent_session_id=parent_id,
        child_session_id=child_id,
        agent="native_impl",
        title="native-task",
        wrapper_label=wrapper_label,
    )
    events: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(
            transport=_make_cancel_server(child_id, events, stop_marks_terminal=True),
            base_url="http://server",
        ) as server_client:
            result_raw = await execute_tool(
                tool_name="sys_cancel_task",
                arguments=json.dumps({"task_id": child_id}),
                server_client=server_client,
                conversation_id=parent_id,
                session_async_tasks={},
            )
    finally:
        runner_app.unregister_subagent_work(child_id)

    result = json.loads(result_raw)

    # The event posted to the child runner must be ``stop_session``; any other
    # value means the harness's hard-stop bridge was bypassed.
    assert len(events) == 1, f"Expected exactly 1 event for {wrapper_label!r}; got {events}"
    assert events[0]["type"] == "stop_session", (
        f"Expected stop_session for uniform-stop harness {wrapper_label!r}; "
        f"got {events[0]['type']!r}.  "
        "The cancel is routing through 'interrupt' (bug) instead of "
        "'stop_session' (fix), so the resident native process keeps running."
    )

    # After the stop marks the entry terminal the result should reflect
    # confirmed cancellation.
    assert result.get("cancelled") is True, (
        f"Expected confirmed cancellation for {wrapper_label!r}; got {result}"
    )


# ---------------------------------------------------------------------------
# Facet 2 — evicted entry: owned stop-capable natives must still be stoppable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_label", _UNIFORM_STOP_WRAPPERS)
async def test_cancel_evicted_uniform_stop_harness_sends_stop_session(
    wrapper_label: str,
) -> None:
    """``sys_cancel_task`` on an evicted non-claude-native entry must stop the session.

    When the work entry has been evicted from the in-process registry (e.g. due
    to a runner restart) but the child session still exists on the server,
    ``_cancel_evicted_claude_native_subagent`` is the fallback.  On the buggy
    build it rejects any label that is not ``claude-code-native-ui`` and returns
    "no in-flight task", leaving the resident process alive with no cleanup path.

    After the fix the function (or its generalised replacement) accepts any owned
    native session that supports a hard stop and posts ``stop_session``.

    **What fails on the buggy build:** the result is an error string containing
    "no in-flight task" and no event is sent.
    """
    parent_id = f"conv_parent_evicted_{wrapper_label}"
    child_id = f"conv_child_evicted_{wrapper_label}"
    # Do NOT register the work entry — simulate the evicted-entry path.

    events: list[dict[str, Any]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        """Serve a GET for the child session (it exists) and record events."""
        if request.method == "GET" and request.url.path == f"/v1/sessions/{child_id}":
            return httpx.Response(
                200,
                json={
                    "id": child_id,
                    "parent_session_id": parent_id,
                    "labels": {WRAPPER_LABEL_KEY: wrapper_label},
                },
            )
        if request.method == "POST" and request.url.path == f"/v1/sessions/{child_id}/events":
            body = json.loads(request.content)
            events.append(body)
            return httpx.Response(204)
        return httpx.Response(404, json={"error": str(request.url)})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        base_url="http://server",
    ) as server_client:
        result_raw = await execute_tool(
            tool_name="sys_cancel_task",
            arguments=json.dumps({"task_id": child_id}),
            server_client=server_client,
            conversation_id=parent_id,
            session_async_tasks={},
        )

    # On the buggy build: result_raw is an error string and events is empty.
    assert "no in-flight task" not in result_raw, (
        f"Evicted {wrapper_label!r} task returned 'no in-flight task' error; "
        "the cancellation fallback is filtering out non-claude-native wrapper labels "
        "so the resident process has no cleanup path.  "
        f"Full result: {result_raw}"
    )
    assert len(events) == 1, (
        f"Expected 1 stop event for evicted {wrapper_label!r} task; got {events}.  "
        f"Result: {result_raw}"
    )
    assert events[0]["type"] == "stop_session", (
        f"Expected stop_session event for evicted {wrapper_label!r} task; "
        f"got {events[0]['type']!r}"
    )


# ---------------------------------------------------------------------------
# Facet 3 — failed entry with a live pane: cancel must still attempt a stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_label", _UNIFORM_STOP_WRAPPERS)
async def test_cancel_failed_uniform_stop_harness_sends_stop_session(
    wrapper_label: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sys_cancel_task`` on a ``failed`` non-claude-native entry must attempt a stop.

    When a native harness's work entry is ``failed`` (the runner entry went
    terminal e.g. due to a connectivity error) but the resident process may
    still be alive, callers should be able to clean it up via
    ``sys_cancel_task``.  On the buggy build, ``can_stop_failed_claude`` is
    ``False`` for every non-claude-native harness, so the function short-circuits
    and returns cached ``{'status': 'failed'}`` without sending any event.

    After the fix, harnesses with a hard-stop capability open the same gate as
    claude-native so the stop event is dispatched.

    The child's pane is simulated alive: ``failed`` does not imply the
    process exited, and a correct build may verify pane liveness before
    hard-stopping (a stop posted to a dead pane can only fail).

    **What fails on the buggy build:** no event is sent and the result contains
    ``status: 'failed'`` with no stop attempt — even though the pane is alive.
    """
    monkeypatch.setattr("omnigent.runtime.get_terminal_registry", lambda: _LivePaneRegistry())
    parent_id = f"conv_parent_failed_{wrapper_label}"
    child_id = f"conv_child_failed_{wrapper_label}"
    runner_app.register_subagent_work(
        parent_session_id=parent_id,
        child_session_id=child_id,
        agent="native_impl",
        title="native-task",
        wrapper_label=wrapper_label,
    )
    # Transition to failed (process might still be alive).
    runner_app.mark_subagent_work_terminal(
        child_id,
        status="failed",
        output="[System: native process crashed]",
    )

    events: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(
            transport=_make_cancel_server(child_id, events),
            base_url="http://server",
        ) as server_client:
            result_raw = await execute_tool(
                tool_name="sys_cancel_task",
                arguments=json.dumps({"task_id": child_id}),
                server_client=server_client,
                conversation_id=parent_id,
                session_async_tasks={},
            )
    finally:
        runner_app.unregister_subagent_work(child_id)

    result = json.loads(result_raw)

    # On the buggy build: events == [] and result == {'cancelled': False, 'status': 'failed'}
    assert len(events) == 1, (
        f"Expected a stop event for failed {wrapper_label!r} entry; got {events}.  "
        f"Result: {result}.  "
        "The cancel is short-circuiting on 'failed' status (bug: can_stop_failed gate "
        "only open for claude-native) so no cleanup is attempted for the potentially "
        "live resident process."
    )
    assert events[0]["type"] == "stop_session", (
        f"Expected stop_session for failed {wrapper_label!r}; got {events[0]['type']!r}"
    )
