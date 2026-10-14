from __future__ import annotations

from scripts import sanitize_public_artifacts as sanitizer


def test_caller_supplied_private_tokens_are_redacted() -> None:
    sanitizer.configure_private_tokens(["ExampleAuthor", "research-host"])
    try:
        value, counts = sanitizer.sanitize_string(
            "ExampleAuthor ran this on research-host."
        )
    finally:
        sanitizer.configure_private_tokens([])

    assert value == "anonymous ran this on anonymous."
    assert counts["identity_or_host"] == 2


def test_private_tokens_are_not_hard_coded() -> None:
    sanitizer.configure_private_tokens([])
    value, counts = sanitizer.sanitize_string("ExampleAuthor")
    assert value == "ExampleAuthor"
    assert not counts
