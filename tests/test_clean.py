import pytest

from utils import clean


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("HELLO World", "hello world"),
        ("check this http://t.co/abc123 now", "check this now"),
        ("see https://t.co/xyz", "see"),
        ("@united my flight is late", "my flight is late"),
        ("@American_Air thanks", "thanks"),
        ("Delayed 3 hours!!!", "delayed hours"),
        ("#badservice", "badservice"),
        ("I can't believe it", "i can't believe it"),
        ("  too    many   spaces  ", "too many spaces"),
        ("great \U0001F600 flight", "great flight"),
    ],
)
def test_clean_examples(raw, expected):
    assert clean(raw) == expected


@pytest.mark.parametrize("raw", ["!!! 123", "@united http://t.co/xyz", "", "   "])
def test_clean_returns_empty_string_when_nothing_is_left(raw):
    assert clean(raw) == ""


def test_clean_is_idempotent():
    text = "@united DELAYED 3 hours!!! http://t.co/abc  can't believe it"
    once = clean(text)
    assert clean(once) == once
