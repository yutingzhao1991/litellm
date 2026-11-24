"""未带 key 的请求不应在写 spend log 时因 _is_master_key 抛 TypeError。"""

from litellm.proxy.spend_tracking.spend_tracking_utils import _is_master_key
from litellm.proxy.utils import hash_token


def test_none_api_key_is_not_master_key():
    assert _is_master_key(None, "sk-master") is False


def test_none_master_key_is_not_master_key():
    assert _is_master_key("sk-1234", None) is False


def test_both_none_is_not_master_key():
    assert _is_master_key(None, None) is False


def test_string_master_key_matches():
    assert _is_master_key("sk-master", "sk-master") is True


def test_hashed_master_key_matches():
    assert _is_master_key(hash_token("sk-master"), "sk-master") is True


def test_other_key_does_not_match():
    assert _is_master_key("sk-other", "sk-master") is False
