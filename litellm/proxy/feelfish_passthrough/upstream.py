"""
Upstream request plumbing for the FeelFish pass-through provider.

The gateway deliberately does not interpret the upstream protocol: it forwards
method / path / query / body and the upstream response as-is, only injecting
the configured credential and (optionally) an allow-listed set of client
headers.

Supported body shapes:

- ``application/json`` -> parsed dict, forwarded as JSON (minus stripped fields)
- ``multipart/form-data`` -> replayed as multipart (files + fields preserved)
- anything else (form-urlencoded, octet-stream, ...) -> raw bytes, verbatim
"""

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
import orjson
from fastapi import Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from .config import AuthStyle, PassthroughConfig, UpstreamAuth

# Characters that may stay literal inside a path segment (RFC 3986 pchar minus
# `%`, which must be re-encoded to survive the decode we receive from FastAPI).
PATH_SEGMENT_SAFE = ":@!$&'()*+,;=-._~"

# Never forwarded to the upstream, even when a client explicitly sends them.
BLOCKED_FORWARDED_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "host",
        "content-length",
        "connection",
        "proxy-authorization",
    }
)


@dataclass
class IncomingBody:
    """The parsed incoming body plus what billing rules may inspect."""

    content_type: str = ""
    json_body: Optional[Any] = None
    # Multipart parts in their original order, with repeated names preserved.
    # A part is (name, (filename|None, content, content_type|None)); a ``None``
    # filename marks a plain form field.
    multipart_parts: Optional[
        List[Tuple[str, Tuple[Optional[str], Any, Optional[str]]]]
    ] = None
    raw_body: Optional[bytes] = None
    billing_body: Optional[Any] = None

    @property
    def is_multipart(self) -> bool:
        return self.multipart_parts is not None


@dataclass
class UpstreamRequest:
    """The fully built request that will be sent upstream."""

    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: List[Tuple[str, str]] = field(default_factory=list)
    json_body: Optional[Any] = None
    files: Optional[
        List[Tuple[str, Tuple[Optional[str], Any, Optional[str]]]]
    ] = None
    raw_body: Optional[bytes] = None

    def build(
        self, client: httpx.AsyncClient, timeout: Optional[float] = None
    ) -> httpx.Request:
        request_kwargs: Dict[str, Any] = {"timeout": timeout} if timeout else {}
        if self.files is not None:
            return client.build_request(
                self.method,
                self.url,
                headers=self.headers,
                params=self.params,
                files=self.files,
                **request_kwargs,
            )
        if self.raw_body is not None:
            return client.build_request(
                self.method,
                self.url,
                headers=self.headers,
                params=self.params,
                content=self.raw_body,
                **request_kwargs,
            )
        return client.build_request(
            self.method,
            self.url,
            headers=self.headers,
            params=self.params,
            json=self.json_body,
            **request_kwargs,
        )


def build_target_url(api_base: str, subpath: str) -> str:
    """
    Join the deployment ``api_base`` with the request subpath.

    ``api_base`` is the full upstream URL and is always the prefix, so a client
    can never redirect the gateway to another host: subpath segments that could
    escape the prefix (``..``, ``//host``, ``scheme:``) are rejected.

    The subpath is appended to the URL **path**; a query string fixed in
    ``api_base`` (e.g. ``?api-version=2026``) is preserved as-is.
    """
    base = (api_base or "").strip()
    if not base:
        raise ValueError("api_base is required for feelfish deployments")

    raw_subpath = subpath or ""
    segments = [segment for segment in raw_subpath.strip("/").split("/") if segment]
    for segment in segments:
        if segment in (".", "..") or segment.startswith("//"):
            raise ValueError(f"invalid upstream path segment: {segment!r}")

    if not segments:
        return base

    # Some upstreams treat a trailing slash as a different resource.
    trailing_slash = raw_subpath.endswith("/")

    base_url = urlsplit(base)
    base_path = base_url.path.rstrip("/")
    # FastAPI hands over an already-decoded subpath: re-encode reserved characters
    # (`?`, `#`, `%`, ...) so they stay part of the path instead of becoming a
    # query string or fragment.
    encoded_segments = [quote(segment, safe=PATH_SEGMENT_SAFE) for segment in segments]
    target_path = f"{base_path}/{'/'.join(encoded_segments)}{'/' if trailing_slash else ''}"
    if not target_path.startswith(base_path + "/"):
        raise ValueError("upstream path escapes the configured api_base")

    return urlunsplit(
        (base_url.scheme, base_url.netloc, target_path, base_url.query, base_url.fragment)
    )


async def parse_incoming_body(
    request: Request, passthrough: PassthroughConfig
) -> IncomingBody:
    """Read and classify the incoming body, applying ``strip_body_fields``."""
    content_type = request.headers.get("content-type", "") or ""
    lowered = content_type.lower()
    strip_fields = [name for name in passthrough.strip_body_fields if name]

    if "form" in lowered:
        return await _parse_form_body(request, strip_fields, content_type, lowered)

    raw = await request.body()

    if "json" in lowered or (not content_type and raw.strip().startswith(b"{")):
        try:
            parsed = orjson.loads(raw) if raw else {}
        except orjson.JSONDecodeError:
            return IncomingBody(
                content_type=content_type, raw_body=raw, billing_body={}
            )
        if parsed is None:
            # `null` is a legitimate JSON body: forwarding `json=None` would send
            # an empty body instead of the four bytes the client sent.
            return IncomingBody(
                content_type=content_type or "application/json",
                raw_body=raw,
                billing_body={},
            )
        if isinstance(parsed, dict):
            for name in strip_fields:
                parsed.pop(name, None)
        return IncomingBody(
            content_type=content_type or "application/json",
            json_body=parsed,
            billing_body=parsed,
        )

    return IncomingBody(content_type=content_type, raw_body=raw, billing_body={})


async def _parse_form_body(
    request: Request,
    strip_fields: List[str],
    content_type: str,
    lowered_content_type: str,
) -> IncomingBody:
    """
    Read a form body (multipart or urlencoded), preserving repeated field names.

    ``request.form()`` returns the cached form when an earlier dependency (the
    auth chain) already parsed it, so this also works after `user_api_key_auth`
    has consumed the stream.
    """
    form = await request.form()

    if "multipart/form-data" in lowered_content_type:
        parts: List[Tuple[str, Tuple[Optional[str], Any, Optional[str]]]] = []
        billing_fields: Dict[str, Any] = {}
        for name, value in form.multi_items():
            if name in strip_fields:
                continue
            if isinstance(value, (UploadFile, StarletteUploadFile)):
                parts.append(
                    (name, (value.filename, await value.read(), value.content_type))
                )
                billing_fields[name] = value.filename
            else:
                parts.append((name, (None, str(value))))
                billing_fields[name] = str(value)
        return IncomingBody(
            content_type=content_type,
            multipart_parts=parts,
            billing_body=billing_fields,
        )

    # urlencoded: rebuild the body so repeated keys and their order survive
    items = [
        (name, str(value))
        for name, value in form.multi_items()
        if name not in strip_fields
    ]
    return IncomingBody(
        content_type=content_type or "application/x-www-form-urlencoded",
        raw_body=urlencode(items).encode(),
        billing_body=dict(items),
    )


def build_upstream_request(
    *,
    request: Request,
    api_base: str,
    api_key: Optional[str],
    auth: UpstreamAuth,
    passthrough: PassthroughConfig,
    extra_headers: Optional[Dict[str, Any]],
    subpath: str,
    body: IncomingBody,
) -> UpstreamRequest:
    """Assemble the upstream request (credentials + allow-listed headers + body)."""
    headers = _build_headers(
        request=request,
        auth=auth,
        api_key=api_key,
        extra_headers=extra_headers,
        passthrough=passthrough,
        body=body,
    )
    target_url = build_target_url(api_base, subpath)
    params = _build_query_params(
        request=request, auth=auth, api_key=api_key, target_url=target_url
    )

    return UpstreamRequest(
        method=request.method,
        url=target_url,
        headers=headers,
        params=params,
        json_body=body.json_body,
        files=body.multipart_parts,
        raw_body=body.raw_body,
    )


def _build_headers(
    *,
    request: Request,
    auth: UpstreamAuth,
    api_key: Optional[str],
    extra_headers: Optional[Dict[str, Any]],
    passthrough: PassthroughConfig,
    body: IncomingBody,
) -> Dict[str, str]:
    headers: Dict[str, str] = {}

    for name, value in (extra_headers or {}).items():
        if value is None:
            continue
        _set_header(headers, str(name), str(value))

    for name in passthrough.forward_headers:
        if not name or name.lower() in BLOCKED_FORWARDED_HEADERS:
            continue
        if name.lower().startswith("x-litellm-"):
            continue
        value = _get_header(request, name)
        if value is not None:
            _set_header(headers, name, value)

    # Injected last so a forwarded client header can never shadow or duplicate
    # the credential configured on the deployment.
    if auth.style == AuthStyle.BEARER and api_key:
        _set_header(headers, "Authorization", f"Bearer {api_key}")
    elif auth.style == AuthStyle.BASIC and api_key:
        encoded = base64.b64encode(f"{api_key}:".encode()).decode()
        _set_header(headers, "Authorization", f"Basic {encoded}")
    elif auth.style == AuthStyle.HEADER and api_key and auth.name:
        _set_header(headers, auth.name, api_key)

    if body.multipart_parts is not None:
        # Let httpx generate the multipart boundary.
        headers.pop("Content-Type", None)
        headers.pop("content-type", None)
    elif body.raw_body is not None:
        if body.content_type:
            headers["Content-Type"] = body.content_type
    else:
        # Preserve protocol-meaningful JSON media types (e.g. merge-patch+json).
        headers["Content-Type"] = body.content_type or "application/json"

    return headers


def _build_query_params(
    *,
    request: Request,
    auth: UpstreamAuth,
    api_key: Optional[str],
    target_url: str,
) -> List[Tuple[str, str]]:
    """
    Merge, in order: query params fixed in ``api_base``, the client query string
    (duplicates preserved), and the credential when ``auth_style: query:<Name>``.

    httpx replaces the URL query whenever ``params`` is passed, so the api_base
    query must be merged explicitly instead of being dropped.
    """
    params: List[Tuple[str, str]] = parse_qsl(
        urlsplit(target_url).query, keep_blank_values=True
    )
    params.extend((key, value) for key, value in request.query_params.multi_items())

    if auth.style == AuthStyle.QUERY and api_key and auth.name:
        # A client-supplied parameter with the same name must not shadow (or be
        # shadowed by, depending on how the upstream reads duplicates) the
        # configured credential.
        params = [(key, value) for key, value in params if key != auth.name]
        params.append((auth.name, api_key))
    return params


def _set_header(headers: Dict[str, str], name: str, value: str) -> None:
    """Set a header, replacing any same-named entry regardless of case."""
    lowered = name.lower()
    for existing in [key for key in headers if key.lower() == lowered]:
        headers.pop(existing, None)
    headers[name] = value


def _get_header(request: Request, name: str) -> Optional[str]:
    try:
        return request.headers.get(name)
    except Exception:  # pragma: no cover - defensive; Starlette headers are safe
        return None


async def send_upstream_request(
    client: httpx.AsyncClient,
    upstream: UpstreamRequest,
    timeout: Optional[float] = None,
) -> httpx.Response:
    """
    Send the request upstream and return a streaming response.

    Redirects are **not** followed: the shared client enables them by default and
    httpx only strips the standard ``Authorization`` header on cross-origin hops,
    so a custom credential header (``auth_style: header:<Name>``) would leak to
    the redirect target. The 3xx is passed through to the client instead.
    """
    built = upstream.build(client, timeout=timeout)
    return await client.send(built, stream=True, follow_redirects=False)


__all__ = [
    "BLOCKED_FORWARDED_HEADERS",
    "IncomingBody",
    "UpstreamRequest",
    "build_target_url",
    "build_upstream_request",
    "parse_incoming_body",
    "send_upstream_request",
]
