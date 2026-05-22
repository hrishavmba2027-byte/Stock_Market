"""Shared Firestore client helper.

Centralizes credential resolution so every ingest module (fundamentals,
news, X) loads the same service account the same way.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Tuple


def init_firestore_client(
    project: Optional[str] = None,
    credentials_path: Optional[str] = None,
) -> Any:
    """Construct a Firestore client.

    Reuses the existing Google service account via ``GOOGLE_CREDENTIALS`` or
    ``GOOGLE_APPLICATION_CREDENTIALS``. ``FIRESTORE_PROJECT`` is optional —
    the project is read from the service-account JSON by default.
    """
    try:
        from google.cloud import firestore
    except ImportError as exc:
        raise RuntimeError(
            "Firestore write requires `google-cloud-firestore`. "
            "Install via requirements.txt."
        ) from exc

    credentials_path = (
        credentials_path
        or os.environ.get("GOOGLE_CREDENTIALS")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    project = (
        project
        or os.environ.get("FIRESTORE_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )

    if credentials_path:
        kwargs = {"project": project} if project else {}
        return firestore.Client.from_service_account_json(credentials_path, **kwargs)
    return firestore.Client(project=project) if project else firestore.Client()


def batch_write(
    client: Any,
    collection: str,
    docs: Iterable[Tuple[str, dict]],
    batch_size: int = 200,
) -> int:
    """Write ``(doc_id, payload)`` pairs to ``collection`` in batches.

    Returns the number of documents written. ``batch_size`` stays under
    Firestore's 500-op-per-batch hard limit by default.
    """
    coll = client.collection(collection)
    batch = client.batch()
    pending = 0
    written = 0

    for doc_id, payload in docs:
        if not doc_id:
            continue
        batch.set(coll.document(doc_id), payload)
        pending += 1
        written += 1
        if pending >= batch_size:
            batch.commit()
            batch = client.batch()
            pending = 0

    if pending:
        batch.commit()
    return written


def wipe_collection(client: Any, collection: str, batch_size: int = 200) -> int:
    """Delete every document in ``collection``. Returns the count deleted.

    Used by writers that want strict "latest-only" semantics — wipe-then-write
    guarantees stale docs from prior runs are gone, not just overwritten on a
    happy-path basis.
    """
    coll = client.collection(collection)
    batch = client.batch()
    pending = 0
    deleted = 0
    for doc in coll.stream():
        batch.delete(doc.reference)
        pending += 1
        deleted += 1
        if pending >= batch_size:
            batch.commit()
            batch = client.batch()
            pending = 0
    if pending:
        batch.commit()
    return deleted
