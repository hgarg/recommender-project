"""
Privacy helpers for pseudonymization and export integrity.
"""

import hashlib
import hmac
import json
import os
import secrets

import pandas as pd


PSEUDO_KEY_ENV = "REC_PSEUDO_KEY"
EXPORT_KEY_ENV = "REC_EXPORT_KEY"


def get_key(env_name, allow_ephemeral=True):
    key = os.environ.get(env_name)
    if key:
        return key.encode()

    if not allow_ephemeral:
        raise RuntimeError(f"{env_name} is not set")

    return secrets.token_bytes(32)


def pseudonymize(ids, key, prefix="u_"):
    """Return stable, keyed pseudonyms for user identifiers."""
    def _one(value):
        digest = hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest()
        return f"{prefix}{digest[:16]}"

    if isinstance(ids, pd.Series):
        return ids.map(_one)

    return [_one(value) for value in ids]


def file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(paths, out_path):
    manifest = {os.path.basename(path): file_sha256(path) for path in paths}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def verify_manifest(paths, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    return {
        os.path.basename(path): manifest.get(os.path.basename(path)) == file_sha256(path)
        for path in paths
    }


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    from cryptography.fernet import Fernet
    return Fernet(key).encrypt(data)


def decrypt_bytes(token: bytes, key: bytes) -> bytes:
    from cryptography.fernet import Fernet
    return Fernet(key).decrypt(token)


def new_fernet_key() -> bytes:
    from cryptography.fernet import Fernet
    return Fernet.generate_key()