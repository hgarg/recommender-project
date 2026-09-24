import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from privacy import (pseudonymize, write_manifest, verify_manifest,
                     encrypt_bytes, decrypt_bytes, new_fernet_key)


def test_pseudonymize_is_stable_and_hides_raw_id():
    key = b"test-key"
    a = pseudonymize(["U00042"], key)[0]
    b = pseudonymize(["U00042"], key)[0]
    assert a == b                      # same key -> same token, joins still work
    assert "00042" not in a            # raw id doesn't leak into the token


def test_different_keys_give_different_tokens():
    assert pseudonymize(["U00042"], b"k1") != pseudonymize(["U00042"], b"k2")


def test_manifest_catches_edited_file(tmp_path):
    f = tmp_path / "events.csv"
    f.write_text("user_id,product_id,rating\nU1,P1,5\n")
    write_manifest([str(f)], str(tmp_path / "m.json"))
    assert verify_manifest([str(f)], str(tmp_path / "m.json"))["events.csv"]
    f.write_text("user_id,product_id,rating\nU1,P1,1\n")   # someone "fixes" a rating
    assert not verify_manifest([str(f)], str(tmp_path / "m.json"))["events.csv"]


def test_encrypt_roundtrip():
    key = new_fernet_key()
    token = encrypt_bytes(b"U1,P1,P2,P3", key)
    assert b"P1" not in token
    assert decrypt_bytes(token, key) == b"U1,P1,P2,P3"
