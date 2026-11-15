"""Fitness functions for metadata-driven integration of *every* upstream family.

A new upstream FamilyDescription should require no fork-specific family-name
branches in the downloader, model locator, destination planner or Preferences UI.
"""
from types import SimpleNamespace

import pytest

from fizgig.families.description import ModelFile
from fizgig.families import registry
from fizgig.scripts import fetch_models
from fizgig.web.schema import get_schema


def test_all_upstream_training_families_participate_in_fork_model_catalog():
    schema = {family["key"]: family for family in get_schema()["families"]}
    for desc in registry.training_families():
        assert desc.key in schema, f"{desc.key}: missing from web schema"
        model_keys = set(schema[desc.key]["model_keys"])
        advertised = {f.pref_key for f in desc.model_files if f.repo and f.path}
        discovery = {
            w.pref_key for w in fetch_models.discovery_candidates((desc.key,))
        }
        bulk = {
            w.pref_key for w in fetch_models.model_candidates((desc.key,))
        }
        assert advertised <= discovery, f"{desc.key}: path-backed model absent from local discovery"
        assert advertised <= model_keys, f"{desc.key}: model absent from Preferences"
        expected_bulk = {f.pref_key for f in desc.model_files if f.repo and f.path and f.bulk_fetch}
        assert expected_bulk <= bulk, f"{desc.key}: bulk-fetch model missing"
        assert not (advertised - expected_bulk) & bulk, (
            f"{desc.key}: intentionally on-demand model leaked into bulk downloads"
        )


def test_future_family_catalog_is_derived_from_upstream_metadata(monkeypatch):
    future = SimpleNamespace(
        key="future_unseen_training_family",
        training_ready=True,
        model_files=(ModelFile(
            "future_unseen_dit", "Future DiT", True,
            "ExampleOrg/Next", "diffusion_models/future.safetensors", 2.5,
        ),),
    )
    monkeypatch.setitem(registry.FAMILIES, future.key, future)
    derived = fetch_models._described_families()
    assert future.key in derived
    weights = [w for w in derived[future.key] if getattr(w, "pref_key", "") == "future_unseen_dit"]
    assert len(weights) == 1
    assert weights[0].repo == "ExampleOrg/Next"

    # Negative control: models without a declared source are not inventively
    # treated as downloadable simply because the family is registered.
    orphan = SimpleNamespace(
        key="future_unavailable_family",
        training_ready=True,
        model_files=(ModelFile("orphan_dit", "Orphan DiT", True),),
    )
    monkeypatch.setitem(registry.FAMILIES, orphan.key, orphan)
    derived = fetch_models._described_families()
    assert orphan.key in derived
    assert all(getattr(w, "pref_key", "") != "orphan_dit" for w in derived[orphan.key])
