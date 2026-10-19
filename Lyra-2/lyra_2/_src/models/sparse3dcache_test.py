# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity tests for the chunked ``Sparse3DCache.retrieve`` path.

Every case compares the chunked projection against the original single-shot
projection and asserts the returned ``(latent_index, frame_id)`` selection is
*identical*. The tests run on CPU and do not need a GPU, checkpoints or the
heavy model dependencies.

The class under test is imported normally when the environment has all of
Lyra-2's runtime dependencies. Otherwise the source of ``lyra2_model.py`` is
parsed and only ``Sparse3DCache`` is executed, so the tested logic is exactly
the shipped code even in a minimal environment.
"""

from __future__ import annotations

import ast
import math
import pathlib

import pytest

torch = pytest.importorskip("torch")

_MODEL_PATH = pathlib.Path(__file__).with_name("lyra2_model.py")


class _StubLog:
    def info(self, *args, **kwargs):  # noqa: D401 - test stub
        pass

    def warning(self, *args, **kwargs):
        pass


def _extract_sparse3dcache():
    """Return the real ``Sparse3DCache`` class.

    Tries the normal import first (upstream has all dependencies). Falls back to
    executing the class definition straight out of the source file so this test
    can run in a minimal CPU environment.
    """
    try:
        from lyra_2._src.models.lyra2_model import Sparse3DCache  # type: ignore

        return Sparse3DCache
    except Exception:
        pass

    tree = ast.parse(_MODEL_PATH.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Sparse3DCache"
    )
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)

    def _unavailable_unproject(*args, **kwargs):
        raise RuntimeError("unproject_points is unavailable in the minimal test loader")

    namespace = {
        "torch": torch,
        "Optional": __import__("typing").Optional,
        "unproject_points": _unavailable_unproject,
        "log": _StubLog(),
    }
    exec(compile(module, str(_MODEL_PATH), "exec"), namespace)
    return namespace["Sparse3DCache"]


Sparse3DCache = _extract_sparse3dcache()


def _make_cache(num_frames: int, batch: int, hp: int, wp: int, seed: int, downsample: int = 4):
    cache = Sparse3DCache(downsample=downsample, store_device="cpu", store_values=False)
    generator = torch.Generator().manual_seed(seed)
    for i in range(num_frames):
        pts = torch.empty(batch, hp, wp, 3)
        pts[..., 0].uniform_(-1.5, 1.5, generator=generator)
        pts[..., 1].uniform_(-1.5, 1.5, generator=generator)
        pts[..., 2].uniform_(0.5, 3.0, generator=generator)
        cache._world_points.append(pts)
        cache._latent_indices.append(i)
        cache._frame_ids.append(i)
    return cache


def _targets(batch: int, views: int, hw: tuple[int, int], downsample: int = 4):
    ht, wt = hw
    intrinsics = []
    poses = []
    for v in range(views):
        fx = 0.9 * wt
        fy = 0.9 * ht
        intrinsics.append(
            torch.tensor([[fx, 0.0, wt / 2.0], [0.0, fy, ht / 2.0], [0.0, 0.0, 1.0]])
        )
        angle = 0.05 * v
        rotation = torch.tensor(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        w2c = torch.eye(4)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = torch.tensor([0.2 * v, 0.1 * v, 3.0])
        poses.append(w2c)
    if views == 1:
        return (
            poses[0].unsqueeze(0).expand(batch, 4, 4).contiguous(),
            intrinsics[0].unsqueeze(0).expand(batch, 3, 3).contiguous(),
        )
    return (
        torch.stack(poses, dim=0).unsqueeze(0).expand(batch, views, 4, 4).contiguous(),
        torch.stack(intrinsics, dim=0).unsqueeze(0).expand(batch, views, 3, 3).contiguous(),
    )


def _retrieve(cache, w2c, k, hw, num_latents, chunk_size, **kwargs):
    return cache.retrieve(
        w2c,
        k,
        hw,
        num_latents=num_latents,
        skip_last_n=kwargs.get("skip_last_n", 0),
        random=kwargs.get("random", False),
        max_coverage=kwargs.get("max_coverage", False),
        depth_threshold=kwargs.get("depth_threshold", 0.1),
        chunk_size=chunk_size,
    )


def _assert_parity(cache, w2c, k, hw, num_latents=8, chunk=8, seed=0, **kwargs):
    torch.manual_seed(seed)
    reference = _retrieve(cache, w2c, k, hw, num_latents, None, **kwargs)
    torch.manual_seed(seed)
    chunked = _retrieve(cache, w2c, k, hw, num_latents, chunk, **kwargs)
    assert chunked == reference, f"chunk={chunk} kwargs={kwargs}: {chunked} != {reference}"
    return reference


def test_single_view_parity():
    cache = _make_cache(num_frames=40, batch=1, hp=16, wp=16, seed=1)
    w2c, k = _targets(batch=1, views=1, hw=(64, 64))
    for max_coverage in (False, True):
        for random in (False, True):
            for chunk in (1, 7, 32, 1000):
                _assert_parity(
                    cache,
                    w2c,
                    k,
                    (64, 64),
                    max_coverage=max_coverage,
                    random=random,
                    chunk=chunk,
                )


def test_skip_last_n_parity():
    cache = _make_cache(num_frames=32, batch=1, hp=12, wp=16, seed=2)
    w2c, k = _targets(batch=1, views=1, hw=(48, 64))
    for skip in (0, 3, 10):
        _assert_parity(cache, w2c, k, (48, 64), skip_last_n=skip, chunk=5, max_coverage=True)


def test_multiview_parity():
    cache = _make_cache(num_frames=24, batch=1, hp=16, wp=16, seed=3)
    w2c, k = _targets(batch=1, views=3, hw=(64, 64))
    for max_coverage in (False, True):
        _assert_parity(
            cache, w2c, k, (64, 64), max_coverage=max_coverage, random=False, chunk=5
        )


def test_random_seed_parity():
    cache = _make_cache(num_frames=48, batch=1, hp=16, wp=16, seed=4)
    w2c, k = _targets(batch=1, views=1, hw=(64, 64))
    for seed in (0, 1, 12345):
        result = _assert_parity(cache, w2c, k, (64, 64), random=True, chunk=6, seed=seed)
        assert len(result) > 0


def test_depth_threshold_parity():
    cache = _make_cache(num_frames=30, batch=1, hp=16, wp=16, seed=5)
    w2c, k = _targets(batch=1, views=1, hw=(64, 64))
    for threshold in (0.0, 0.1, 0.5):
        _assert_parity(
            cache, w2c, k, (64, 64), max_coverage=True, depth_threshold=threshold, chunk=4
        )


def test_chunk_size_fallbacks():
    cache = _make_cache(num_frames=20, batch=1, hp=16, wp=16, seed=6)
    w2c, k = _targets(batch=1, views=1, hw=(64, 64))
    torch.manual_seed(0)
    reference = _retrieve(cache, w2c, k, (64, 64), 6, None, max_coverage=True)
    for chunk in (None, 0, -1, 20, 21, 10_000):
        torch.manual_seed(0)
        got = _retrieve(cache, w2c, k, (64, 64), 6, chunk, max_coverage=True)
        assert got == reference, f"chunk={chunk}: {got} != {reference}"


def test_empty_and_no_valid():
    empty = _make_cache(num_frames=0, batch=1, hp=16, wp=16, seed=7)
    w2c, k = _targets(batch=1, views=1, hw=(64, 64))
    assert _retrieve(empty, w2c, k, (64, 64), 8, None, max_coverage=True) == []
    assert _retrieve(empty, w2c, k, (64, 64), 8, 4, max_coverage=True) == []

    behind = Sparse3DCache(downsample=4, store_device="cpu", store_values=False)
    pts = torch.zeros(1, 16, 16, 3)
    pts[..., 2] = -10.0  # behind the camera, z_cam < 0
    behind._world_points.append(pts)
    behind._latent_indices.append(0)
    behind._frame_ids.append(0)
    assert _retrieve(behind, w2c, k, (64, 64), 8, None, max_coverage=True) == []
    assert _retrieve(behind, w2c, k, (64, 64), 8, 4, max_coverage=True) == []


def test_projected_lin_keys():
    # Candidates whose source grid indices differ from their projected pixel
    # coordinates: using valid.nonzero() source indices instead of the rounded
    # projected coordinates silently changes depth ownership.
    cache = _make_cache(num_frames=12, batch=1, hp=16, wp=16, seed=8)
    w2c, k = _targets(batch=1, views=1, hw=(64, 64))
    result = _assert_parity(cache, w2c, k, (64, 64), max_coverage=True, chunk=3)
    assert len(result) > 0


def test_min_candidate_owner():
    # Two candidates projecting to exactly the same pixel at exactly the same
    # depth must be resolved by the minimum candidate id, as in the unchunked
    # scatter_reduce(amin) path.
    cache = Sparse3DCache(downsample=4, store_device="cpu", store_values=False)
    for frame_id in (0, 1):
        pts = torch.zeros(1, 4, 4, 3)
        pts[..., 2] = 1.0
        cache._world_points.append(pts)
        cache._latent_indices.append(frame_id)
        cache._frame_ids.append(frame_id)
    w2c = torch.eye(4).unsqueeze(0)
    w2c[:, 2, 3] = 3.0
    k = torch.tensor([[[64.0, 0.0, 8.0], [0.0, 64.0, 8.0], [0.0, 0.0, 1.0]]])
    torch.manual_seed(0)
    reference = _retrieve(cache, w2c, k, (16, 16), 1, None, max_coverage=False)
    torch.manual_seed(0)
    chunked = _retrieve(cache, w2c, k, (16, 16), 1, 1, max_coverage=False)
    assert chunked == reference
    assert reference == [(cache._latent_indices[0], cache._frame_ids[0])]
