"""Proves the vectorized silence-point search in infer/modules/vc/pipeline.py
matches the original O(window * n) loop.

Run from repo root: .venv/bin/python tests/pipeline_equiv.py
"""

import numpy as np

WINDOW = 160
T_QUERY = 16000 * 6
T_CENTER = 16000 * 38


def original(audio, audio_pad, window, t_center, t_query):
    audio_sum = np.zeros_like(audio)
    for i in range(window):
        audio_sum += np.abs(audio_pad[i : i - window])
    opt_ts = []
    for t in range(t_center, audio.shape[0], t_center):
        opt_ts.append(
            t
            - t_query
            + np.where(
                audio_sum[t - t_query : t + t_query]
                == audio_sum[t - t_query : t + t_query].min()
            )[0][0]
        )
    return opt_ts


def vectorized(audio, audio_pad, window, t_center, t_query):
    # float64 cumsum: float32 accumulation loses ~1e-2 absolute precision over
    # 1M samples, enough to flip the argmin between near-equal silent valleys
    csum = np.zeros(audio_pad.shape[0] + 1, dtype=np.float64)
    np.cumsum(np.abs(audio_pad), dtype=np.float64, out=csum[1:])
    audio_sum = (csum[window:] - csum[:-window])[: audio.shape[0]]
    opt_ts = []
    for t in range(t_center, audio.shape[0], t_center):
        seg = audio_sum[t - t_query : t + t_query]
        opt_ts.append(t - t_query + int(np.argmin(seg)))
    return opt_ts


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(5):
        n = int(16000 * (40 + 80 * rng.random()))  # 40-120s of 16k audio
        audio = (rng.standard_normal(n) * 0.1).astype(np.float32)
        # carve silent valleys so the argmin targets are unambiguous
        for _ in range(12):
            c = int(rng.integers(T_QUERY, n - T_QUERY))
            audio[c - 400 : c + 400] *= 0.001
        audio_pad = np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect")
        a = original(audio, audio_pad, WINDOW, T_CENTER, T_QUERY)
        b = vectorized(audio, audio_pad, WINDOW, T_CENTER, T_QUERY)
        assert len(a) == len(b), (trial, a, b)
        # float accumulation order differs between the two methods, so allow
        # a tie-adjacent index as long as the windowed sums there are equal
        # to float32 precision; the split point is a silence heuristic.
        abs_pad = np.abs(audio_pad).astype(np.float64)
        csum = np.concatenate(([0.0], np.cumsum(abs_pad)))
        exact = csum[WINDOW:] - csum[:-WINDOW]
        for ia, ib in zip(a, b):
            assert ia == ib or np.isclose(exact[ia], exact[ib], rtol=1e-4), (
                trial,
                ia,
                ib,
                exact[ia],
                exact[ib],
            )
        print(
            "trial",
            trial,
            "ok:",
            len(a),
            "split points,",
            a == b and "identical" or "tie-equivalent",
        )
    print("PIPELINE EQUIV OK")
