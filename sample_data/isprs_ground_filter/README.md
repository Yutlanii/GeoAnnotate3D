# Sample data — ISPRS ground-filtering benchmark (15 sites)

15 real, hand-classified airborne LiDAR scenes from the classic **ISPRS
Filter Test** (Sithole & Vosselman, 2004) — the reference benchmark study
used to compare bare-earth extraction ("ground filtering") algorithms.
Together they total **384,955 points**, each labeled **ground (ASPRS code
2)** or **non-ground ("created, never classified", code 0)** — vegetation,
buildings, cars, and other above-ground objects, all lumped into that one
"non-ground" class as per the original benchmark's design.

| File | Points | Ground | Non-ground |
|---|---:|---:|---:|
| samp11-utm.laz | 38,010 | 21,786 (57%) | 16,224 |
| samp12-utm.laz | 52,119 | 26,691 (51%) | 25,428 |
| samp21-utm.laz | 12,960 | 10,085 (78%) | 2,875 |
| samp22-utm.laz | 32,706 | 22,504 (69%) | 10,202 |
| samp23-utm.laz | 25,095 | 13,223 (53%) | 11,872 |
| samp24-utm.laz | 7,492 | 5,434 (73%) | 2,058 |
| samp31-utm.laz | 28,862 | 15,556 (54%) | 13,306 |
| samp41-utm.laz | 11,231 | 5,602 (50%) | 5,629 |
| samp42-utm.laz | 42,470 | 12,443 (29%) | 30,027 |
| samp51-utm.laz | 17,845 | 13,950 (78%) | 3,895 |
| samp52-utm.laz | 22,474 | 20,112 (89%) | 2,362 |
| samp53-utm.laz | 34,378 | 32,989 (96%) | 1,389 |
| samp54-utm.laz | 8,608 | 3,983 (46%) | 4,625 |
| samp61-utm.laz | 35,060 | 33,854 (97%) | 1,206 |
| samp71-utm.laz | 15,645 | 13,875 (89%) | 1,770 |

Numbers after `samp` group scenes by original test site (1–7), each with
its own terrain challenge: steep slopes, low vegetation, bridges, disjoint
terrain features, and so on — the whole point of the original study was
that no single ground-filter algorithm handled every case well, which
makes this a good, small, real-world stress test for the app's own
**CSF (Cloth Simulation Filter)** ground-classification tool: load one of
these, run CSF, and compare its result against the ground truth already
in the file (color by "Clasificación" to see the original labels, then
by "Confianza"/your own CSF output to compare).

Use it for:
1. Trying/validating GeoAnnotate3D's CSF ground-detection tool against a
   real published benchmark instead of an unlabeled file.
2. A second, different kind of training set (binary ground/non-ground)
   to combine with `sample_data/autzen-classified.copc.laz`'s richer
   21-category scheme — export both into the **same** dataset folder to
   see the incremental/append-safe export feature in action (though note
   the two use different class schemas — see the caveat in the main
   `docs/user-guide.md` export section about keeping schemas consistent
   across clouds you combine into one dataset).

## Origin & license

- Original test data: Sithole, G. and Vosselman, G. (2004). "Experimental
  comparison of filter algorithms for bare-earth extraction from airborne
  laser scanning point clouds." *ISPRS Journal of Photogrammetry and
  Remote Sensing* 59(1-2), 85-101.
  <https://doi.org/10.1016/j.isprsjprs.2004.05.004>
- Reprojected to UTM zone 32U and converted to LAZ as part of the
  [PDAL/data](https://github.com/PDAL/data) repository, distributed under
  a **CC BY 4.0** license (attribution required, redistribution and reuse
  permitted).
- Source: <https://github.com/PDAL/data/tree/main/isprs>

These are the *reference* files (`samp*-utm.laz`) — the ones that carry
the actual ground/non-ground classification. The repository's `isprs`
folder also has `CSite*`/`FSite*` "test" files for the same eight sites;
those are the **unclassified** raw point clouds the original filters were
run against (verified: `classification` is all-zero in those), so they
weren't included here — they're for benchmarking a filter's *output*
against these labeled reference files, not for training/annotation
practice on their own.

If you redistribute these files elsewhere, keep this attribution.
