# Concrete-Crack-Detection-Segmentation — Engineering Audit & Product Viability Assessment

**Scope of this document.** A code-level engineering audit of the repository as it stands, an
assessment of whether it can become a product people pay for, an analysis of who those people
would be and what they actually need, and a phased plan to get from here to there.

**Method.** Every engineering finding below was derived by reading the code in this repository and,
where possible, verified by execution. Market claims are grounded in the current commercial
landscape and in the published photogrammetry/crack-measurement literature; sources are cited
inline. Where something is a judgement call rather than a verified fact, it is labelled as such.

---

## 1. Executive summary

### 1.1 What this repository is

It is a competent **research artifact**: a PyTorch re-implementation of DeepCrack (a hierarchical
CNN for crack segmentation) wrapped in a thin FastAPI service and a Create-React-App front end,
plus an unrelated Jupyter notebook that fine-tunes ResNet-50 as a patch classifier on the METU
concrete-crack dataset. It demonstrates, on curated images, that a crack can be segmented and
annotated with a length, a width, and an orientation category.

### 1.2 What it is not

It is not a product, and — more importantly — it is not currently a *correct* measurement tool.
The audit found that the headline output, the crack width in real-world units, is computed by a
method that is wrong by construction, at an image resolution roughly **30–40× too coarse** to say
anything meaningful about the crack widths that engineering standards actually care about.

### 1.3 The single most important finding

The published literature on photogrammetric crack measurement converges on a requirement of
roughly **0.09–0.15 mm per pixel** ground sampling distance, with a crack needing to span
**10–30 pixels** for its width to be estimated reliably ([Information 2025, 16(6):448](https://www.mdpi.com/2078-2489/16/6/448)).
ACI 224R-01 sets allowable crack widths under service load between **0.10 mm and 0.41 mm**
depending on exposure, and AASHTO element inspection puts the narrow/moderate boundary at
**1.6 mm**.

This pipeline force-resizes every input to **256 × 256** (`util/inference_utils.py:47`). A
photograph covering one metre of wall therefore has a ground sampling distance of
**3.9 mm/pixel**. A 0.3 mm crack — precisely the durability threshold an engineer is looking for —
occupies **0.08 of a pixel**. The system will still confidently print a width in millimetres next
to it.

Everything else in this audit is fixable with ordinary engineering effort. This one is
architectural, and until it is addressed, *the product's core promise is not merely inaccurate, it
is unfalsifiable*: there is no evaluation anywhere in the repository that would have caught it.

### 1.4 Verdict on product probability

| Path | Probability of becoming genuinely valuable | Why |
|---|---|---|
| Head-on commercial SaaS vs. incumbents | Very low | Well-funded competitors already ship 3D digital twins, BIM integration and standards-compliant reporting |
| Vertical wedge (one asset class, one compliance regime, one geography) | Moderate | Narrow scope makes the accuracy problem tractable and the report format ownable |
| Internal tool / consultancy accelerator at an infrastructure engineering firm | **Highest** | Capture protocol can be mandated, ground truth is available, distribution is free, liability sits with the existing engineer |
| Credible open-source reference implementation | High for adoption, low for revenue | The honest, reproducible, well-evaluated crack-segmentation service does not currently exist in public |

The commit history contains a commit authored from an infrastructure-engineering domain address,
which suggests the author already sits inside the industry that would use this. That is a real and
underexploited asset — see §6.4.

### 1.5 The one-line answer to "what does it need"

It needs to stop being a demo that *shows* a crack and become a tool that *defends a number*:
reproducible to run, honest about when it cannot measure, validated against ground truth, and
wrapped in the report an engineer signs their name to.

---

## 2. Repository inventory

| Component | Files | Status |
|---|---|---|
| Inference service | `main.py` | FastAPI, single endpoint, ~73 lines |
| Model + post-processing (live) | `util/inference_utils.py`, `util/cv2_utils.py` | Imported by `main.py` |
| Model + post-processing (dead) | `inference_utils.py`, `cv2_utils.py` (repo root) | Duplicates, **diverged**, unreachable |
| Network definitions | `util/models/` | DeepCrack + pix2pix scaffolding + unused RoadNet |
| Weights | `pretrained_net_G.pth` | 59 MB, committed to git |
| Front end | `web-deepcrack/` | Create React App, default boilerplate README and tests |
| Notebook | `Crack-recognition.ipynb` | 3.6 MB, ResNet-50 classifier, unrelated to the service |
| Packaging / CI / tests / licence | — | **None** |

37 commits between 2020-12 and 2026-02, essentially all by one author. `.git` is 56 MB, almost
entirely the committed checkpoint.

### 2.1 Architecture as built

```
Browser (CRA, hardcoded http://127.0.0.1:8000)
   │  POST /predict/{width}-{height}/{unit}   multipart image
   ▼
FastAPI  main.py
   │  create_model()  ← rebuilds the network and reloads 59 MB from disk PER REQUEST
   ▼
DeepCrackNet  →  5 side maps + 1 fused map, all at 256×256
   │
   ▼
cv2 post-processing  →  Canny on the mask → contours → length / width / angle / category
   │
   ▼
7 base64 PNGs returned as a JSON array
```

There is no persistence, no job queue, no authentication, no configuration layer, and no
observability.

---

## 3. Engineering audit

Findings are ordered by severity. File and line references are to the code as audited.

### 3.1 P0 — The service cannot run on a current stack

**3.1.1 NumPy 2.x breaks image decoding outright.**
`util/inference_utils.py:40` calls `np.fromstring(bytesImg, np.uint8)`. The binary mode of
`np.fromstring` was removed in NumPy 2.0. Verified in this environment (NumPy 2.4.4):

```
ValueError: The binary mode of fromstring is removed, use frombuffer instead
```

Every single request fails. The fix is a one-word change to `np.frombuffer`, but nothing in the
repository would have detected it, which is the actual problem.

**3.1.2 A CPU-only deployment crashes on a bare assertion.**
`main.py:27` defaults `gpu_ids: List[int] = [0]`. That reaches
`util/models/networks.py:105`, `assert(torch.cuda.is_available())`. On any machine without CUDA —
which includes most cheap cloud instances, most laptops, and every CI runner — the service dies
with an `AssertionError` carrying no message. There is no CPU fallback and no device
auto-detection, despite `base_model.py:35` already containing the logic to do it.

**3.1.3 A vertical crack raises `ZeroDivisionError`.**
`util/cv2_utils.py:114`:

```python
slope = (P1[1] - P2[1]) / (P2[0] - P1[0])
```

When a crack's minimum-area rectangle is axis-aligned vertically, `P2[0] - P1[0]` is zero. The
duplicate file at the repository root has the guard (`... if (P2[0] - P1[0]) != 0 else 0`); the
file that is actually imported does not. **The fix exists in the repository and is in the dead
copy.** Vertical cracks are among the most common orientations in walls and columns, so this is
not a corner case.

**3.1.4 There is no dependency manifest.**
No `requirements.txt`, no `pyproject.toml`, no lockfile, no Dockerfile, no pinned versions.
The service silently requires `torch`, `torchvision`, `opencv-python`, `numpy`, `scipy`,
`imutils`, `pillow`, `fastapi`, `uvicorn` and `python-multipart`. The last of these is
load-bearing (FastAPI's `UploadFile` will not work without it) and is documented only as a bare
comment on `main.py:57` reading `#python-multipart`. A new user cannot install this project
without reverse-engineering it from import errors.

**3.1.5 The checkpoint path is CWD-relative.**
`create_model(opt, cp_path='pretrained_net_G.pth')` at `util/inference_utils.py:53`. The service
works only when launched from the repository root, and there is no way to point it at a different
model without editing source.

### 3.2 P0 — The measurements are not defensible

This is the section that matters. The application's stated value is that it "outputs the length,
width, category of the cracks." Each of those three outputs has a correctness problem.

**3.2.1 Width is computed by counting an entire image row.**
`util/cv2_utils.py:138-140`:

```python
mask = gray.copy()
mask[mask < 40] = 0
width = cv2.countNonZero(mask[cY][:])
```

This counts every non-zero pixel across the **full width of the image** on the row passing through
the contour's centroid — not the crack's local thickness. Consequences:

- A second crack anywhere on that row is added to the first crack's width.
- Surface texture, staining, or shadow that survives thresholding is added to the width.
- For a diagonal crack, a horizontal row-scan measures the hypotenuse of the crack's cross-section,
  not its perpendicular opening, inflating the result by `1/sin θ`. At 30° from horizontal that is
  a **2× overestimate**.
- Even for an ideal single vertical crack on a clean surface, it measures at one arbitrary row
  rather than reporting the maximum or a distribution along the crack — and maximum width is the
  quantity standards are written against.

The established approach is a **Euclidean distance transform from the mask boundary evaluated on
the crack skeleton** (giving a width profile along the entire crack), or orthogonal profile
extraction perpendicular to the local skeleton tangent. Neither is expensive; both are correct.

**3.2.2 Length is half a perimeter.**
`util/cv2_utils.py:132`: `length = cv2.arcLength(c, True) / 2. * pixelsPerMetricWidth`.
Perimeter-over-two approximates the length of a long thin ribbon and nothing else. It
overestimates for branching or highly tortuous cracks and is undefined in meaning for a crack
network. It also multiplies by the **width** scale factor only, ignoring `pixelsPerMetricHeight`,
so any non-square effective pixel produces a systematic error. Skeleton path length is the correct
measure.

**3.2.3 Only the first crack in an image is ever measured.**
`util/cv2_utils.py:171`: the `return orig` sits **inside** the `for c in cnts:` loop. The function
returns on the first contour with area ≥ 100 px. An image with five cracks reports one. Worse,
`orig = overlay_img.copy()` is also inside the loop, so even without the early return each
iteration would discard the previous iteration's annotations. There is no per-crack instance
identity anywhere in the pipeline.

**3.2.4 The "crack percentage" is not a probability, and the two copies disagree about it.**
`util/inference_utils.py:106`: `confidence = visuals['fused'].max()`, where `fused` has already been
rescaled from a sigmoid output to `[-1, 1]` by `util/models/deepcrack_model.py:87`. It is the single
most-activated pixel in the map — not a calibrated probability, and not an image-level score. Any
image containing one confident pixel, including a false positive on a shadow or a form-tie hole,
saturates it.

The live copy renders it as `"cat=" + cat + " Crack percentage={:.2f}".format(confidence.item())`
(`util/cv2_utils.py:169`), so it prints a raw `[-1, 1]` value under the label "percentage" — a value
of `1.00` next to the word *percentage* reads as 1%, when it means maximum activation. The dead root
copy renders the same quantity as `"Crack {:.2f}%".format(confidence.item() * 100)`, which reads as
100%. Neither is meaningful, and the fact that the two copies disagree on the presentation of the
system's only trust signal is itself the finding. Printing this next to a measurement, in a document
an engineer may rely on, is worse than printing nothing.

**3.2.5 Post-processing runs Canny on the segmentation mask.**
`util/cv2_utils.py:55` applies `cv2.Canny` to the blurred fused **mask**, then extracts contours
from the edge map. But the mask is already a binary segmentation — its contours are directly
available and are in fact computed at `util/inference_utils.py:121` and then discarded. Running an
edge detector over a binary mask returns the *outline* of the mask, so each crack yields a closed
contour tracing both of its sides. This is why the perimeter-over-two heuristic for length exists
at all: it is a workaround for a self-inflicted problem.

**3.2.6 Thresholds are hardcoded, undocumented, and mutually inconsistent.**
The same fused map is binarised at **90** in `util/inference_utils.py:119-120` and at **40** in
`util/cv2_utils.py:21-22` and `:139`. Neither value is derived from anything, neither is
configurable, and the two code paths disagree about what counts as crack. The
`cv2.contourArea(c) < 100` filter at `util/cv2_utils.py:77` discards any crack occupying under
100 px of a 65,536 px frame — which, given the resize, means most real hairline cracks.

**3.2.7 `extract_bboxes` hardcodes the input size and returns one box.**
`util/cv2_utils.py:23` calls `mask.reshape(256, 256, 1)`, so the function breaks on any other input
size. It then returns a single bounding box spanning all crack pixels in the image, which
`util/cv2_utils.py:71-72` draws as though it were per-crack.

**3.2.8 The scale model is physically invalid for real photographs.**
The user types the real-world width and height of the entire photograph, and scale is derived as
`realWidth / imgWidth`. This assumes:

- the imaged surface is a plane, exactly fronto-parallel to the sensor;
- the whole frame is at a uniform distance;
- lens distortion is zero;
- the user knows the frame's real dimensions to begin with.

None of these hold for a handheld or drone photograph. An oblique capture produces a scale error
that varies continuously across the frame. Compounding this, `util/inference_utils.py:47` resizes
to a fixed 256 × 256 **without preserving aspect ratio**, so a 4:3 photograph is anisotropically
squashed — yet `pixelsPerMetricHeight` and `pixelsPerMetricWidth` are computed against the
*resized* dimensions, and §3.2.2 then applies only the width factor. The errors stack.

The industry solution is a **fiducial marker of known size placed in frame**, or camera intrinsics
plus a photogrammetric reconstruction with a scale bar or ground control point. Both are
well-understood. Neither is present.

**3.2.9 The resolution ceiling is the binding constraint.**
Restating §1.3 with the arithmetic, because this determines what the product can honestly claim:

| Frame coverage | GSD at 256 px | Pixels spanned by a 0.3 mm crack | Usable? |
|---|---|---|---|
| 1000 mm | 3.9 mm/px | 0.08 | No |
| 500 mm | 2.0 mm/px | 0.15 | No |
| 100 mm | 0.39 mm/px | 0.77 | No |
| 30 mm | 0.12 mm/px | 2.6 | Detection only |
| **10 mm** | **0.04 mm/px** | **7.7** | Approaching the 10–30 px literature target |

To measure a 0.3 mm crack the model would need to see roughly a **1 cm** field of view per 256 px
tile. The pipeline throws away all the resolution a modern 12–50 MP sensor provides. Published
photogrammetry-plus-CNN pipelines report ±0.26–0.71 mm error for sub-5 mm cracks
([Sensors 2023, 23(13):5878](https://www.mdpi.com/1424-8220/23/13/5878)) — and they operate at
appropriate GSD. This pipeline has no error figure at all.

**3.2.10 There is no evaluation of the deployed model. Anywhere.**
No IoU, no F1, no ODS/OIS, no precision-recall curve, no held-out test set, no confusion matrix,
no measurement-error validation against physically measured crack widths, no documented failure
modes. The 98.4% validation accuracy in the notebook is from a **different model** (ResNet-50
classifier) on a **different task** (patch-level binary classification) on a near-saturated
benchmark of 227 × 227 pre-cropped patches, and the README presents it as "the first type of
result" as though it were part of the same system. For a product whose entire value is a number
an engineer relies on, the absence of evaluation is the credibility gap.

### 3.3 P1 — Service and operational engineering

**3.3.1 The model is rebuilt and reloaded from disk on every request.**
`main.py:63`, inside the request handler:

```python
model = create_model(parameters)
```

This constructs `DeepCrackNet`, runs Xavier initialisation across every layer, and then
`torch.load`s the 59 MB checkpoint — per request. Throughput will be dominated by this, and memory
will spike unboundedly under concurrency. The model should be loaded once into application state
via a FastAPI lifespan handler.

**3.3.2 Blocking work runs on the async event loop.**
`async def predict` performs synchronous Torch inference and OpenCV processing directly. A single
in-flight inference stalls every other connection, including health checks. Inference belongs in a
thread pool or a separate worker process.

**3.3.3 `torch.load` is unsafe and non-portable.**
`util/inference_utils.py:55`: no `map_location`, so a GPU-saved checkpoint fails to load on a CPU
host; and no `weights_only=True`, making checkpoint loading an arbitrary-code-execution sink. If
model files are ever accepted from users or fetched from a bucket, this is a remote code execution
vector.

**3.3.4 The response is bloated with debug output.**
`main.py:70-73` returns seven base64-encoded PNGs — the annotated result plus the fused map plus
all five side-output maps. Base64 inflates payload by ~33%, and the side maps are of interest to a
researcher inspecting the model, not to an inspector doing a survey. This is a research
visualisation shipped as an API contract.

**3.3.5 No health endpoint, no logging, no metrics, no error handling.**
`main.py:61`, `[float(x) for x in dim.split('-')]`, raises an unhandled `ValueError` on malformed
input and returns an opaque 500. Negative and zero dimensions are accepted and propagate into the
scale factors. There is no `/health` or `/ready`, no structured logs, no request IDs, no latency or
error-rate metrics, and no timeouts anywhere.

**3.3.6 No batching, no queue, no async job model.**
Real inspection workloads are hundreds to thousands of images per site visit. A synchronous
one-image-per-HTTP-request design cannot serve that; it needs an upload-then-poll or
webhook-callback job model.

### 3.4 P1 — Security

**3.4.1 CORS is both wide open and self-contradictory.**
`main.py:32-38` sets `allow_origins=["*"]` together with `allow_credentials=True`. Browsers reject
this combination outright per the CORS specification, so the setting is simultaneously insecure in
intent and broken in effect. It signals that no origin policy was considered.

**3.4.2 There is no authentication, authorisation, or tenancy.**
Anyone who can reach the URL can run inference. There is no notion of a user, an account, or an
organisation. Inspection imagery is frequently commercially sensitive and sometimes
security-sensitive (dams, bridges, utilities); an unauthenticated endpoint is a non-starter for
any enterprise buyer.

**3.4.3 Uploads are unvalidated and unbounded.**
No size limit, no MIME verification, no content sniffing, no pixel-count guard. `cv2.imdecode`
runs on arbitrary attacker-controlled bytes. A decompression bomb or a large upload takes the
process down, and the front end's `accept` attribute is a client-side hint that offers no
server-side protection.

**3.4.4 No rate limiting.** GPU inference is expensive; an unauthenticated, unthrottled inference
endpoint is a denial-of-wallet target.

**3.4.5 The front end hardcodes localhost.**
`web-deepcrack/src/App.js:54` fetches `http://127.0.0.1:8000/...`. The app cannot be deployed
without a source edit, and under HTTPS it fails as mixed content.

**3.4.6 Model weights are committed to git.**
`pretrained_net_G.pth` (59 MB) is in the tree, inflating `.git` to 56 MB. Weights have no checksum,
no version, no training provenance, and no record of the dataset they came from. Every clone pays
the cost forever, and rotating the model rewrites history or grows it again.

### 3.5 P2 — Repository hygiene, licensing, and quality

**3.5.1 Two diverging copies of the core logic.**
`cv2_utils.py` and `inference_utils.py` exist at both the repository root and in `util/`, with
genuinely different content. As established in §3.1.3, the copy that runs is the one missing a bug
fix. This is the clearest possible symptom of the absence of tests and CI: a fix was made and
silently applied to dead code.

**3.5.2 The `.gitignore` is for a different programming language.**
It ignores `.Rhistory`, `.RData`, `.Rproj.user/`, `vignettes/*.html` and `.Renviron` — an
**R** template, in a Python and JavaScript project. Nothing ignores `__pycache__`, and consequently
compiled `.pyc` files for both Python 3.8 and 3.10 are committed under `util/__pycache__/` and
`util/models/__pycache__/`.

**3.5.3 There is no licence, and the copyright claim conflicts with the code's provenance.**
The README states the author "has the legal right to be referenced if parts of the repo or the full
repository is used in other systems or applications, papers or publications." But:

- `util/models/deepcrack_model.py:1`, `deepcrack_networks.py:2` and `roadnet_model.py:1` carry
  upstream author headers from the DeepCrack authors;
- `util/models/base_model.py` and `networks.py` are substantially the pix2pix / CycleGAN
  scaffolding, including its `define_G`, `init_net`, `get_scheduler` and `GANLoss` machinery.

There is **no LICENSE file**. A repository with no licence grants no rights to anyone, while
simultaneously asserting a blanket copyright over code the author did not write. No enterprise
procurement or open-source compliance review will clear this. It must be resolved before any
commercial conversation, and it is cheap to resolve: add the upstream licences, add a `NOTICE`
attributing derived files, and licence the original work explicitly.

**3.5.4 Significant dead code.**
`roadnet_model.py` and `roadnet_networks.py` are never referenced. Most of `networks.py` (ResNet and
U-Net generators, discriminators, GAN losses, gradient penalty) is unused. `Parameters` in
`main.py:16-28` declares `output_nc`, `ndf`, `netG='resnet_9blocks'`, `init_type` and `init_gain`
that the DeepCrack path never reads, which is actively misleading — `netG='resnet_9blocks'` implies
a ResNet generator is in use, and it is not.

**3.5.5 There are no tests, and the one test file is a lie.**
`web-deepcrack/src/App.test.js` is the untouched CRA default asserting that a "learn react" link
renders. `App.js` no longer renders it, so the only test in the repository **fails**.
`web-deepcrack/README.md` is likewise unmodified CRA boilerplate. There are no Python tests at all.

**3.5.6 No CI, linting, formatting, or type checking.** No GitHub Actions, no pre-commit, no ruff /
black / mypy, no eslint config beyond the CRA default.

**3.5.7 The notebook is a 3.6 MB unrelated artifact.** It trains a different model for a different
task and is committed with all output images embedded. It also contains `scheduler.step()` before
`optimizer.step()` — a known PyTorch ordering warning — and its train/val split is done by
irreversibly `shutil.move`-ing files around the user's disk.

### 3.6 Findings summary

| # | Severity | Finding | Location |
|---|---|---|---|
| 1 | P0 | `np.fromstring` removed in NumPy 2 — every request fails | `util/inference_utils.py:40` |
| 2 | P0 | CUDA asserted; no CPU fallback | `main.py:27`, `networks.py:105` |
| 3 | P0 | `ZeroDivisionError` on vertical cracks (fix exists in dead copy) | `util/cv2_utils.py:114` |
| 4 | P0 | No dependency manifest of any kind | repo root |
| 5 | P0 | Width = count of non-zero pixels across a whole image row | `util/cv2_utils.py:140` |
| 6 | P0 | Length = perimeter ÷ 2, wrong scale factor | `util/cv2_utils.py:132` |
| 7 | P0 | Only the first crack is measured (`return` inside loop) | `util/cv2_utils.py:171` |
| 8 | P0 | "Crack percentage" is a raw max activation, not a probability | `util/inference_utils.py:106` |
| 9 | P0 | Scale model assumes fronto-parallel plane; aspect ratio destroyed | `util/inference_utils.py:47` |
| 10 | P0 | 256×256 ceiling ⇒ ~30–40× too coarse for standards-relevant widths | `util/inference_utils.py:47` |
| 11 | P0 | No evaluation of the deployed model whatsoever | repo-wide |
| 12 | P1 | 59 MB model reloaded per request | `main.py:63` |
| 13 | P1 | Blocking inference on the async event loop | `main.py:59` |
| 14 | P1 | `torch.load` without `map_location` / `weights_only` | `util/inference_utils.py:55` |
| 15 | P1 | CORS `*` + credentials; no auth; no upload limits; no rate limit | `main.py:32-38` |
| 16 | P1 | Front end hardcodes `127.0.0.1:8000` | `web-deepcrack/src/App.js:54` |
| 17 | P1 | No health check, logging, metrics, or error handling | `main.py` |
| 18 | P2 | Duplicate diverging modules | root vs. `util/` |
| 19 | P2 | No LICENSE; copyright claim over third-party code | `README.md` |
| 20 | P2 | R-language `.gitignore`; `.pyc` and 59 MB weights committed | repo root |
| 21 | P2 | Zero tests; the one existing test fails | `web-deepcrack/src/` |
| 22 | P2 | Dead code: RoadNet, most of `networks.py`, unused params | `util/models/` |

---

## 4. Product viability assessment

### 4.1 The market is real and it is not empty

Infrastructure inspection is a genuine, large, and structurally growing market: ageing bridge and
tunnel stock, mandatory periodic inspection regimes, a shrinking pool of qualified inspectors, and
falling drone-capture costs. The demand is not in question.

What *is* in question is whether this repository can capture any of it, because the space already
has funded, shipping competitors:

| Competitor | Positioning |
|---|---|
| [SwissInspect](https://swissinspect.io/en/) | 3D damage mapping, multi-class segmentation, crack width, BIM integration, automated reporting; models labelled by structural engineers; deployed with federal/cantonal offices, railways, power plants |
| [Fadron HALO AI](https://fadron.com/halo-ai/) | Sub-millimetre crack detection at 0.3–0.5 mm GSD, 3D digital twins, ASTM D6433 PCI scoring, FAA Part 139 / ICAO Annex 14 compliance |
| [Niricson AUTOSPEX](https://www.niricson.com/autospex/) | Optical + thermal + acoustic; surface *and* subsurface defects, delamination to 50 mm depth; millimetre-level 3D |
| [ConeLabs](https://conelabs.ai/) | Photogrammetry-first platform, collaborative 3D review, bridge/roadway pilots |

Note carefully what every one of them leads with: **GSD figures, standards compliance, 3D
geometry, and reporting** — not model architecture. The segmentation model is a commodity. DeepCrack
is a 2019 paper; anyone can reproduce it in an afternoon, and modern segmentation backbones beat
it. **The model is not the moat and never was.**

### 4.2 Where the defensibility actually lives

Ranked by how hard each is to copy:

1. **Ground truth and evaluation.** A labelled, held-out, physically-measured validation set for a
   specific asset class, with published error bars. Expensive to build, immediately credible,
   directly answers the procurement question "how do I know this is right?"
2. **The capture protocol.** Telling the user exactly how to photograph so the measurement is
   valid, and *refusing to produce a number* when it is not — see §5.2. This is a product decision,
   not a research one, and it is the thing incumbents encode in their GSD guidance.
3. **The report artifact.** The deliverable an engineer signs. Format, defensibility, and audit
   trail are what the customer is actually buying.
4. **Longitudinal comparison.** Same crack, same asset, successive inspections: *is it growing, and
   how fast?* This is the question that drives every maintenance budget decision, and it is the
   feature most likely to make a tool indispensable once adopted.
5. **Integration.** Asset-management systems, BIM, GIS.
6. The segmentation model. Last, and by a wide margin.

The current repository has item 6 and nothing else.

### 4.3 Honest probability assessment

**Path A — Head-on commercial SaaS. Probability: very low (<5%).**
Competing directly against funded teams that already ship 3D reconstruction, standards-compliant
reporting and enterprise integrations, starting from a research demo with no evaluation and no
licence, is not a winnable fight. The gap is not the model; it is the four other layers listed
above, plus sales into a conservative, liability-driven, procurement-heavy industry.

**Path B — A narrow vertical wedge. Probability: moderate.**
Pick one asset class, one capture modality, one compliance regime, one geography. The narrower the
scope, the more tractable the accuracy problem becomes, because the capture conditions can be
constrained. Strongest candidate wedges, in order of technical tractability:

- **Precast concrete plant QA.** Fixed lighting, fixed camera-to-surface distance, fixed geometry,
  repeatable throughput, and a customer with an existing QA budget. The scale-calibration problem
  largely evaporates because the rig is fixed. This is by far the easiest place to make the numbers
  actually correct, and it is the wedge most people overlook because it is less glamorous than
  bridges.
- **Façade / building envelope surveys** for property condition assessments — high volume,
  standardised deliverable, lower liability stakes than structural certification.
- **Tunnel lining inspection** — high value, constrained geometry, and see §6.4.

**Path C — Internal tool at an engineering firm. Probability: highest.**
See §6.4. If the author has access to an inspection practice, this dominates every other path on
risk-adjusted terms: distribution is free, ground truth is available, the capture protocol can be
mandated rather than requested, and professional liability continues to sit with the reviewing
engineer rather than with a software vendor.

**Path D — A credible open-source reference implementation.** High probability of genuine adoption,
low probability of revenue. The niche is real: there is currently no public, reproducible,
honestly-evaluated crack segmentation *service* with a documented capture protocol and published
error bars. Filling that gap would earn citations, contributors and credibility — which are the
raw material for Paths B and C.

**What is not viable under any path** is the current framing: a general-purpose "upload a photo,
get a crack width" web app. It cannot be made accurate at arbitrary capture conditions, and
shipping a number that looks authoritative but is not is a liability risk, not a product.

---

## 5. Users and their needs

### 5.1 Who the users would be

**Primary — the inspecting engineer.** Structural or civil engineer performing condition
assessments on bridges, tunnels, dams, façades or parking structures. They photograph defects,
measure a sample by hand with a crack comparator card or pocket microscope, and write a report
whose conclusions they are professionally liable for. They are the buyer, the user, and the person
whose trust must be earned.

**Secondary — the asset owner / infrastructure manager.** Municipality, rail operator, highway
agency, utility, port authority, or large property owner. They do not use the tool; they consume
its output. They care about coverage, comparability across inspections and across contractors,
defensibility to a regulator, and cost per asset.

**Secondary — the drone or survey service provider.** Captures imagery at volume and needs to sell
analysis on top of it rather than delivering a folder of JPEGs. Strong channel partner; weak direct
customer, because they will not carry structural liability.

**Tertiary — the precast / ready-mix QA technician.** Repeatable in-plant inspection under
controlled conditions. Technically the easiest user to serve well (§4.3, Path B).

**Tertiary — the property condition assessor / building surveyor.** Transaction-driven, high
volume, standardised deliverable, faster sales cycle than infrastructure.

**Current de facto users — students and researchers.** Everyone using this repository today is
reproducing a paper or writing a thesis. Their needs are real but are entirely satisfied by
Path D, and they will never pay.

### 5.2 What they actually need

Framed as jobs to be done, ordered by how much they matter:

**"I need to trust the number, or I cannot use it."**
The inspecting engineer signs the report. A width with no stated uncertainty is worse than no
width, because it invites reliance it cannot support. This means: a stated error bar, a validation
study they can read, and — critically — **the tool must refuse to report a width when the ground
sampling distance is insufficient**. A system that says "this image is 2.1 mm/pixel; I can confirm
a crack is present but I cannot measure its width — re-shoot at 30 cm distance" earns more trust in
one interaction than a system that always answers earns in a hundred.

**"I need it to cover the whole asset, not one photo."**
Inspections produce hundreds to thousands of images per site. Any workflow that is
one-photo-at-a-time is a toy. Batch upload, per-asset organisation, and progress tracking are table
stakes.

**"I need to know if it changed since last time."**
The single highest-value question in the domain is not "is there a crack" — the engineer can see
that — it is "is this crack wider than it was eighteen months ago, and how fast is it growing?"
That drives repair prioritisation and budget. It requires stable crack identity across inspections,
consistent registration to asset geometry, and consistent measurement. It is the feature that makes
a tool sticky, and no part of the current system moves toward it.

**"I need to correct the model when it is wrong."**
Non-negotiable. The engineer must be able to reject false positives, add missed cracks, and adjust
a measurement — with the override recorded and attributed. Human-in-the-loop is not a fallback for
a weak model; it is the liability architecture the profession requires. Every override is also
free training data.

**"I need the report I already produce, not a new one."**
The deliverable is a document in a specific format, with the client's severity taxonomy, referencing
the right standard (ACI, AASHTO, EN, or a national code). If the tool produces a PNG the engineer
must then manually transcribe, it has added work.

**"I need it to work where the asset is."**
Tunnels, undersides of bridge decks, and remote sites have no connectivity. Capture must work
offline with deferred processing.

**"I need to not be the one who leaked the client's imagery."**
Data residency, access control, retention policy, and an audit trail. For public infrastructure
this can extend to formal security requirements.

**"I need it to be cheaper than the alternative."**
The alternative is a person with a crack card, or rope access, or a snooper truck. The economics
are usually favourable — that is precisely why the market exists — but the tool must be measured
against them explicitly.

### 5.3 What they do *not* need

Worth stating, because the current product delivers several of these:

- Five side-output feature maps from the network's intermediate layers.
- A model-architecture explanation.
- A percentage that is not a probability.
- A single annotated image with no underlying structured data.

---

## 6. The plan

Four phases. Each has a clear deliverable and a gate that must pass before the next begins. Effort
is characterised by which subsystems change and how invasive the edits are, not by calendar time.

### Phase 0 — Make it true

*Goal: a person who is not the author can install it, run it, and get the same result twice.
Nothing here adds a feature; it makes the existing claims survive contact with a second machine.*

**Scope: `main.py`, `util/`, repository root. Mostly additive, low risk.**

1. Add `pyproject.toml` with pinned dependencies and a lockfile. Add a Dockerfile with CPU and CUDA
   variants. Fix #4.
2. Delete the root-level `cv2_utils.py` and `inference_utils.py` duplicates; keep `util/` as the
   single source of truth and port over the divide-by-zero guard the dead copy had. Fixes #3, #18.
3. Replace `np.fromstring` with `np.frombuffer`. Fixes #1.
4. Auto-detect the device; make `gpu_ids` empty by default; replace the bare `assert` with a clear
   error. Fixes #2.
5. Load the model once at application startup via a FastAPI lifespan handler; make the checkpoint
   path configurable and absolute; add `map_location` and `weights_only=True`. Fixes #12, #14.
6. Move inference off the event loop into a thread pool. Fixes #13.
7. Move `pretrained_net_G.pth` out of git into a release asset or object store, fetched on first run
   and verified by SHA-256. Record its training provenance. Fixes #20 in part.
8. Replace the R `.gitignore` with a Python + Node one; purge committed `__pycache__`. Fixes #20.
9. Add `LICENSE`, plus a `NOTICE` attributing DeepCrack and pix2pix on the derived files, and
   correct the README's copyright section to distinguish original from derived work. **This is a
   blocker for every downstream path and should be done first.** Fixes #19.
10. Delete `roadnet_*.py`, the unused generator/discriminator machinery in `networks.py`, and the
    unread fields in `Parameters`. Fixes #22.
11. Add golden-image regression tests: a handful of fixture images with committed expected mask
    hashes and expected measurement values, so any change to the pipeline that alters output is
    caught. Add a front-end test that reflects what `App.js` actually renders. Fixes #21.
12. Add CI running lint, type check, tests, and a Docker build on every push. Fixes #6 in §3.5.
13. Add `/health`, structured JSON logging with request IDs, input validation on the dimension and
    unit parameters, and explicit error responses. Fixes #17.
14. Rewrite the README: what it is, what it is not, how to run it, what it has been measured to do,
    and the known limitations. Remove the ResNet notebook results from the product narrative or
    label them clearly as a separate experiment.

**Gate 0:** a clean clone builds in Docker and passes CI on a CPU-only runner; golden tests are
green; `LICENSE` and `NOTICE` are in place.

### Phase 1 — Make the number defensible

*Goal: any width this system reports is either correct within a stated tolerance, or explicitly
refused. This is the phase that determines whether there is a product at all.*

**Scope: rewrite of `util/cv2_utils.py`, significant changes to `util/inference_utils.py`, new
calibration and evaluation modules. Invasive; this is the core work.**

1. **Tiled full-resolution inference.** Stop resizing to 256 × 256. Tile the native-resolution
   image with overlap, run the network per tile, and stitch the masks with feathered blending.
   Preserves the sensor's resolution, which is the entire point. Fixes #10.
2. **Real scale calibration.** Support, in order of preference: (a) a fiducial marker of known
   dimensions detected in-frame — ArUco or ChArUco — which also yields the homography needed to
   rectify oblique captures; (b) camera intrinsics plus a known standoff distance; (c) the current
   manual entry, retained but clearly flagged as low-confidence. Rectify to a fronto-parallel plane
   before measuring. Fixes #9.
3. **A GSD gate.** Compute ground sampling distance from the calibration. If GSD exceeds the
   threshold for the requested measurement precision, **return detection only and an explicit
   explanation of why width was not reported, with a concrete instruction for re-capture.** Publish
   the threshold table (§3.2.9). This is the highest-trust feature in the entire plan and it costs
   almost nothing to build.
4. **Correct width measurement.** Skeletonise each crack mask; compute a Euclidean distance
   transform from the mask boundary; sample it along the skeleton to produce a **width profile**.
   Report maximum, mean, and the profile itself, with a confidence interval derived from GSD. Fixes
   #5.
5. **Correct length measurement.** Skeleton path length, not perimeter over two, with correct
   anisotropic scale factors. Fixes #6.
6. **Per-crack instance segmentation.** Connected components on the mask, with morphological
   bridging of small gaps; stable IDs; measure and return every crack, not the first. Move the
   `return` out of the loop. Fixes #7.
7. **Use the mask contours directly.** Delete the Canny-on-a-mask step. Fixes #3.2.5.
8. **Calibrated confidence.** Replace the max-activation figure with a per-crack score derived from
   the mask probability distribution over that instance, calibrated on a held-out set (temperature
   scaling or isotonic regression) so that a reported 0.8 means something. Or remove it entirely
   until it can be calibrated. Fixes #8.
9. **Configuration, not magic numbers.** Every threshold into a config object with documented
   defaults and a single consistent value across the codebase. Fixes #3.2.6.
10. **Build an evaluation set and publish the results.** This is the deliverable that makes Phase 1
    real. Assemble a held-out set of field images with pixel-level crack labels *and* physically
    measured ground-truth widths (crack comparator or pocket microscope, per ACI 224.1R-07).
    Report IoU, F1, ODS/OIS for segmentation, and mean absolute error with a Bland–Altman plot for
    width — stratified by GSD band. Commit the harness so results are reproducible. Fixes #11.
11. **Structured output.** Return JSON: a list of crack instances, each with ID, polygon, skeleton,
    length, width statistics, orientation, confidence, and the GSD under which it was measured. The
    annotated image becomes a rendering of that data, not the product. Drop the five side maps from
    the default response. Fixes #3.3.4.

**Gate 1:** published measurement error on the held-out set, stratified by GSD, that is competitive
with the ±0.26–0.71 mm range reported in the literature for sub-5 mm cracks — **or** a documented
GSD threshold below which the system honestly declines to measure. Either outcome is a pass;
shipping an unvalidated number is the only failure.

### Phase 2 — Make it a workflow

*Goal: an engineer can run a real inspection end to end and produce the document they actually
deliver. This is where it stops being an API and starts being a product.*

**Scope: new persistence layer, new backend services, substantial front-end work. The largest
phase by volume, but the lowest technical risk.**

1. Projects, assets, and inspections as first-class entities with a real database.
2. Batch upload of hundreds of images; asynchronous job queue; progress reporting; per-image status.
3. Authentication, organisations, role-based access, and per-tenant data isolation. Fixes #15.
4. **Human-in-the-loop review UI**: accept, reject, edit, and add crack instances; every override
   recorded with user and timestamp. This is a liability requirement, not a nice-to-have, and it
   generates the training data for Phase 3.
5. EXIF and GPS ingestion; asset-relative positioning so a crack has a location, not just an image.
6. **Report generation**: PDF and CSV, mapped to a configurable severity taxonomy (ACI 224R-01,
   AASHTO narrow/moderate/wide, or a client-specific scheme), including the capture metadata, the
   GSD, the measurement uncertainty, and the reviewer's sign-off.
7. Rebuild the front end properly: configurable API endpoint, upload progress, error states, empty
   states, keyboard-driven review. Fixes #16.
8. Upload limits, MIME validation, pixel-count guards, rate limiting, and a real CORS policy. Fixes
   #15.
9. Metrics, tracing, alerting, and a documented deployment.

**Gate 2:** a qualified engineer completes a real inspection end to end and reports that the output
saved them time versus their existing process. If they would not use it again unprompted, stop and
reassess rather than proceeding.

### Phase 3 — Make it defensible

*Goal: the features a competitor cannot trivially copy and that make the tool hard to remove once
adopted.*

**Scope: new subsystems; each is independently valuable and can be sequenced by customer demand.**

1. **Longitudinal crack tracking.** Register successive inspections of the same asset; match crack
   instances across time; report growth rate with statistical significance. This is the highest-value
   feature in the entire plan (§5.2) and the primary reason a customer renews.
2. **Guided capture.** A mobile or drone-planning aid that computes the required standoff distance
   for a target GSD and warns in real time when the operator is too far away or too oblique. Turns
   the Phase 1 GSD gate from a rejection into a prevention.
3. **Photogrammetric 3D.** Structure-from-motion reconstruction so defects are located on asset
   geometry rather than in isolated photos. Required for parity with incumbents; large effort;
   sequence it only when customers demand it.
4. **Active learning.** Feed Phase 2's human overrides back into fine-tuning, with per-customer
   domain adaptation. The review UI turns into a compounding data advantage.
5. **Multi-class defects.** Spalling, delamination, efflorescence, corrosion staining, exposed
   rebar. Cracks alone are a subset of what an inspection records.
6. **Integrations.** Asset management systems, BIM, GIS export.

### 6.4 A strategic note on distribution

The commit history contains a commit authored from an infrastructure-engineering company domain.
If the author works in or alongside an inspection practice, that changes the recommended sequence
materially, and for the better:

- Ground truth for Phase 1 becomes available rather than expensive — real assets, real crack cards,
  real inspection records.
- The capture protocol from Phase 1 can be **mandated** for internal field crews rather than
  requested of strangers. This removes the single hardest product problem in the whole plan.
- Distribution costs nothing, and the feedback loop of Gate 2 is measured in days rather than sales
  cycles.
- Professional liability remains with the reviewing engineer under existing insurance, rather than
  being transferred to a software vendor with no track record.
- If it works internally, it has a reference customer, a validation study, and a case study — which
  is exactly the evidence needed to attempt Path B externally.

**The recommendation is to treat Path C as the primary route and Path D as the public face of it.**
Ship the honest open-source reference implementation to build credibility and attract contributors;
prove the accuracy internally on real assets; only then consider commercialising a narrow vertical.

### 6.5 Decision gates and kill criteria

Stop or reassess if:

- **After Phase 1**, measurement error cannot be brought within a range that an engineer will accept
  for *any* capture protocol. If the numbers cannot be made defensible, there is no product, and the
  right outcome is Path D alone — a good open-source segmentation service, honestly scoped.
- **After Gate 2**, the pilot engineer does not reach for the tool unprompted on their next
  inspection. Time saved is the only metric that matters here; enthusiasm in a demo is not evidence.
- At any point, an incumbent ships the same narrow wedge at a price that cannot be matched. Reassess
  the wedge rather than the whole effort.

### 6.6 Immediate next actions, in order

1. Add `LICENSE` and `NOTICE`, and correct the README's copyright section. Nothing else can proceed
   commercially until this is done, and it is the cheapest item in the plan.
2. Fix the three P0 crashes (#1, #2, #3) and add `pyproject.toml`. The service currently does not
   run.
3. Delete the duplicate root modules and add golden-image tests plus CI, so the next fix cannot be
   applied to dead code.
4. Implement the GSD gate before implementing better measurement. Being honest about what cannot be
   measured is a smaller change than measuring correctly, and it is worth more in trust per unit of
   effort than anything else in this document.
5. Assemble even a small ground-truth set — twenty images with physically measured crack widths is
   enough to know whether Phase 1 is achievable.

---

## 7. Closing assessment

The engineering in this repository is not bad research code. It is a faithful re-implementation
with a working demo, and the author clearly understands the domain. The problem is a category
error: a research demo has been described as a measurement product, and the gap between those two
things is almost entirely composed of the work nobody enjoys — evaluation, calibration,
reproducibility, licensing, and honest refusal.

The good news is that the hardest problem here is not a research problem. Every finding in §3 has a
known, published solution. The resolution ceiling is fixed by tiling. The width measurement is fixed
by a distance transform. The scale problem is fixed by a fiducial marker. The trust problem is fixed
by publishing an error bar and declining to answer when the data does not support an answer.

What the project needs most is not a better model. It is the discipline to measure how wrong it
currently is, publish that, and fix it in the open.
