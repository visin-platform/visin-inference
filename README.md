# visin-infer

Runs object detection (YOLO) and click-to-segment (SAM) on **your own GPU** for the Visin label editor.

The editor in your browser sends a frame directly to this server on your machine and gets boxes or a mask back.
This server does not upload frames to Visin.

```
your PC                                                 your Visin site
┌──────────────────────────────┐                        ┌──────────────────────┐
│ browser: label editor ───────┼── frames, labels ──────┤ label-service        │
│     │  http://localhost:8000                          └──────────────────────┘
│     ▼                        │
│ visin-infer (YOLO + SAM)     │  127.0.0.1 by default; allowed browser origins
└──────────────────────────────┘
```

## Run it

Requires Python 3.10+. Before the first release, install from a checkout with
`python -m pip install '.[models]'`. The base package installs NumPy and Pillow;
the `models` extra adds Ultralytics and its inference dependencies. A GPU is
recommended; CPU inference is supported but slower.

```sh
python -m pip install 'visin-infer[models]'  # after the first PyPI release
visin-infer serve --allow-origin https://visin.example.com
```

`--allow-origin` is the address of your Visin site (exactly what is in the browser's address bar, without a path).
Weights (`yolo26s.pt`, `sam2.1_b.pt`) are downloaded on first start. Then open the label editor, press **Model**,
and **Connect**.

| Option | |
|---|---|
| `--allow-origin URL` | required, repeatable; never `*` |
| `--host` | default `127.0.0.1`. A wider bind lets other machines use your GPU |
| `--port` | default `8000` |
| `--detector WEIGHTS` / `none` | any ultralytics detector; `none` for segmentation only |
| `--segmenter WEIGHTS` / `none` | any ultralytics SAM weights; `none` for detection only |
| `--models-dir DIR` | weight cache (default `./models`, or `VISIN_INFER_MODELS`) |
| `--token [VALUE]` | require `X-Visin-Token`; with no value one is generated and printed |

## Protocol

Small on purpose, so another model can sit behind it. All coordinates are pixels of the image you sent.

```
GET  /info
  → { name, version, device, classes: [name], supports: { detect, segment } }

POST /detect?minScore=0.25        body: the image bytes (JPEG or PNG)
  → { boxes: [{ bbox: [x, y, w, h], class, score }] }

POST /segment                     body: JSON
     { imageId, image?: base64, points: [{ x, y, label: 1|0 }], box?: [x, y, w, h] }
  → { rle: base64, width, height, bbox: [x, y, w, h], score }
  → 409 { error: "image_needed" } when the server holds no image for `imageId`; resend with `image`
```

`rle` is a mask as run-length bytes (`"VMSK"`, version 1, width and height `u32le`, then `value u8` and an unsigned
LEB128 length per run) — the same form the editor uploads its own masks in. The server keeps the image embedding
for the last `imageId`, so only the first click on a frame pays for the SAM encoder.

## Security

- Binds `127.0.0.1` by default; anything wider is your decision.
- Answers a browser only when its page is one of `--allow-origin`, and says
  `Access-Control-Allow-Private-Network: true` to those. Browser local-network permissions may also be required.
- `--token` requires `X-Visin-Token`. Clients without an `Origin` header are allowed; use a token for authentication.
- An `https` page may call `http://localhost`, but not `http://192.168.x.x`; use localhost.

## Tests

```sh
python3 -m unittest discover -s tests -t .
```

The tests run the real HTTP server with fake models, so they need no GPU and none of the model packages.

## Development and documentation

```sh
make install
make check  # lint, types, offline tests/coverage, package, strict docs build
```

Real-model tests are opt-in: install `.[models]`, run `visin-infer fetch`, then
`make test-models`. Routine tests never download weights.

See [getting started](docs/getting-started.md), [configuration](docs/configuration.md),
[protocol](docs/protocol.md), [contributing](CONTRIBUTING.md),
[security reporting](SECURITY.md), and [release setup](docs/releases.md).
The documentation site is published through GitHub Pages after repository setup.

## License

[AGPL-3.0-only](LICENSE). Dependencies and downloaded model weights have their
own license terms. No model weights are bundled in the package.
