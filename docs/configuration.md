# Configuration

```sh
visin-infer serve --help
visin-infer fetch --help
```

| Serve option | Default | Meaning |
| --- | --- | --- |
| `--allow-origin URL` | Required | Exact browser origin; repeat for multiple sites. `*` is rejected. |
| `--host` | `127.0.0.1` | Bind address. A wider address exposes access to other machines. |
| `--port` | `8000` | HTTP port. |
| `--detector` | `yolo26s.pt` | Ultralytics detector weights, or `none`. |
| `--segmenter` | `sam2.1_b.pt` | Ultralytics SAM weights, or `none`. |
| `--models-dir` | `models` | Cache directory; `VISIN_INFER_MODELS` sets the default. |
| `--token [VALUE]` | Disabled | Require `X-Visin-Token`; omit VALUE to generate and print one. |

`fetch` accepts the detector, segmenter and models-directory options and exits
without starting the server. An existing path is used as given. Otherwise the
filename is resolved under the models directory and downloaded if missing.

For detection only:

```sh
visin-infer serve --allow-origin https://visin.example.com --segmenter none
```

For a protocol smoke test without model dependencies:

```sh
visin-infer serve --allow-origin http://localhost:3008 --detector none --segmenter none
curl http://localhost:8000/info
```

This reports both capabilities as false. Detection and segmentation return 501.
