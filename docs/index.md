# visin-infer

Run object detection and click-to-segment on your own machine for the
[Visin](https://github.com/visin-platform/visin) label editor. The browser sends
images directly to the local server; the server returns boxes or a mask.

Start with [Getting started](getting-started.md). The [HTTP protocol](protocol.md)
also lets you use the server from scripts or replace its model backends.

The package uses Python 3.10 or newer. YOLO and SAM run through the optional
Ultralytics backend. A GPU is recommended; CPU inference is possible but slower.
The server is intended for one annotator, with model calls serialized.

The source is licensed under AGPL-3.0-only; see the repository's
[LICENSE](https://github.com/visin-platform/visin-inference/blob/main/LICENSE).
Model dependencies and downloaded weights carry their own licenses; weights are
not included in the Python package or container image.
