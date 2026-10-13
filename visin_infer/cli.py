"""`visin-infer serve --allow-origin https://your-visin.example`, or `visin-infer fetch` to download the weights."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

from .server import Inference, create_server

DEFAULT_DETECTOR = "yolo26s.pt"
DEFAULT_SEGMENTER = "sam2.1_b.pt"


def resolve_weights(name: str, models_dir: Path) -> str:
    """Where a model's weights are: a path as given, else `models_dir/<name>`.

    A bare name that is not there yet is downloaded into `models_dir` (ultralytics
    publishes its weights as release assets), so the first start fetches them and
    every later one finds them.
    """
    given = Path(name)
    if given.exists():
        return str(given)
    target = models_dir / given.name
    if not target.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        from ultralytics.utils.downloads import attempt_download_asset

        print(f"Downloading {given.name} into {models_dir} ...", flush=True)
        attempt_download_asset(str(target))
    return str(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="visin-infer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="start the server")
    serve.add_argument(
        "--allow-origin",
        action="append",
        required=True,
        metavar="URL",
        help="the address of the Visin site that may use this server (repeatable); never *",
    )
    serve.add_argument(
        "--host", default="127.0.0.1", help="default 127.0.0.1; a wider bind lets other machines use your GPU"
    )
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--detector", default=DEFAULT_DETECTOR, help="ultralytics detector weights, or 'none'")
    serve.add_argument("--segmenter", default=DEFAULT_SEGMENTER, help="ultralytics SAM weights, or 'none'")
    serve.add_argument(
        "--models-dir",
        default=os.environ.get("VISIN_INFER_MODELS", "models"),
        help="where weights are kept and downloaded to (default ./models, or $VISIN_INFER_MODELS)",
    )
    serve.add_argument(
        "--token",
        nargs="?",
        const="",
        default=None,
        help="require this token (X-Visin-Token); with no value, one is generated and printed",
    )
    fetch = sub.add_parser("fetch", help="download the weights and exit, e.g. while building an image")
    fetch.add_argument("--detector", default=DEFAULT_DETECTOR)
    fetch.add_argument("--segmenter", default=DEFAULT_SEGMENTER)
    fetch.add_argument("--models-dir", default=os.environ.get("VISIN_INFER_MODELS", "models"))
    args = parser.parse_args(argv)

    if args.command == "fetch":
        for name in (args.detector, args.segmenter):
            if name.lower() != "none":
                print(resolve_weights(name, Path(args.models_dir)), flush=True)
        return 0

    if "*" in args.allow_origin:
        parser.error("--allow-origin must name the site, not *")
    token = args.token
    if token == "":
        token = secrets.token_urlsafe(16)

    models_dir = Path(args.models_dir)
    detector = segmenter = None
    if args.detector.lower() != "none":
        from .backends import UltralyticsDetector

        weights = resolve_weights(args.detector, models_dir)
        print(f"Loading detector {weights} ...", flush=True)
        detector = UltralyticsDetector(weights)
    if args.segmenter.lower() != "none":
        from .backends import UltralyticsSegmenter

        weights = resolve_weights(args.segmenter, models_dir)
        print(f"Loading segmenter {weights} ...", flush=True)
        segmenter = UltralyticsSegmenter(weights)

    inference = Inference(detector, segmenter)
    server = create_server(inference, args.allow_origin, args.host, args.port, token)
    print(f"visin-infer on http://{args.host}:{args.port}  ({inference.device})", flush=True)
    print(f"Allowed origin(s): {', '.join(args.allow_origin)}", flush=True)
    if token:
        print(f"Token: {token}   (paste it into the editor's Model dialog)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
