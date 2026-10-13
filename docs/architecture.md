# Architecture

| Module | Responsibility |
| --- | --- |
| `cli.py` | CLI arguments, weight download/cache, model startup. |
| `server.py` | HTTP, CORS, optional token, input validation, serialized inference. |
| `backends.py` | Detector/segmenter protocols and lazy Ultralytics adapters. |
| `rle.py` | VMSK mask encoding and decoding. |

Images are decoded as RGB with Pillow. The Ultralytics adapters convert them to
contiguous BGR arrays. `Inference` locks model calls; the SAM adapter remembers
one image ID and embedding. The HTTP handler uses the standard library's
`ThreadingHTTPServer`. It is intended for local use, without a production reverse
proxy, TLS server, persistent image storage, or multi-user scheduling.

To substitute a backend, implement `Detector` or `Segmenter`, instantiate
`Inference`, and pass it to `create_server`. A segmenter must raise `ImageNeeded`
when an omitted image is not cached. Return a 2-D mask matching the image and a
score. Ultralytics and PyTorch are imported when constructing real models, not
when importing the package or testing the HTTP protocol.
