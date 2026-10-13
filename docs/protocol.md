# HTTP protocol

Image coordinates are pixels in the image sent. Bounding boxes use `[x, y, width,
height]`. Requests carrying an `Origin` must match an allowed origin. If a token
is configured, include `X-Visin-Token` on all requests except CORS preflight.

## GET /info

```json
{"name":"visin-infer","version":"0.1.0","device":"CPU","classes":["person"],"supports":{"detect":true,"segment":true}}
```

The classes come from the detector; without a detector the list is empty.

## POST /detect?minScore=0.25

Send image bytes, usually JPEG or PNG, with an appropriate `Content-Type`.
The threshold defaults to 0.25, is clamped to 0–1, and must be finite.

```json
{"boxes":[{"bbox":[10,20,30,40],"class":"person","score":0.91}]}
```

## POST /segment

Send JSON with a frame identifier and one or more point or box prompts:

```json
{"imageId":"frame-1","image":"<base64 image bytes>","points":[{"x":100,"y":200,"label":1}],"box":[50,100,200,300]}
```

`imageId` must be a nonempty string. Point label 1 means foreground and 0 means
background; omitted labels default to 1. A box requires positive width and
height. Include a point or a box (both are allowed). The first request for an
image ID supplies `image`; later clicks can omit it. Only one image embedding
is cached, so changing frames can evict it. Treat IDs as unique to image content.

```json
{"rle":"<base64 VMSK bytes>","width":640,"height":480,"bbox":[50,100,200,300],"score":0.98}
```

A 409 with `{"error":"image_needed"}` means resend the prompt with its image.
The response bbox encloses nonzero mask pixels; an empty mask has `[0,0,0,0]`.

## Mask encoding

Decode base64 to obtain the binary VMSK format:

| Field | Encoding |
| --- | --- |
| Magic | Four ASCII bytes `VMSK` |
| Version | One byte, currently 1 |
| Width, height | Two unsigned 32-bit little-endian integers |
| Runs | Repeated value byte followed by unsigned LEB128 run length |

Runs traverse rows in C order, covering width × height pixels. Zero denotes
background. `visin_infer.rle.decode_mask` decodes server-generated masks for
scripts; it is not intended to validate arbitrary untrusted mask files.

## Errors and limits

Errors use `{"error":"message"}`. Status codes: 400 invalid input or body over
64 MiB, 401 missing/wrong token, 403 disallowed origin, 404 unknown route, 409
image needed, 501 unavailable model capability, 500 backend failure. Model calls
are serialized; the service is intended for one local annotator.
