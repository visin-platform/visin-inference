# Getting started

## Install from source

Before the first PyPI release, install from a checkout:

```sh
git clone https://github.com/visin-platform/visin-inference.git
cd visin-inference
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[models]'
visin-infer serve --allow-origin https://visin.example.com
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
After the package is published, use `python -m pip install 'visin-infer[models]'`.
Install the PyTorch build appropriate for your GPU before installing the models
extra if your platform needs a particular CUDA or accelerator build.

The default weights are `yolo26s.pt` and `sam2.1_b.pt`. First start downloads them
through Ultralytics into `./models`. You can download in advance:

```sh
visin-infer fetch --models-dir ./models
```

Open the Visin label editor, select **Model**, and connect to
`http://localhost:8000`. Use your site's exact origin for `--allow-origin`, without
a path. If `--token` is enabled, paste the printed token into the editor.

## Docker with an NVIDIA GPU

The host needs Docker and NVIDIA Container Toolkit configured for `--gpus all`.

```sh
docker build -t visin-infer .
docker run --rm --gpus all -p 127.0.0.1:8000:8000 \
  -v visin-infer-weights:/data \
  visin-infer --allow-origin https://visin.example.com --token
```

The volume keeps downloaded weights under `/data/models`. The container binds all
interfaces internally; publishing its port on host loopback restricts access.
Omit `--gpus all` for CPU inference. No weights are downloaded while building.

## Troubleshooting

- Missing `ultralytics`: install the `[models]` extra into the same environment as
  the command. The base package intentionally installs only NumPy and Pillow.
- No GPU reported: check your PyTorch accelerator installation. `/info` reports
  the device detected by PyTorch, not a per-model device guarantee.
- Browser connection fails: check the exact allowed origin, port, token, and the
  browser's local-network permissions. Use localhost when the site is HTTPS;
  accessing an HTTP LAN address can be blocked as mixed content.
- Download fails: check outbound access to the Ultralytics release assets, or
  provide already downloaded weight paths through the configuration options.
