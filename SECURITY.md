# Security

Report vulnerabilities privately through this repository's
[GitHub security advisories](https://github.com/visin-platform/visin-inference/security/advisories/new).
Include the version, configuration, impact and reproduction steps. Avoid public
issues containing exploit details or private images/tokens. Fixes target the
latest released version.

The service defaults to loopback and allows configured browser origins. Clients
without an Origin header are permitted; CORS is a browser restriction, not
client authentication. Enable `--token` when you need authentication. The token
is printed on startup for use in the editor; treat terminal output as sensitive.

Binding a LAN/public interface exposes your inference service to clients that
can reach it. This standard-library HTTP server is intended for a single local
annotator. It has no TLS or per-client resource quotas. Request bodies are capped
at 64 MiB, and Pillow applies its own image-size checks. Do not expose it directly
to the internet.

Only load trusted model files. Model loading and initial downloads are performed
by third-party libraries. Frames go from the browser to this server; the server
code does not upload them to Visin. Review your dependencies and deployment for
additional network behavior.
