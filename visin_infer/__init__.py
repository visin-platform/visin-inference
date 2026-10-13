"""A small local inference server for the Visin label editor.

The editor, running in a browser on visin.eu (or wherever Visin is deployed),
sends a frame to this server on the annotator's own machine and gets boxes or a
mask back. Nothing here talks to Visin.
"""

__version__ = "0.1.0"
