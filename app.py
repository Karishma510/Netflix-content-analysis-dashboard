# Gunicorn entrypoint for Render deployment.
# Imports the Dash app and exposes the underlying Flask server.
from Karishma_NetflixContentAnalysis import server  # noqa: F401
