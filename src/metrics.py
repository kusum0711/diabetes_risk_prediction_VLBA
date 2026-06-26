import urllib.error
import urllib.request

from src.env import PUSHGATEWAY_URL


def push_metrics(job, metrics, grouping=None):
    """Push metrics to a Prometheus Pushgateway.

    No-op when PUSHGATEWAY_URL is not set (local dev / unit tests). Uses the
    Pushgateway text exposition format over plain HTTP so we don't need an extra
    client dependency. `metrics` is a dict of {name: float}.
    """

    base = PUSHGATEWAY_URL
    if not base:
        print("  PUSHGATEWAY_URL not set — skipping metrics push")
        return

    # Build the /metrics/job/<job>[/<label>/<value>...] target URL.
    url = f"{base.rstrip('/')}/metrics/job/{job}"
    if grouping:
        for key, value in grouping.items():
            url += f"/{key}/{value}"

    lines = []
    for name, value in metrics.items():
        try:
            lines.append(f"{name} {float(value)}")
        except (TypeError, ValueError):
            continue
    body = ("\n".join(lines) + "\n").encode("utf-8")

    request = urllib.request.Request(url, data=body, method="PUT")
    request.add_header("Content-Type", "text/plain")

    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            resp.read()
        print(f"  Pushed {len(lines)} metrics to {url}")
    except (urllib.error.URLError, OSError) as exc:
        # Metrics are best-effort; never fail the pipeline on a push error.
        print(f"  ⚠️  Pushgateway push failed ({exc}) — continuing")
