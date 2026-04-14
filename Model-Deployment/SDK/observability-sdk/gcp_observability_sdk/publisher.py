import json
from datetime import datetime

try:
    from google.cloud import pubsub_v1
except ImportError:  # pragma: no cover
    pubsub_v1 = None


def _json_default(value):
    try:
        return value.item()
    except AttributeError:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)


class ShiftHappensTracker:
    def __init__(self, project_id: str, topic_id: str):
        if pubsub_v1 is None:
            raise ImportError(
                "google-cloud-pubsub is required for ShiftHappensTracker. "
                "Install it with `pip install google-cloud-pubsub`."
            )
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_id)

    def track_prediction(
        self,
        features: dict,
        prediction,
        model_version: str,
        client_id: str = "unknown",
        prediction_probability=None,
        source_system: str = "sdk",
    ):
        """
        Asynchronously publish a prediction event to Pub/Sub.
        This does not block the caller waiting for the publish result.
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "client_id": client_id,
            "model_version": model_version,
            "source_system": source_system,
            "prediction": prediction,
            "prediction_probability": prediction_probability,
            "features": features,
            "event_type": "prediction",
        }
        data = json.dumps(payload, default=_json_default).encode("utf-8")
        future = self.publisher.publish(self.topic_path, data)
        return future
