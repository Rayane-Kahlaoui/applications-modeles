"""Verify that generic MLflow runs and artifacts use the configured store."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlflow
from mlflow.tracking import MlflowClient

from projet.mlops import tracking


class TrackingTest(unittest.TestCase):
    def test_run_and_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(tracking, "TRACKING_DIR", Path(temporary) / "mlflow"):
                tracking.configure_tracking("fixture")
                with mlflow.start_run() as run:
                    mlflow.log_param("model", "example")
                    mlflow.log_metric("score", 0.5)
                    mlflow.log_dict({"seed": 0}, "config.json")
                client = MlflowClient()
                saved = client.get_run(run.info.run_id)
                self.assertEqual(saved.data.params["model"], "example")
                self.assertEqual(saved.data.metrics["score"], 0.5)
                self.assertEqual(client.list_artifacts(run.info.run_id)[0].path, "config.json")
                client._tracking_client.store.engine.dispose()

    def test_named_severstal_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(tracking, "TRACKING_DIR", Path(temporary) / "mlflow"):
                with tracking.start_run("model-a", phase="validation", seed=7,
                                        description="first comparison") as run:
                    mlflow.log_metric("validation_recall", 0.7)
                client = MlflowClient()
                saved = client.get_run(run.info.run_id)
                self.assertEqual(saved.info.run_name, "model-a | validation | seed 7")
                self.assertEqual(saved.data.tags["dataset"], "Severstal")
                self.assertEqual(saved.data.tags["mlflow.note.content"], "first comparison")
                self.assertEqual(saved.data.params["seed"], "7")
                client._tracking_client.store.engine.dispose()


if __name__ == "__main__":
    unittest.main()
