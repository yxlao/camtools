import requests
from pathlib import Path


class _ArtifactManager:
    BASE_URL = "https://github.com/yxlao/camtools-artifacts/releases/download"

    def __init__(self):
        self.cache_dir = Path.home() / ".camtools"

    def get_path(self, artifact_key: str, verbose: bool = False):
        """
        Checks if the artifact is locally available, and if not, attempts to download it.
        """
        artifact_path = self.cache_dir / artifact_key
        if artifact_path.exists():
            if verbose:
                print(f"Found local artifact: {artifact_path}.")
        else:
            url = f"{self.BASE_URL}/{artifact_key}"
            self._try_download_artifact(url, artifact_path)
        return artifact_path

    def _try_download_artifact(self, url, artifact_path):
        """
        Attempts to download the artifact from the provided URL.
        """
        try:
            print(f"Attempting to download from {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_bytes(response.content)
                print(f"Downloaded to {artifact_path}.")
            else:
                raise RuntimeError(
                    f"Artifact download failed. Please check the URL {url}"
                )
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while downloading the artifact: {e}")


def get_artifact_path(artifact_key: str):
    """
    Checks if the artifact is locally available, and if not, attempts to download it.
    """
    artifact_manager = _ArtifactManager()
    return artifact_manager.get_path(artifact_key)
