import os
import sys
from google.cloud import aiplatform
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions

# Configuration (replace these with your actual values)
PROJECT_ID = "lofty-reserve-454316-q5"  # Your Google Cloud project ID
LOCATION = "us-central1"       # Region where your endpoint is deployed (e.g., us-central1)
ENDPOINT_ID = "your-endpoint-id"  # ID of your deployed Vertex AI endpoint
CREDENTIALS_PATH = "credential/vertex-ai-key.json"  # Path to your service account JSON key file

# Example input data (adjust based on your model's expected input schema)
SAMPLE_INSTANCES = [
    {"feature1": 1.0, "feature2": 2.0}  # Replace with your model's input format
]

def initialize_vertex_ai():
    """
    Initialize the Vertex AI client with the provided credentials.
    Returns the initialized client or exits on failure.
    """
    try:
        # Set the environment variable for credentials if not already set
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            if not os.path.exists(CREDENTIALS_PATH):
                print(f"❌ Error: Credentials file not found at {CREDENTIALS_PATH}")
                sys.exit(1)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
            print(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS to {CREDENTIALS_PATH}")

        # Initialize the Vertex AI client
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        print(f"✅ Vertex AI client initialized for project {PROJECT_ID} in {LOCATION}")
        return True
    except auth_exceptions.GoogleAuthError as e:
        print(f"❌ Authentication error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during initialization: {e}")
        sys.exit(1)

def get_endpoint(endpoint_id):
    """
    Retrieve the Vertex