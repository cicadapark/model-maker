import os
import tempfile

# The tests run jobs through the pipeline, which appends to the job ledger: keep their lines out of the real one.
os.environ["MODEL_MAKER_LEDGER"] = tempfile.mkdtemp(prefix="model-maker-test-ledger-")
# Every picture the pipeline writes is signed (credentials.py): by a test certificate made for the tests alone, never
# by one the PC has, and never by a key given in the environment.
os.environ["MODEL_MAKER_TEST_CREDENTIALS"] = tempfile.mkdtemp(prefix="model-maker-test-credentials-")
os.environ.pop("MODEL_MAKER_SIGNING_KEY", None)
os.environ.pop("MODEL_MAKER_SIGNING_CERT", None)
os.environ.pop("MODEL_MAKER_TIMESTAMP_URL", None)
