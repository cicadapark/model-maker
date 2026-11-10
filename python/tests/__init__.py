import os
import tempfile

# The tests run jobs through the pipeline, which appends to the job ledger: keep their lines out of the real one.
os.environ["MODEL_MAKER_LEDGER"] = tempfile.mkdtemp(prefix="model-maker-test-ledger-")
