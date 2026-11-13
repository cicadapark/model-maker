"""Content credentials: a signed C2PA manifest in every picture model-maker writes, saying what made it.

A picture painted by a model (a try, the chosen picture, a painted view, a touch-up, the texture, the report's
pictures of them) carries a manifest with one action, `c2pa.created`, whose digital source type (the IPTC vocabulary)
says it was made by a trained model: `trainedAlgorithmicMedia`. A report picture that draws our own marks over a
model's work (the outline over the picture, the grip and tip, the paint's coverage) says
`compositeWithTrainedAlgorithmicMedia`, and a picture our own code makes with no model (a touch-up's mask)
`algorithmicMedia`. Each names the generator (model-maker and its version) and the models whose work it shows, by name.

What a manifest never holds: the prompt, a file's path or name, the user's or the machine's name. The prompt is in
the object's sidecar, a file kept beside it; a picture travels alone (posted, shared, copied), and a manifest is
signed: what is in it cannot be taken out without breaking it. So the picture says that it was made by AI and by
what, and the sidecar says from which words.

The manifest is signed with the key and certificate the user gives (`--signing-key` and `--signing-cert`, or
MODEL_MAKER_SIGNING_KEY and MODEL_MAKER_SIGNING_CERT), else with a test certificate setup.sh makes on this computer
(.tools/c2pa-test, or MODEL_MAKER_TEST_CREDENTIALS). Checkers call a picture signed with the test certificate
untrusted, which it is: the mark is still there and still machine-readable. docs/setup.md says how to get a
certificate checkers trust.

    model-maker credentials                    which certificate signs, and whether it works
    model-maker credentials --make-test        make the test certificate if there is none (setup.sh runs it)
    model-maker credentials --read PICTURE...  what each picture's manifest says
"""

import dataclasses
import datetime
import io
import json
import os
import stat

# The IPTC digital source types (http://cv.iptc.org/newscodes/digitalsourcetype/).
IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
TRAINED = IPTC + "trainedAlgorithmicMedia"
COMPOSITE = IPTC + "compositeWithTrainedAlgorithmicMedia"
ALGORITHMIC = IPTC + "algorithmicMedia"
SOURCE_TYPES = (TRAINED, COMPOSITE, ALGORITHMIC)
GENERATOR = "model-maker"
DESCRIPTIONS = {TRAINED: "Made by AI models.",
                COMPOSITE: "Made from AI models' work, with marks drawn by model-maker's own code.",
                ALGORITHMIC: "Made by model-maker's own code, with no AI model."}

# Which models' work each kind of picture shows (registry ids; the manifest names them by their registry names).
PICTURE_MODELS = ("flux1-schnell",)
CUT_OUT_MODELS = ("flux1-schnell", "birefnet")
SHAPE_MODELS = ("flux1-schnell", "birefnet", "triposg")
OBJECT_MODELS = ("flux1-schnell", "birefnet", "triposg", "mv-adapter-ig2mv-sdxl", "sdxl-base-1.0", "sdxl-vae-fp16-fix")
# The report's pictures, by the start of their names (the longest start that fits): the picture and the tries, its
# cut-out, the painted views and the turntables show the models' work alone; the rest draw our marks over it.
REPORT_KINDS = {"picture-cut": (TRAINED, CUT_OUT_MODELS), "picture": (TRAINED, PICTURE_MODELS),
                "try-": (TRAINED, PICTURE_MODELS), "views-": (TRAINED, OBJECT_MODELS),
                "turntable": (TRAINED, OBJECT_MODELS), "bare": (TRAINED, SHAPE_MODELS),
                "outline": (COMPOSITE, SHAPE_MODELS)}
REPORT_OTHER = (COMPOSITE, OBJECT_MODELS)

KEY_VARIABLE = "MODEL_MAKER_SIGNING_KEY"
CERT_VARIABLE = "MODEL_MAKER_SIGNING_CERT"
TIMESTAMP_VARIABLE = "MODEL_MAKER_TIMESTAMP_URL"
TEST_FOLDER_VARIABLE = "MODEL_MAKER_TEST_CREDENTIALS"
TEST_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".tools",
                                            "c2pa-test"))
TEST_KEY, TEST_CHAIN = "signer.key", "signer-chain.pem"
# The test certificate's names: the same on every computer, so they say nothing about the one that made them.
TEST_ORGANISATION = "model-maker test certificate (untrusted)"
TEST_ROOT_NAME = "model-maker test root"
TEST_SIGNER_NAME = "model-maker test signer"
TEST_YEARS = 10
# A test certificate this close to its end is made again.
TEST_RENEW_DAYS = 30

MIME_TYPES = {".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
FORMATS = {".png": "PNG", ".webp": "WEBP", ".jpg": "JPEG", ".jpeg": "JPEG"}
# What a picture signed with a certificate no trust list holds is told by every checker: true, and not a fault here.
UNTRUSTED = "signingCredential.untrusted"
# The claim's thumbnail is left out: the picture is its own, and each would grow by a JPEG of itself.
CONTEXT_SETTINGS = {"builder": {"thumbnail": {"enabled": False}}}


class NotSigned(Exception):
    """A picture that could not be signed, or a signing key and certificate that do not work."""


@dataclasses.dataclass
class Signing:
    """The key and certificate chain that sign, in PEM files; `test`: the test certificate made on this computer."""
    key_path: str
    chain_path: str
    test: bool
    timestamp_url: str = None

    def __post_init__(self):
        self._signer = None
        self._algorithm = None

    @property
    def words(self):
        if self.test:
            return ("content credentials (C2PA): every picture is signed with a test certificate made on this computer; "
                    "checkers call it untrusted (docs/setup.md says how to sign with your own)")
        return "content credentials (C2PA): every picture is signed with the certificate given"

    def record(self):
        """What the sidecar says of the signing: the kind of certificate, never its files' places."""
        return {"standard": "C2PA", "certificate": "test" if self.test else "given",
                "timestamped": bool(self.timestamp_url)}

    def algorithm(self):
        if self._algorithm is None:
            self._algorithm = algorithm_of(read_bytes(self.key_path))
        return self._algorithm

    def signer(self):
        import c2pa

        if self._signer is None:
            try:
                info = c2pa.C2paSignerInfo(self.algorithm(), read_bytes(self.chain_path), read_bytes(self.key_path),
                                           self.timestamp_url)
                self._signer = c2pa.Signer.from_info(info)
            except c2pa.C2paError as error:
                raise NotSigned(f"the signing key and certificate do not work together: {error}") from None
        return self._signer


def read_bytes(path):
    with open(path, "rb") as file:
        return file.read()


def algorithm_of(key_pem):
    """The C2PA signing algorithm a PEM private key signs with."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

    try:
        key = serialization.load_pem_private_key(key_pem, password=None)
    except (TypeError, ValueError) as error:
        raise NotSigned(f"the signing key is not an unencrypted PEM private key: {error}") from None
    if isinstance(key, ec.EllipticCurvePrivateKey):
        by_curve = {"secp256r1": "es256", "secp384r1": "es384", "secp521r1": "es512"}
        if key.curve.name not in by_curve:
            raise NotSigned(f"the signing key's curve ({key.curve.name}) is not one C2PA signs with")
        return by_curve[key.curve.name]
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return "ed25519"
    if isinstance(key, rsa.RSAPrivateKey):
        return "ps256"
    raise NotSigned("the signing key is of a kind C2PA does not sign with (EC P-256/384/521, Ed25519 or RSA)")


def test_folder():
    return os.environ.get(TEST_FOLDER_VARIABLE) or TEST_FOLDER


def made_test_certificate(folder=None):
    """The test certificate's key and chain in `folder` (made if missing or near its end): a root of its own and a
    signer under it, the names the same on every computer. Only the user who made the key can read it."""
    folder = folder or test_folder()
    key_path, chain_path = os.path.join(folder, TEST_KEY), os.path.join(folder, TEST_CHAIN)
    if os.path.isfile(key_path) and os.path.isfile(chain_path) and not test_certificate_ending(chain_path):
        return key_path, chain_path
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    def name(common_name):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                          x509.NameAttribute(NameOID.ORGANIZATION_NAME, TEST_ORGANISATION)])

    def usage(signs_certificates):
        return x509.KeyUsage(digital_signature=not signs_certificates, content_commitment=False,
                             key_encipherment=False, data_encipherment=False, key_agreement=False,
                             key_cert_sign=signs_certificates, crl_sign=signs_certificates, encipher_only=False,
                             decipher_only=False)

    now = datetime.datetime.now(datetime.timezone.utc)
    start, end = now - datetime.timedelta(days=1), now + datetime.timedelta(days=365 * TEST_YEARS)
    root_key, key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    root = (x509.CertificateBuilder().subject_name(name(TEST_ROOT_NAME)).issuer_name(name(TEST_ROOT_NAME))
            .public_key(root_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(start).not_valid_after(end)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(usage(True), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), critical=False)
            .sign(root_key, hashes.SHA256()))
    # C2PA's certificate profile: an end certificate that signs (digitalSignature) for e-mail protection.
    signer = (x509.CertificateBuilder().subject_name(name(TEST_SIGNER_NAME)).issuer_name(root.subject)
              .public_key(key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(start).not_valid_after(end)
              .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
              .add_extension(usage(False), critical=True)
              .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
              .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()),
                             critical=False)
              .sign(root_key, hashes.SHA256()))
    os.makedirs(folder, mode=0o700, exist_ok=True)
    os.chmod(folder, stat.S_IRWXU)
    key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    descriptor = os.open(key_path + ".new", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(key_pem)
    with open(chain_path + ".new", "wb") as file:
        file.write(signer.public_bytes(serialization.Encoding.PEM) + root.public_bytes(serialization.Encoding.PEM))
    os.replace(key_path + ".new", key_path)
    os.replace(chain_path + ".new", chain_path)
    return key_path, chain_path


def test_certificate_ending(chain_path):
    """Whether the test certificate in `chain_path` ends within TEST_RENEW_DAYS (or cannot be read)."""
    from cryptography import x509

    try:
        first = x509.load_pem_x509_certificates(read_bytes(chain_path))[0]
    except (OSError, ValueError, IndexError):
        return True
    left = first.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)
    return left < datetime.timedelta(days=TEST_RENEW_DAYS)


def chosen(key=None, chain=None, timestamp_url=None, environment=None):
    """The signing to use: the key and certificate given (here, else by MODEL_MAKER_SIGNING_KEY and
    MODEL_MAKER_SIGNING_CERT), else the test certificate (made if there is none)."""
    environment = os.environ if environment is None else environment
    key = key or environment.get(KEY_VARIABLE)
    chain = chain or environment.get(CERT_VARIABLE)
    timestamp_url = timestamp_url or environment.get(TIMESTAMP_VARIABLE) or None
    if bool(key) != bool(chain):
        raise NotSigned("give both a signing key and its certificate (--signing-key and --signing-cert, or "
                        f"{KEY_VARIABLE} and {CERT_VARIABLE}), or neither for the test certificate")
    if key:
        for path in (key, chain):
            if not os.path.isfile(path):
                raise NotSigned(f"no such file: {path}")
        signing = Signing(key, chain, test=False, timestamp_url=timestamp_url)
        signing.algorithm()
        return signing
    key, chain = made_test_certificate()
    return Signing(key, chain, test=True, timestamp_url=timestamp_url)


_current = None


def use(signing):
    """Signs every picture written from now on with `signing` (the command line's choice)."""
    global _current
    _current = signing


def current():
    """The signing in use: the one given to `use`, else `chosen()`'s, kept for the process."""
    global _current
    if _current is None:
        _current = chosen()
    return _current


# The packages signing needs, by the name they are imported by and installed by (an update from 0.2.0 that did not
# run setup again has neither).
SIGNING_PACKAGES = {"c2pa": "c2pa-python", "cryptography": "cryptography"}


def missing_packages():
    """The signing packages that cannot be imported, by their install names."""
    import importlib.util

    return [package for module, package in SIGNING_PACKAGES.items() if importlib.util.find_spec(module) is None]


_version = None


def generator_version():
    global _version
    if _version is None:
        from .pipeline import repository_version

        _version = repository_version()
    return _version


def model_names(model_ids):
    from . import registry

    known = registry.models()
    # The registry's name up to any note after a comma ("TripoSG (shape), its weights run with our own code").
    return [known[model_id]["name"].split(", ")[0] for model_id in model_ids]


# A picture another tool made (tools/sign_pictures.py with a `generator`): said without naming model-maker's code.
OTHER_DESCRIPTIONS = {TRAINED: "Made by AI models.",
                      COMPOSITE: "Made from AI models' work, put together with other work."}


def manifest(source_type=TRAINED, model_ids=(), version=None, generator=None, models=None):
    """The manifest's definition: the generator, one `c2pa.created` action with its digital source type, and the
    models by name. Nothing about the prompt, the user or the machine. `generator` and `models` (names as given) for
    a picture another tool made: its name and version (`version`), and its models, which model-maker's registry
    need not know."""
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"not a digital source type model-maker writes: {source_type}")
    if generator and (not version or source_type not in OTHER_DESCRIPTIONS):
        raise ValueError("another tool's picture needs its version, and is made by AI models (ai or composite)")
    agent = {"name": generator or GENERATOR, "version": version or generator_version()}
    description = OTHER_DESCRIPTIONS[source_type] if generator else DESCRIPTIONS[source_type]
    action = {"action": "c2pa.created", "digitalSourceType": source_type, "softwareAgent": agent,
              "description": description}
    named = list(models) if models else model_names(model_ids) if model_ids else []
    if named:
        action["parameters"] = {"models": named}
    return {"claim_generator_info": [agent],
            "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [action]}}]}


def signed_bytes(data, mime_type, source_type=TRAINED, model_ids=(), signing=None, version=None, generator=None,
                 models=None):
    """`data` (a picture's file bytes) with a signed manifest in it; its pixels are not touched."""
    import c2pa

    signing = signing or current()
    context = c2pa.Context.from_dict(CONTEXT_SETTINGS)
    builder = c2pa.Builder.from_json(json.dumps(manifest(source_type, model_ids, version, generator, models)), context)
    out = io.BytesIO()
    try:
        builder.sign(signing.signer(), mime_type, io.BytesIO(data), out)
    except c2pa.C2paError as error:
        raise NotSigned(f"could not sign a {mime_type} picture: {error}") from None
    return out.getvalue()


def mime_type_of(path):
    ending = os.path.splitext(path)[1].lower()
    if ending not in MIME_TYPES:
        raise NotSigned(f"not a picture model-maker signs (PNG, WebP, JPEG): {os.path.basename(path)}")
    return MIME_TYPES[ending]


def write_whole(path, data):
    """`data` into `path` whole or not at all (a file beside it, then renamed over it)."""
    folder = os.path.dirname(os.path.abspath(path))
    partial = os.path.join(folder, f".{os.path.basename(path)}.signing")
    with open(partial, "wb") as file:
        file.write(data)
    os.replace(partial, path)


def save_picture(image, path, model_ids=(), source_type=TRAINED, signing=None):
    """A PIL picture written to `path` (its kind by its ending) with a signed manifest in it."""
    ending = os.path.splitext(path)[1].lower()
    mime_type = mime_type_of(path)
    plain = io.BytesIO()
    image.save(plain, format=FORMATS[ending])
    write_whole(path, signed_bytes(plain.getvalue(), mime_type, source_type, model_ids, signing))
    return path


def sign_file(path, model_ids=(), source_type=TRAINED, signing=None, version=None, generator=None, models=None):
    """A picture already written gets a signed manifest, its pixels as they are."""
    write_whole(path, signed_bytes(read_bytes(path), mime_type_of(path), source_type, model_ids, signing, version,
                                   generator, models))
    return path


@dataclasses.dataclass
class Reading:
    """What a picture's manifest says: `found` False when it has none."""
    found: bool
    state: str = None
    source_type: str = None
    generator: str = None
    version: str = None
    models: list = dataclasses.field(default_factory=list)
    failures: list = dataclasses.field(default_factory=list)
    issuer: str = None
    manifest: dict = None

    @property
    def marked(self):
        """A manifest that holds (signed, the picture unchanged since) and says what made the picture: an
        untrusted certificate (the test one) is no fault."""
        return (self.found and self.state in ("Valid", "Trusted") and self.source_type in SOURCE_TYPES
                and not [code for code in self.failures if code != UNTRUSTED])

    @property
    def trusted(self):
        return self.state == "Trusted"


def read(path=None, data=None, mime_type=None):
    """The manifest of a picture file (or of `data`, a picture's bytes, with its MIME type)."""
    import c2pa

    if data is None:
        data = read_bytes(path)
    mime_type = mime_type or mime_type_of(path)
    try:
        reader = c2pa.Reader(mime_type, io.BytesIO(data))
    except c2pa.C2paError as error:
        if "ManifestNotFound" in type(error).__name__ or "ManifestNotFound" in str(error):
            return Reading(found=False)
        return Reading(found=True, state="Invalid", failures=[str(error)])
    active = reader.get_active_manifest() or {}
    results = (reader.get_validation_results() or {}).get("activeManifest") or {}
    failures = [entry.get("code") for entry in results.get("failure", [])]
    created = [action for assertion in active.get("assertions", []) if assertion.get("label", "").startswith(
        "c2pa.actions") for action in assertion.get("data", {}).get("actions", []) if action.get("action") ==
        "c2pa.created"]
    action = created[0] if created else {}
    agent = action.get("softwareAgent") or {}
    if isinstance(agent, str):
        agent = {"name": agent}
    return Reading(found=True, state=reader.get_validation_state(), source_type=action.get("digitalSourceType"),
                   generator=agent.get("name"), version=agent.get("version"),
                   models=(action.get("parameters") or {}).get("models", []), failures=failures,
                   issuer=(active.get("signature_info") or {}).get("issuer"), manifest=active)


def report_picture_kind(name):
    """(digital source type, model ids) of a report picture, by its file name."""
    starts = sorted((start for start in REPORT_KINDS if name.startswith(start)), key=len, reverse=True)
    return REPORT_KINDS[starts[0]] if starts else REPORT_OTHER


def job_picture_kind(relative):
    """(digital source type, model ids) of a picture in a job's folder by its path under it; None for a file that is
    not one of the pictures model-maker writes there."""
    folder, _, name = relative.replace(os.sep, "/").rpartition("/")
    if not name.endswith(".png"):
        return None
    if folder == "pictures":
        return (TRAINED, CUT_OUT_MODELS) if name.endswith("-mask.png") else (TRAINED, PICTURE_MODELS)
    if folder == "" and name in ("picture.png", "mask.png"):
        return (TRAINED, PICTURE_MODELS) if name == "picture.png" else (TRAINED, CUT_OUT_MODELS)
    if folder == "views":
        # A touch-up's mask is where our own code asks the inpainter to paint: no model made it.
        if name.startswith("touch-up-") and name.endswith("-mask.png"):
            return (ALGORITHMIC, ())
        return (TRAINED, OBJECT_MODELS)
    if folder == "report":
        return report_picture_kind(name)
    return None


def job_pictures(job_folder):
    """Every picture model-maker writes in a job's folder: {path: (digital source type, model ids)}."""
    found = {}
    for root, _, names in os.walk(job_folder):
        for name in names:
            path = os.path.join(root, name)
            kind = job_picture_kind(os.path.relpath(path, job_folder))
            if kind is not None:
                found[path] = kind
    return found


def marked_pictures(pictures, root, signing=None):
    """Every picture of `pictures` ({path: (digital source type, model ids)}) carrying a manifest that holds: one
    without (made before model-maker signed its pictures) is signed now, its pixels as they are; one whose manifest no
    longer holds is reported (by its path under `root`), never signed over. Returns the record the sidecar and the
    report keep."""
    signing = signing or current()
    signed_now, broken, kinds = [], [], {}
    for path, (source_type, model_ids) in sorted(pictures.items()):
        reading = read(path)
        if not reading.found:
            sign_file(path, model_ids, source_type, signing)
            reading = read(path)
            signed_now.append(path)
        if not reading.marked:
            broken.append(path)
            continue
        kind = reading.source_type.rsplit("/", 1)[-1]
        kinds[kind] = kinds.get(kind, 0) + 1
    return {**signing.record(), "pictures": len(pictures), "marked": len(pictures) - len(broken),
            "signed_at_finish": len(signed_now), "digital_source_types": kinds,
            "not_marked": sorted(os.path.relpath(path, root).replace(os.sep, "/") for path in broken)}


def words_of(record):
    """The report's line on its pictures' content credentials."""
    if not record:
        return "Its pictures carry no content credentials: made before model-maker signed its pictures."
    whose = ("a test certificate made on the computer that made it, which checkers call untrusted"
             if record["certificate"] == "test" else "the maker's own certificate")
    said = (f"{record['marked']} of the {record['pictures']} pictures made for this object carry content credentials "
            f"(C2PA): a signed mark saying they were made by AI models, and by which; signed with {whose}.")
    if record["not_marked"]:
        said += f" {len(record['not_marked'])} do not: their marks no longer hold (changed since they were signed)."
    return said


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="model-maker credentials",
                                     description="The content credentials model-maker signs its pictures with.")
    parser.add_argument("--make-test", action="store_true", help="make the test certificate if there is none")
    parser.add_argument("--signing-key", help="a PEM private key (or MODEL_MAKER_SIGNING_KEY)")
    parser.add_argument("--signing-cert", help="its certificate chain, PEM (or MODEL_MAKER_SIGNING_CERT)")
    parser.add_argument("--read", nargs="+", metavar="PICTURE", help="what each picture's manifest says")
    args = parser.parse_args(argv)
    if args.read:
        for path in args.read:
            reading = read(path)
            if not reading.found:
                print(f"{path}: no content credentials")
                continue
            print(f"{path}: {'marked' if reading.marked else 'NOT HOLDING'} ({reading.state}); "
                  f"{reading.source_type}; by {reading.generator} {reading.version}; models: "
                  f"{', '.join(reading.models) or 'none'}; signed by {reading.issuer}"
                  f"{'; ' + ', '.join(reading.failures) if reading.failures else ''}")
        return 0
    try:
        if args.make_test:
            made_test_certificate()
        signing = chosen(args.signing_key, args.signing_cert)
        from PIL import Image

        trial = io.BytesIO()
        Image.new("RGB", (8, 8), (128, 128, 128)).save(trial, format="PNG")
        reading = read(data=signed_bytes(trial.getvalue(), "image/png", signing=signing, version="check"),
                       mime_type="image/png")
    except NotSigned as error:
        print(f"model-maker credentials: {error}")
        return 1
    print(signing.words)
    print(f"a picture signed now reads back {'marked' if reading.marked else 'NOT marked'} ({reading.state}; signed "
          f"by {reading.issuer})")
    return 0 if reading.marked else 1
