"""The upstream AST signature gate must go red for a real API regression."""

from upstream_signature_gate import signature_violations


def test_added_fork_kwargs_do_not_break_existing_upstream_interface():
    upstream = '''
def train_family(dit, *, training_maps=False, training_maps_images=3):
    return dit
class Handler:
    def handle(self, path, *, strict=True): pass
'''
    fork = '''
def train_family(dit, *, training_maps=False, training_maps_images=3, recaption_replace_rules=''):
    return dit
class Handler:
    def handle(self, path, *, strict=True, tracing=False): pass
'''
    assert signature_violations(upstream, fork) == []


def test_removed_upstream_training_maps_signature_fails():
    upstream = 'def train_family(*, training_maps=False, training_maps_images=3): pass'
    fork = 'def train_family(*, recaption_replace_rules=""): pass'
    failures = signature_violations(upstream, fork)
    assert 'train_family: upstream keyword parameter lost: training_maps' in failures
    assert 'train_family: upstream keyword parameter lost: training_maps_images' in failures


def test_removal_of_upstream_function_and_method_fails():
    upstream = 'def h3_loader(): pass\nclass Driver:\n    def load(self, path): pass'
    fork = 'class Driver:\n    pass'
    failures = signature_violations(upstream, fork)
    assert 'h3_loader: upstream callable missing from fork' in failures
    assert 'Driver.load: upstream callable missing from fork' in failures


def test_fork_var_kwargs_can_handle_new_upstream_optional_keywords():
    upstream = 'def fetch_weight(w, *, token=None): pass'
    fork = 'def fetch_weight(w, **kwargs): pass'
    assert signature_violations(upstream, fork) == []


def test_removed_default_fails_for_positional_and_keyword_only():
    upstream = 'def load(path, strict=True, *, device="cpu"): pass'
    fork = 'def load(path, strict, *, device): pass'
    failures = signature_violations(upstream, fork)
    assert 'load: upstream optional parameter made required: strict' in failures
    assert 'load: upstream optional parameter made required: device' in failures


def test_positional_parameter_made_keyword_only_fails():
    upstream = 'def save(model, path): pass'
    fork = 'def save(model, *, path): pass'
    assert signature_violations(upstream, fork) == ['save: upstream positional parameter 2 (path) no longer accepted by position']


def test_reordered_positional_parameters_fail():
    upstream = 'def blend(base, overlay): pass'
    fork = 'def blend(overlay, base): pass'
    failures = signature_violations(upstream, fork)
    assert 'blend: upstream positional parameter 1 (base) is now overlay' in failures
    assert 'blend: upstream positional parameter 2 (overlay) is now base' in failures


def test_positional_or_keyword_made_positional_only_fails():
    upstream = 'def scale(value, factor): pass'
    fork = 'def scale(value, factor, /): pass'
    assert 'scale: upstream keyword parameter lost: factor' in signature_violations(upstream, fork)


def test_removed_var_keyword_and_var_positional_fail():
    upstream = 'def run(job, *args, **kwargs): pass'
    fork = 'def run(job): pass'
    failures = signature_violations(upstream, fork)
    assert 'run: upstream *args removed' in failures
    assert 'run: upstream **kwargs removed' in failures


def test_fork_adds_required_parameter_fails():
    upstream = 'def encode(image): pass\ndef decode(latent, *, size=512): pass'
    fork = 'def encode(image, vae): pass\ndef decode(latent, *, size=512, dtype): pass'
    failures = signature_violations(upstream, fork)
    assert 'encode: fork adds required parameter: vae' in failures
    assert 'decode: fork adds required parameter: dtype' in failures


def test_compatible_widening_passes():
    upstream = 'def fit(data, epochs=1, /, *, lr=0.1): pass'
    fork = 'def fit(data, epochs=1, warmup=0, /, *args, lr=0.1, decay=None, **kwargs): pass'
    assert signature_violations(upstream, fork) == []
