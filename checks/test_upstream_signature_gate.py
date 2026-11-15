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
