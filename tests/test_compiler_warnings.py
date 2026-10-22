import pytest

from cyber_threat_modelling.models.compiler_warnings import is_dynamic_global_cache_warning

MESSAGE = ('Cannot cache compiled function "elemwise" as it uses dynamic globals '
           '(such as ctypes pointers and large global arrays)')


@pytest.mark.parametrize(('message', 'category', 'expected'), [
    (MESSAGE, 'NumbaWarning', True),
    ('\x1b[1m\x1b[1m'+MESSAGE+'\x1b[0m\x1b[0m', 'NumbaWarning', True),
    (MESSAGE.replace('elemwise', 'specify_shape'), 'NumbaWarning', True),
    (MESSAGE, 'RuntimeWarning', False),
    (MESSAGE+'; nonfinite calculation', 'NumbaWarning', False),
    (MESSAGE.replace('dynamic globals', 'unsupported values'), 'NumbaWarning', False),
    (MESSAGE.replace('elemwise', 'elemwise\ninvalid'), 'NumbaWarning', False),
])
def test_cache_warning_classification_is_narrow(message, category, expected):
    assert is_dynamic_global_cache_warning(message, category) is expected
