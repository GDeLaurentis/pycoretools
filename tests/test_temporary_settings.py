import pycoretools

from pycoretools import TemporarySetting

SETTING_FOR_TEST = "something"
pycoretools.USE_ELLIPSIS_FOR_PRINT = False


def test_module_temp_setting():
    with TemporarySetting(pycoretools, "USE_ELLIPSIS_FOR_PRINT", True):
        assert pycoretools.USE_ELLIPSIS_FOR_PRINT is True
    with TemporarySetting(pycoretools, "USE_ELLIPSIS_FOR_PRINT", False):
        assert pycoretools.USE_ELLIPSIS_FOR_PRINT is False


def test_module_as_str_temp_setting():
    with TemporarySetting("pycoretools", "USE_ELLIPSIS_FOR_PRINT", True):
        assert pycoretools.USE_ELLIPSIS_FOR_PRINT is True
    with TemporarySetting("pycoretools", "USE_ELLIPSIS_FOR_PRINT", False):
        assert pycoretools.USE_ELLIPSIS_FOR_PRINT is False


def test_global_temp_setting():
    with TemporarySetting(globals(), "SETTING_FOR_TEST", True):
        assert SETTING_FOR_TEST is True
    with TemporarySetting(globals(), "SETTING_FOR_TEST", "something else"):
        assert SETTING_FOR_TEST == "something else"
