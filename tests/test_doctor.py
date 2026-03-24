from __future__ import annotations

from npu_model.cli.doctor import (
    CheckResult,
    _check_genai_builder,
    _check_package,
    _probe_module_import,
    run_doctor,
)


def test_doctor_runs() -> None:
    checks = run_doctor()
    assert isinstance(checks, list)
    assert len(checks) > 0
    assert all(isinstance(c, CheckResult) for c in checks)


def test_doctor_python_ok() -> None:
    checks = run_doctor()
    python_check = next(c for c in checks if c.name == "Python")
    assert python_check.ok is True


def test_doctor_core_deps_ok() -> None:
    checks = run_doctor()
    for name in ("typer", "rich", "huggingface_hub"):
        check = next(c for c in checks if c.name == name)
        assert check.ok is True, f"{name} should be installed in dev env"


def test_doctor_registry_loads() -> None:
    checks = run_doctor()
    reg_checks = [c for c in checks if c.name.startswith("Registry:")]
    assert len(reg_checks) > 0
    for c in reg_checks:
        assert c.ok is True


def test_probe_module_import_handles_failure() -> None:
    check = _probe_module_import(
        "module_that_does_not_exist_xyz",
        "missing import probe",
        "install it",
    )
    assert isinstance(check, CheckResult)
    assert check.ok is False
    assert "import failed" in check.detail


def test_check_package_uses_metadata_lookup() -> None:
    # This should not try importing the module object itself.
    check = _check_package("rich", "rich", dist_name="rich")
    assert isinstance(check, CheckResult)
    assert check.name == "rich"
    assert check.ok is True


def test_check_genai_builder_never_raises() -> None:
    check = _check_genai_builder()
    assert isinstance(check, CheckResult)
