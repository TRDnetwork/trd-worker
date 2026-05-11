"""
trd-worker CLI entry point.

Commands:
  trd-worker login     — detect GPU, register with backend, save token
  trd-worker start     — run daemon (heartbeat + poll + execute + submit)
  trd-worker status    — show config + last heartbeat
  trd-worker logout    — wipe local config (does not unregister on backend)
  trd-worker version   — print version
"""

from __future__ import annotations
import platform
import socket
import sys

import click

from . import __version__, api, config, daemon, doctor as doctor_module, gpu


@click.group(help="TRD Compute Network — community GPU worker")
@click.version_option(__version__, prog_name="trd-worker")
def cli() -> None:
    pass


@cli.command(help="Register this machine with TRD Compute Network")
@click.option("--email", prompt="Email", help="Email used for compute.trdn.io")
@click.option("--hostname", default=None, help="Custom worker name (defaults to system hostname)")
@click.option(
    "--api-base",
    default=None,
    help="Override backend URL (else uses TRD_CN_API env or production default)",
)
def login(email: str, hostname: str | None, api_base: str | None) -> None:
    if config.is_logged_in():
        if not click.confirm("⚠ Already logged in. Re-register and overwrite saved token?"):
            click.echo("Aborted.")
            return

    click.echo("🔍 Detecting GPU...")
    info = gpu.detect()
    if not info:
        click.echo(
            "✗ No supported GPU detected (need NVIDIA / Apple Silicon / AMD).",
            err=True,
        )
        sys.exit(1)

    click.echo(f"   Vendor:  {info.vendor}")
    click.echo(f"   Model:   {info.model}")
    click.echo(f"   VRAM:    {info.vram_gb} GB")
    if info.driver_version:
        click.echo(f"   Driver:  {info.driver_version}")
    if info.cuda_version:
        click.echo(f"   CUDA:    {info.cuda_version}")

    # Models the worker should advertise = intersection of (VRAM-tier suggestion)
    # and (actually downloaded GGUF files). Phase 3+: workers only claim jobs
    # for models they can actually execute.
    from . import models as model_mod
    suggested = gpu.suggest_supported_models(info)
    local_names = {m.name for m in model_mod.list_local()}
    # Match suggested names + their aliases against the local registry
    supported: list[str] = []
    for name in suggested:
        spec = model_mod._by_name(name)
        if spec and spec.name in local_names:
            supported.append(spec.name)
    # De-dupe while preserving order
    seen: set[str] = set()
    supported = [s for s in supported if not (s in seen or seen.add(s))]

    if supported:
        click.echo(f"   Models:  {', '.join(supported)}")
    else:
        click.echo("   Models:  (none downloaded)", err=False)
        click.echo(
            "\n⚠ No supported models are downloaded yet. The worker would register "
            "but never receive jobs.",
            err=True,
        )
        click.echo(
            "  Run `trd-worker models list` to see what's available, then "
            "`trd-worker models pull <name>` to download.",
            err=True,
        )
        click.echo(
            "  (You can also re-run `trd-worker login` after downloading models "
            "to register the updated capability set.)",
            err=True,
        )
        if not click.confirm("\nRegister anyway?", default=False):
            click.echo("Aborted.")
            return

    if not hostname:
        hostname = socket.gethostname() or "unknown-host"
    click.echo(f"   Host:    {hostname}")

    if api_base:
        click.echo(f"   Backend: {api_base} (override)")

    if not click.confirm("\nRegister this worker?", default=True):
        click.echo("Aborted.")
        return

    # Persist api_base override before the call so api module picks it up
    if api_base:
        cfg = config.load()
        cfg["api_base"] = api_base
        config.save(cfg)

    click.echo("📡 Registering...")
    try:
        result = api.register(
            email=email.strip().lower(),
            hostname=hostname,
            gpu_vendor=info.vendor,
            gpu_model=info.model,
            gpu_vram_gb=info.vram_gb,
            supported_models=supported,
            cli_version=__version__,
            cuda_version=info.cuda_version,
            driver_version=info.driver_version,
            os=gpu.os_string(),
        )
    except api.ApiError as e:
        click.echo(f"✗ Registration failed: {e}", err=True)
        if e.status == 403:
            click.echo("   (Sign up at https://compute.trdn.io first if you haven't.)", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Network error: {e}", err=True)
        sys.exit(1)

    cfg: config.Config = {
        "api_base": api_base or config.api_base(),
        "worker_id": result.worker_id,
        "auth_token": result.auth_token,
        "email": email.strip().lower(),
        "hostname": hostname,
        "base_credits_per_hour": result.base_rate_credits_per_hour,
    }
    config.save(cfg)

    click.echo("")
    click.echo(f"✅ Registered. Worker ID: {result.worker_id}")
    click.echo(f"   Rate: {result.base_rate_credits_per_hour} credits/hour")
    click.echo(f"   Token saved to {config.CONFIG_FILE}")
    click.echo("")
    click.echo("Run `trd-worker start` to begin earning.")


@cli.command(help="Run the worker daemon (heartbeat + poll + execute jobs)")
def start() -> None:
    sys.exit(daemon.main())


@cli.command(help="Show local config and login state")
def status() -> None:
    if not config.is_logged_in():
        click.echo("✗ Not logged in. Run `trd-worker login` first.")
        sys.exit(1)
    cfg = config.load()
    click.echo(f"  Email:    {cfg.get('email', '?')}")
    click.echo(f"  Worker:   {cfg.get('worker_id', '?')}")
    click.echo(f"  Hostname: {cfg.get('hostname', '?')}")
    click.echo(f"  Rate:     {cfg.get('base_credits_per_hour', '?')} credits/hour")
    click.echo(f"  Backend:  {config.api_base()}")
    click.echo(f"  Config:   {config.CONFIG_FILE}")


@cli.command(help="Re-advertise supported models after pulling/removing GGUF files")
def refresh() -> None:
    if not config.is_logged_in():
        click.echo("✗ Not logged in. Run `trd-worker login` first.", err=True)
        sys.exit(1)

    cfg = config.load()
    token = cfg.get("auth_token")
    if not token:
        click.echo("✗ No token in config.", err=True)
        sys.exit(1)

    # Recompute supported set: VRAM-tier ∩ locally-downloaded
    from . import models as model_mod
    info = gpu.detect()
    if not info:
        click.echo("✗ Could not detect GPU.", err=True)
        sys.exit(1)
    suggested = gpu.suggest_supported_models(info)
    local_names = {m.name for m in model_mod.list_local()}
    supported: list[str] = []
    seen: set[str] = set()
    for name in suggested:
        spec = model_mod._by_name(name)
        if spec and spec.name in local_names and spec.name not in seen:
            supported.append(spec.name)
            seen.add(spec.name)

    click.echo(f"Current GPU:  {info.vendor} {info.model} ({info.vram_gb}GB)")
    click.echo(f"Will advertise: {', '.join(supported) if supported else '(none — no models downloaded)'}")

    if not supported:
        click.echo(
            "\n⚠ No downloaded models match this GPU's tier. "
            "The worker will register but never receive jobs.",
            err=True,
        )
        if not click.confirm("Submit empty list anyway?", default=False):
            click.echo("Aborted.")
            return

    try:
        accepted = api.update_capabilities(token, supported)
    except api.ApiError as e:
        if e.status in (401, 403):
            click.echo(
                "✗ Auth rejected. Token may be revoked. "
                "Run `trd-worker login` to register fresh.",
                err=True,
            )
        else:
            click.echo(f"✗ Update failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Network error: {e}", err=True)
        sys.exit(1)

    click.echo(f"\n✓ Updated. Server now advertises: {', '.join(accepted) if accepted else '(empty)'}")


@cli.command(help="Wipe local config and optionally revoke token on the server")
@click.option(
    "--revoke/--no-revoke",
    default=None,
    help="Also revoke token server-side (default: prompt)",
)
def logout(revoke: bool | None) -> None:
    if not config.is_logged_in():
        click.echo("Not logged in.")
        return

    cfg = config.load()
    token = cfg.get("auth_token")

    if revoke is None:
        revoke = click.confirm(
            "Also revoke this token on the server? (recommended if you suspect leak)",
            default=True,
        )

    if revoke and token:
        try:
            api.revoke(token)
            click.echo("✓ Token revoked server-side.")
        except api.ApiError as e:
            if e.status in (401, 403):
                click.echo("(Token was already invalid/banned — nothing to revoke.)")
            else:
                click.echo(f"⚠ Server revoke failed: {e}", err=True)
        except Exception as e:
            click.echo(f"⚠ Server revoke network error: {e}", err=True)

    if not click.confirm("Wipe local config?", default=True):
        return
    config.clear()
    click.echo("✓ Local config cleared.")


@cli.command(help="Show effective worker settings (fetched from backend) — useful for debugging why a worker is paused")
def settings() -> None:
    if not config.is_logged_in():
        click.echo("✗ Not logged in. Run `trd-worker login` first.", err=True)
        sys.exit(1)

    cfg = config.load()
    token = cfg.get("auth_token")
    if not token:
        click.echo("✗ No token in config.", err=True)
        sys.exit(1)

    from . import settings as settings_mod, power as power_mod
    settings_mod.maybe_refresh(token, __version__)
    s = settings_mod.get()

    click.echo("Worker settings (effective):")
    click.echo(f"  Max GPU util:     {s.max_gpu_utilization_pct}%")
    click.echo(f"  Schedule:         " + (
        f"{s.schedule_start_hour:02d}:00–{s.schedule_end_hour:02d}:00 (local time)"
        if s.schedule_enabled and s.schedule_start_hour is not None and s.schedule_end_hour is not None
        else "disabled (always on)"
    ))
    click.echo(f"  Allowed models:   " + (
        ", ".join(s.allowed_models) if s.allowed_models else "(all supported)"
    ))
    click.echo(f"  Pause on battery: {'yes' if s.pause_when_on_battery else 'no'}")
    click.echo("")

    state = power_mod.is_on_battery()
    power_label = (
        "on battery" if state is True
        else "on AC power" if state is False
        else "indeterminate (no battery / desktop)"
    )
    click.echo(f"Current power state: {power_label}")

    accept, reason = settings_mod.should_accept_jobs()
    if accept:
        click.echo("Status: ✅ would accept jobs right now")
    else:
        click.echo(f"Status: ⏸  paused — {reason}")


@cli.command(help="Print version")
def version() -> None:
    click.echo(__version__)


# ── trd-worker models ───────────────────────────────────────────────────────
@cli.group(help="Manage local model files (GGUF)")
def models() -> None:
    pass


@models.command("list", help="Show available + locally-downloaded models")
def models_list() -> None:
    from . import models as model_mod
    available = model_mod.list_available()
    if not available:
        click.echo("No models in registry.")
        return
    click.echo(f"{'NAME':<28}  {'SIZE':>7}  {'VRAM':>6}  {'STATUS':<12}  DESCRIPTION")
    click.echo("─" * 88)
    for spec in available:
        local = "✓ downloaded" if model_mod.is_downloaded(spec.name) else "  not pulled"
        click.echo(
            f"{spec.name:<28}  "
            f"{spec.file_size_gb:>5.1f}GB  "
            f"{spec.min_vram_gb:>4}GB  "
            f"{local:<12}  "
            f"{spec.display_name}"
        )


@models.command("pull", help="Download a model GGUF to ~/.trd-worker/models/")
@click.argument("name")
@click.option("--force", is_flag=True, help="Re-download even if cached")
def models_pull(name: str, force: bool) -> None:
    from . import models as model_mod
    spec = model_mod._by_name(name)
    if not spec:
        click.echo(f"✗ Unknown model '{name}'.", err=True)
        click.echo("Available:")
        for m in model_mod.list_available():
            click.echo(f"  {m.name}")
        raise SystemExit(1)

    if model_mod.is_downloaded(spec.name) and not force:
        path = model_mod.model_path(spec.name)
        click.echo(f"✓ Already downloaded: {path}")
        click.echo("  (Use --force to re-download)")
        return

    try:
        path = model_mod.download_model(spec.name, progress=True, force=force)
        click.echo(f"\n✓ Ready: {path}")
    except Exception as e:
        click.echo(f"\n✗ Download failed: {e}", err=True)
        raise SystemExit(1)


@models.command("where", help="Print the local cache directory")
def models_where() -> None:
    from . import models as model_mod
    click.echo(str(model_mod.MODELS_DIR))


@models.command("rm", help="Delete a downloaded model from disk")
@click.argument("name")
def models_rm(name: str) -> None:
    from . import models as model_mod
    spec = model_mod._by_name(name)
    if not spec:
        click.echo(f"✗ Unknown model '{name}'.", err=True)
        raise SystemExit(1)
    if not model_mod.is_downloaded(spec.name):
        click.echo(f"Not downloaded: {spec.name}")
        return
    path = model_mod.model_path(spec.name)
    if not click.confirm(f"Delete {path}?"):
        return
    path.unlink()
    # Also remove parent dir if empty
    try:
        path.parent.rmdir()
    except OSError:
        pass
    click.echo(f"✓ Deleted {path}")



@cli.command("doctor", help="Run end-to-end environment diagnostic (8 checks)")
@click.option("--json", "json_output", is_flag=True, help="Output JSON instead of human-readable")
def doctor(json_output: bool) -> None:
    """Wrap doctor_module.cmd_doctor in a click command named 'doctor'."""
    class _Args:
        pass
    a = _Args()
    a.json = json_output
    doctor_module.cmd_doctor(a)


if __name__ == "__main__":
    cli()
