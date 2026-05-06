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

from . import __version__, api, config, daemon, gpu


@click.group(help="TRD Compute Network — community GPU worker")
@click.version_option(__version__, prog_name="trd-worker")
def cli() -> None:
    pass


@cli.command(help="Register this machine with TRD Compute Network")
@click.option("--email", prompt="Waitlist email", help="Email used for the compute.trdn.io waitlist")
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

    supported = gpu.suggest_supported_models(info)
    click.echo(f"   Models:  {', '.join(supported)}")

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


@cli.command(help="Wipe local config (note: does NOT unregister with backend)")
def logout() -> None:
    if not config.is_logged_in():
        click.echo("Not logged in.")
        return
    if not click.confirm("Wipe saved token? You'll need to log in again."):
        return
    config.clear()
    click.echo("✓ Local config cleared.")


@cli.command(help="Print version")
def version() -> None:
    click.echo(__version__)


if __name__ == "__main__":
    cli()
