import click

from .enhanced_commands import register_commands


@click.group()
def cli():
    """TrackRealties AI Platform CLI"""
    pass


register_commands(cli)

if __name__ == "__main__":
    cli()
