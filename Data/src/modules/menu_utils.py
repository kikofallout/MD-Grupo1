from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

def display_menu():
    """Displays the search menu and returns the user's choice."""
    console = Console()
    table = Table(title="[bold magenta]Search Menu[/bold magenta]", show_lines=True)
    table.add_column("Option", justify="center", style="cyan", no_wrap=True)
    table.add_column("Source", justify="center", style="green")
    
    sources = [
        ("1", "PubMed"),
        ("2", "Europe PMC"),
        ("3", "Semantic Scholar"),
        ("4", "Wikipedia"),
        ("5", "Google Scholar"),
        ("6", "EatRight"),
        ("7", "Dietary Guidelines"),
        ("8", "All Sources (except Wikipedia)")
    ]

    for option, source in sources:
        table.add_row(option, source)

    console.print(table)
    choice = Prompt.ask("[bold white]Enter the number of the source to search (or 'q' to quit)[/bold white]")
    return choice
