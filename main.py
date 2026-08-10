from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from model.config import TranslationConfig, device
from model.inference import generate_translation, load_model_for_inference
from model.transformer import EncoderDecoderTransfomer

console = Console()


def load_model():
    path = "checkpoints/best_checkpoint_translation.pt"
    config = TranslationConfig()

    with console.status("[bold cyan]Loading model...", spinner="dots"):
        model = load_model_for_inference(path, EncoderDecoderTransfomer, config, device)

    console.print("[bold green]✓[/bold green] Model loaded\n")
    return model


def translate_once(model, prompt: str) -> str:
    with console.status("[bold cyan]Translating...", spinner="dots"):
        translation = generate_translation(model, prompt)
    return translation


def main():
    console.print(
        Panel.fit(
            "[bold]English → German Translator[/bold]\n"
            "[dim]Multi30k-trained transformer[/dim]",
            border_style="cyan",
        )
    )
    console.print(
        "[dim]Tip: this model was trained on short image-caption style sentences "
        '(e.g. "A man is standing on a bridge") — it works best on similar phrasing.[/dim]\n'
    )

    model = load_model()

    while True:
        prompt = Prompt.ask("[bold yellow]EN[/bold yellow]")

        if prompt.strip().lower() in {"exit", "quit", "q"}:
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not prompt.strip():
            console.print("[red]Please enter a sentence.[/red]\n")
            continue

        translation = translate_once(model, prompt)

        console.print(f"[bold green]DE[/bold green] {translation}\n")


if __name__ == "__main__":
    main()
