import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.text import Text

from common import SeqDataset, GRUClassifier, load_json

# Initialize Rich console
console = Console()

def accuracy(logits, target):
    return (logits.argmax(1) == target).float().mean().item()

def create_training_table(epoch, total_epochs, train_metrics, val_metrics, best_val):
    """Create a training progress table."""
    table = Table(title=f"Training Progress - Epoch {epoch}/{total_epochs}", box=box.ROUNDED)
    table.add_column("Split", style="cyan")
    table.add_column("Loss", style="magenta")
    table.add_column("Accuracy", style="green")
    table.add_column("Status", style="yellow")
    
    # Add training row
    table.add_row(
        "Train", 
        f"{train_metrics['loss']:.4f}", 
        f"{train_metrics['acc']:.3%}",
        "📊 Training"
    )
    
    # Add validation row
    status = "🏆 Best!" if val_metrics['acc'] >= best_val else "✓ Done"
    table.add_row(
        "Validation", 
        f"{val_metrics['loss']:.4f}", 
        f"{val_metrics['acc']:.3%}",
        status
    )
    
    return table

def create_model_info_panel(model, device, num_classes, class_names):
    """Create model information panel."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info_text = (
        f"[cyan]Model:[/cyan] GRU Classifier\n"
        f"[cyan]Device:[/cyan] {device.upper()}\n"
        f"[cyan]Classes:[/cyan] {num_classes}\n"
        f"[cyan]Parameters:[/cyan] {total_params:,} (trainable: {trainable_params:,})\n"
        f"[cyan]Classes:[/cyan] {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}"
    )
    
    return Panel(info_text, title="🤖 Model Information", border_style="blue")

def main():
    # Create header
    header = Panel(
        "[bold blue]🚀 ISL GRU Classifier Training[/bold blue]\n"
        "[italic]Training pose-based sign language recognition model[/italic]",
        title="🎯 Training Pipeline",
        border_style="blue"
    )
    console.print(header)
    
    p = argparse.ArgumentParser("Train GRU on ISL pose sequences")
    p.add_argument("--features_root", required=True, help="features/ produced by extractor")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load class mapping
    class_to_idx = load_json(os.path.join(args.features_root, "class_to_idx.json"))
    num_classes = len(class_to_idx)
    class_names = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])
    
    console.print(f"[green]✅ Loaded class mapping with {num_classes} classes[/green]")
    
    # Create configuration table
    config_table = Table(title="Training Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="magenta")
    config_table.add_row("Device", device.upper())
    config_table.add_row("Classes", str(num_classes))
    config_table.add_row("Epochs", str(args.epochs))
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Learning Rate", f"{args.lr:.1e}")
    config_table.add_row("Max Length", str(args.max_len))
    config_table.add_row("Hidden Size", str(args.hidden))
    config_table.add_row("Layers", str(args.layers))
    config_table.add_row("Dropout", str(args.dropout))
    
    console.print(config_table)
    
    # Load datasets
    console.print("[yellow]📂 Loading datasets...[/yellow]")
    train_ds = SeqDataset(os.path.join(args.features_root, "train"), class_to_idx, max_len=args.max_len, train=True)
    val_ds   = SeqDataset(os.path.join(args.features_root, "val"),   class_to_idx, max_len=args.max_len, train=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Dataset info table
    dataset_table = Table(title="Dataset Information", box=box.ROUNDED)
    dataset_table.add_column("Split", style="cyan")
    dataset_table.add_column("Samples", style="magenta")
    dataset_table.add_column("Batches", style="green")
    dataset_table.add_row("Train", str(len(train_ds)), str(len(train_dl)))
    dataset_table.add_row("Validation", str(len(val_ds)), str(len(val_dl)))
    
    console.print(dataset_table)
    
    # Create model
    console.print("[yellow]🔧 Initializing model...[/yellow]")
    model = GRUClassifier(in_dim=150, hid=args.hidden, num_layers=args.layers,
                          num_classes=num_classes, dropout=args.dropout, bidir=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    # Show model info
    console.print(create_model_info_panel(model, device, num_classes, class_names))

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Training progress tracking
    best_val = 0.0
    training_history = []
    
    console.print(Panel("[bold green]📝 Starting Training[/bold green]", border_style="green"))
    
    # Training loop with Rich progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        epoch_task = progress.add_task("[cyan]Training epochs...", total=args.epochs)
        
        for epoch in range(1, args.epochs+1):
            # Training phase
            model.train()
            tr_loss, tr_acc_sum, n = 0.0, 0.0, 0
            
            train_task = progress.add_task(f"[green]Epoch {epoch} - Training", total=len(train_dl))
            
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
                bs = y.size(0)
                tr_loss += loss.item() * bs
                tr_acc_sum += accuracy(logits.detach(), y) * bs
                n += bs
                progress.advance(train_task)
                
            progress.remove_task(train_task)
            tr_loss /= max(1, n)
            tr_acc = tr_acc_sum / max(1, n)

            # Validation phase
            model.eval()
            va_loss, va_acc_sum, n = 0.0, 0.0, 0
            
            val_task = progress.add_task(f"[blue]Epoch {epoch} - Validation", total=len(val_dl))
            
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = crit(logits, y)
                    bs = y.size(0)
                    va_loss += loss.item() * bs
                    va_acc_sum += accuracy(logits, y) * bs
                    n += bs
                    progress.advance(val_task)
                    
            progress.remove_task(val_task)
            va_loss /= max(1, n)
            va_acc = va_acc_sum / max(1, n)
            
            # Store metrics
            train_metrics = {'loss': tr_loss, 'acc': tr_acc}
            val_metrics = {'loss': va_loss, 'acc': va_acc}
            training_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Display results
            table = create_training_table(epoch, args.epochs, train_metrics, val_metrics, best_val)
            console.print(table)

            # Save best model
            if va_acc > best_val:
                best_val = va_acc
                torch.save({
                    "model": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "args": args,
                    "training_history": training_history
                }, os.path.join(args.out_dir, "best_gru.pt"))
                console.print(f"[bold green]✨ New best model saved! Validation accuracy: {best_val:.3%}[/bold green]")
            
            progress.advance(epoch_task)
    
    # Final summary
    final_summary = Panel(
        f"[bold green]🏆 Training Completed![/bold green]\n\n"
        f"[cyan]Best Validation Accuracy:[/cyan] [bold]{best_val:.3%}[/bold]\n"
        f"[cyan]Total Epochs:[/cyan] {args.epochs}\n"
        f"[cyan]Model saved to:[/cyan] {os.path.join(args.out_dir, 'best_gru.pt')}\n\n"
        f"[yellow]Next steps:[/yellow]\n"
        f"[white]1. Evaluate: python eval.py --features_root {args.features_root} --checkpoint {os.path.join(args.out_dir, 'best_gru.pt')}[/white]\n"
        f"[white]2. Test single video: python predict.py --video path/to/video.mp4 --checkpoint {os.path.join(args.out_dir, 'best_gru.pt')}[/white]",
        title="✅ Success",
        border_style="green"
    )
    console.print(final_summary)

if __name__ == "__main__":
    main()
