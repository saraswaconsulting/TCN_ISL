#!/usr/bin/env python3
"""
Enhanced keypoint extraction script with configurable class selection.
Supports extracting keypoints for specific subsets of classes based on YAML configuration.
"""

import os, glob, argparse, yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.text import Text
from rich.columns import Columns
from rich import box
from common import extract_sequence_from_video, make_class_index, save_json

# Initialize Rich console
console = Console()

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        console.print(f"[red]❌ Configuration file not found: {config_path}[/red]")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    console.print(f"[green]✅ Configuration loaded from: {config_path}[/green]")
    return config

def get_classes_in_folder_order(data_dir):
    """Get classes in the exact order they appear in the folder."""
    classes = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            classes.append(item)
    return classes

def filter_classes(all_classes_list, config):
    """
    Filter classes based on configuration settings, maintaining original folder order.
    
    Args:
        all_classes_list (list): List of all available classes in folder order
        config (dict): Configuration dictionary
    
    Returns:
        dict: Filtered classes dictionary with folder order preserved
    """
    mode = config.get('selection_mode', 'all')
    exclude_list = config.get('exclude_classes', [])
    
    # Remove excluded classes while maintaining order
    if exclude_list:
        all_classes_list = [c for c in all_classes_list if c not in exclude_list]
        console.print(f"[yellow]⚠️  Excluded classes: {exclude_list}[/yellow]")
    
    if mode == "all":
        selected_classes = all_classes_list
        console.print(f"[blue]🌍 Selection mode: ALL classes[/blue]")
    elif mode == "first_n":
        n = config.get('first_n_classes', 10)
        selected_classes = all_classes_list[:n]
        console.print(f"[blue]🔢 Selection mode: FIRST {n} classes[/blue]")
    elif mode == "specific":
        specific = config.get('specific_classes', [])
        # Maintain the order from the folder, but only include specified classes
        selected_classes = [c for c in all_classes_list if c in specific]
        missing = [c for c in specific if c not in all_classes_list]
        if missing:
            console.print(f"[yellow]⚠️  Specified classes not found: {missing}[/yellow]")
        console.print(f"[blue]🎯 Selection mode: SPECIFIC classes ({len(selected_classes)} out of {len(specific)} specified)[/blue]")
    elif mode == "range":
        range_config = config.get('class_range', {})
        start = range_config.get('start', 0)
        end = range_config.get('end', len(all_classes_list))
        selected_classes = all_classes_list[start:end]
        console.print(f"[blue]📊 Selection mode: RANGE from index {start} to {end-1}[/blue]")
    else:
        console.print(f"[red]❌ Unknown selection mode: {mode}[/red]")
        raise ValueError(f"Unknown selection mode: {mode}")
    
    # Create filtered class_to_idx mapping preserving folder order
    filtered_classes = {name: i for i, name in enumerate(selected_classes)}
    
    # Create summary table
    table = Table(title="Class Selection Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_row("Total Available", str(len(all_classes_list)))
    table.add_row("Selected", str(len(selected_classes)))
    table.add_row("Excluded", str(len(exclude_list)) if exclude_list else "0")
    
    console.print(table)
    
    # Show selected classes in columns
    if len(selected_classes) <= 20:  # Only show if reasonable number
        class_items = [Text(cls, style="green") for cls in selected_classes]
        columns = Columns(class_items, equal=True, expand=True)
        console.print(Panel(columns, title="Selected Classes (Folder Order)", border_style="green"))
    else:
        console.print(f"[green]Selected classes: {selected_classes[:5]}... and {len(selected_classes)-5} more[/green]")
    
    return filtered_classes

def extract_keypoints_for_classes(data_root, out_root, class_selection, config):
    """
    Extract keypoints for selected classes.
    
    Args:
        data_root (Path): Root directory containing train/val splits
        out_root (Path): Output directory for features
        class_selection (dict): Selected classes {class_name: index}
        config (dict): Configuration dictionary
    """
    extraction_config = config.get('extraction', {})
    fps = extraction_config.get('fps', 15.0)
    max_frames = extraction_config.get('max_frames', None)
    overwrite = extraction_config.get('overwrite', False)
    verbose = config.get('output', {}).get('verbose', True)
    
    total_videos_processed = 0
    split_results = {}
    
    # Create main progress for splits
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("[cyan]Processing splits...", total=2)
        
        for split in ["train", "val"]:
            split_dir = data_root / split
            if not split_dir.exists():
                console.print(f"[yellow]⚠️  Skip: {split_dir} (not found)[/yellow]")
                progress.advance(main_task)
                continue
            
            console.print(Panel(f"Processing {split.upper()} Split", style="bold blue"))
            
            videos_in_split = 0
            class_results = {}
            
            # Create progress for classes in this split
            class_task = progress.add_task(f"[green]Classes in {split}...", total=len(class_selection))
            
            for cls_name in class_selection:
                class_dir = split_dir / cls_name
                if not class_dir.exists():
                    if verbose:
                        console.print(f"[yellow]⚠️  Class directory not found: {class_dir}[/yellow]")
                    progress.advance(class_task)
                    continue
                    
                # Find all video files
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
                vids = []
                for ext in video_extensions:
                    vids.extend(glob.glob(str(class_dir / ext)))
                vids = sorted(vids)
                
                if not vids:
                    if verbose:
                        console.print(f"[yellow]⚠️  No video files found in {class_dir}[/yellow]")
                    progress.advance(class_task)
                    continue
                
                # Create output directory
                out_dir = out_root / split / cls_name
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Create progress for videos in this class
                video_task = progress.add_task(f"[magenta]{split}/{cls_name}", total=len(vids))
                
                processed_count = 0
                skipped_count = 0
                error_count = 0
                
                for vp in vids:
                    out_path = out_dir / (Path(vp).stem + ".npy")
                    
                    if out_path.exists() and not overwrite:
                        skipped_count += 1
                        progress.advance(video_task)
                        continue
                    
                    try:
                        seq = extract_sequence_from_video(
                            vp, 
                            target_fps=fps, 
                            max_frames=max_frames,
                            detection_conf=extraction_config.get('detection_confidence', 0.5),
                            tracking_conf=extraction_config.get('tracking_confidence', 0.5)
                        )
                        np.save(out_path, seq)
                        processed_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        if verbose:
                            console.print(f"[red]❌ Error processing {Path(vp).name}: {str(e)}[/red]")
                    
                    progress.advance(video_task)
                
                progress.remove_task(video_task)
                
                class_results[cls_name] = {
                    'processed': processed_count,
                    'skipped': skipped_count,
                    'errors': error_count,
                    'total': len(vids)
                }
                
                videos_in_split += processed_count
                progress.advance(class_task)
            
            progress.remove_task(class_task)
            split_results[split] = {
                'total_videos': videos_in_split,
                'classes': class_results
            }
            
            # Show split summary table
            split_table = Table(title=f"{split.upper()} Split Results", box=box.ROUNDED)
            split_table.add_column("Class", style="cyan")
            split_table.add_column("Processed", style="green")
            split_table.add_column("Skipped", style="yellow")
            split_table.add_column("Errors", style="red")
            split_table.add_column("Total", style="blue")
            
            for cls_name, results in class_results.items():
                split_table.add_row(
                    cls_name,
                    str(results['processed']),
                    str(results['skipped']),
                    str(results['errors']),
                    str(results['total'])
                )
            
            console.print(split_table)
            console.print(f"[bold green]✅ {split.upper()} Total: {videos_in_split} videos processed[/bold green]\n")
            
            total_videos_processed += videos_in_split
            progress.advance(main_task)
    
    # Final summary
    summary_panel = Panel(
        f"[bold green]✅ EXTRACTION COMPLETED[/bold green]\n\n"
        f"[cyan]Total videos processed:[/cyan] [bold]{total_videos_processed}[/bold]\n"
        f"[cyan]Classes processed:[/cyan] [bold]{len(class_selection)}[/bold]\n"
        f"[cyan]Output directory:[/cyan] [bold]{out_root}[/bold]",
        title="🎉 Summary",
        border_style="green"
    )
    console.print(summary_panel)

def save_class_mapping(class_selection, out_root, config):
    """Save the class mapping for selected classes."""
    output_config = config.get('output', {})
    create_filtered = output_config.get('create_filtered_class_map', True)
    
    if create_filtered:
        class_map_path = out_root / "class_to_idx.json"
        save_json(class_selection, class_map_path)
        
        # Create a nice table for class mapping
        mapping_table = Table(title="Class Mapping", box=box.ROUNDED)
        mapping_table.add_column("Class Name", style="cyan")
        mapping_table.add_column("Index", style="magenta")
        
        for cls_name, idx in sorted(class_selection.items(), key=lambda x: x[1]):
            mapping_table.add_row(cls_name, str(idx))
        
        console.print(mapping_table)
        console.print(f"[green]✅ Class mapping saved to: {class_map_path}[/green]")

def main():
    # Create header
    header = Panel(
        "[bold blue]🔍 ISL Keypoint Extractor[/bold blue]\n"
        "[italic]Enhanced extraction with configurable class selection[/italic]",
        title="🚀 Welcome",
        border_style="blue"
    )
    console.print(header)
    
    parser = argparse.ArgumentParser(
        description="Extract pose+hand features with configurable class selection"
    )
    parser.add_argument("--data_root", required=True, 
                       help="Data directory with train/ and val/ folders")
    parser.add_argument("--out_root", required=True, 
                       help="Output features/ directory")
    parser.add_argument("--config", default="extract_config.yaml",
                       help="YAML configuration file (default: extract_config.yaml)")
    
    # Override options (optional, will override config file)
    parser.add_argument("--fps", type=float, 
                       help="Target FPS (overrides config)")
    parser.add_argument("--max_frames", type=int,
                       help="Maximum frames per video (overrides config)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing files (overrides config)")
    parser.add_argument("--first_n", type=int,
                       help="Process first N classes (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Show configuration panel
    config_text = f"[cyan]Selection Mode:[/cyan] {config.get('selection_mode', 'all')}\n"
    if config.get('selection_mode') == 'first_n':
        config_text += f"[cyan]First N Classes:[/cyan] {config.get('first_n_classes', 10)}\n"
    elif config.get('selection_mode') == 'range':
        range_config = config.get('class_range', {})
        config_text += f"[cyan]Range:[/cyan] {range_config.get('start', 0)} to {range_config.get('end', 'end')}\n"
    config_text += f"[cyan]FPS:[/cyan] {config.get('extraction', {}).get('fps', 15.0)}\n"
    config_text += f"[cyan]Overwrite:[/cyan] {config.get('extraction', {}).get('overwrite', False)}"
    
    config_panel = Panel(config_text, title="⚙️ Configuration", border_style="cyan")
    console.print(config_panel)
    
    # Apply command line overrides
    if args.fps is not None:
        config.setdefault('extraction', {})['fps'] = args.fps
        console.print(f"[yellow]⚙️  Override: FPS set to {args.fps}[/yellow]")
    if args.max_frames is not None:
        config.setdefault('extraction', {})['max_frames'] = args.max_frames
        console.print(f"[yellow]⚙️  Override: Max frames set to {args.max_frames}[/yellow]")
    if args.overwrite:
        config.setdefault('extraction', {})['overwrite'] = True
        console.print(f"[yellow]⚙️  Override: Overwrite enabled[/yellow]")
    if args.first_n is not None:
        config['selection_mode'] = 'first_n'
        config['first_n_classes'] = args.first_n
        console.print(f"[yellow]⚙️  Override: Processing first {args.first_n} classes[/yellow]")
    
    # Setup paths
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    
    if not data_root.exists():
        console.print(f"[red]❌ Data directory not found: {data_root}[/red]")
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Show paths
    paths_table = Table(box=box.SIMPLE)
    paths_table.add_column("Path Type", style="cyan")
    paths_table.add_column("Location", style="green")
    paths_table.add_row("Data Root", str(data_root))
    paths_table.add_row("Output Root", str(out_root))
    console.print(paths_table)
    
    # Get all available classes from train directory in folder order
    train_dir = data_root / "train"
    if not train_dir.exists():
        console.print(f"[red]❌ Train directory not found: {train_dir}[/red]")
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    # Get classes in the exact order they appear in the folder
    all_classes_list = get_classes_in_folder_order(train_dir)
    console.print(f"[green]✅ Found {len(all_classes_list)} total classes in dataset (folder order)[/green]")
    
    # Filter classes based on configuration
    class_selection = filter_classes(all_classes_list, config)
    
    if not class_selection:
        console.print("[red]❌ No classes selected for processing. Check your configuration.[/red]")
        return
    
    # Extract keypoints
    console.print(Panel("[bold yellow]📝 Starting Keypoint Extraction[/bold yellow]", border_style="yellow"))
    extract_keypoints_for_classes(data_root, out_root, class_selection, config)
    
    # Save class mapping
    save_class_mapping(class_selection, out_root, config)
    
    # Final success message
    success_panel = Panel(
        "[bold green]🎉 Keypoint extraction completed successfully![/bold green]\n\n"
        f"[cyan]Next step:[/cyan] Train model with:\n"
        f"[yellow]python train.py --features_root {out_root}[/yellow]",
        title="✅ Success",
        border_style="green"
    )
    console.print(success_panel)

if __name__ == "__main__":
    main()