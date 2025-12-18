#!/usr/bin/env python3
"""
Extract specific frames from an animated GIF.

Usage:
    python extract_gif_frames.py quantum_demo.gif --steps 0 500 1000 1499
    python extract_gif_frames.py quantum_demo.gif --steps 10 500 1000 1450 --output frames/
"""

import argparse
from pathlib import Path
from PIL import Image


def extract_frames(gif_path, target_steps, output_dir=None, prefix="frame_"):
    """
    Extract specific frames from a GIF animation.
    
    Args:
        gif_path: Path to input GIF file
        target_steps: List of step numbers (0-indexed) to extract
        output_dir: Directory to save frames (default: same as GIF)
        prefix: Prefix for output filenames
        
    Returns:
        List of paths to extracted frames
    """
    gif_path = Path(gif_path)
    
    if output_dir is None:
        output_dir = gif_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_frames = []
    
    with Image.open(gif_path) as im:
        # Count total frames
        n_frames = 0
        try:
            while True:
                im.seek(n_frames)
                n_frames += 1
        except EOFError:
            pass
        
        print(f"Total frames in GIF: {n_frames}")
        
        # Extract requested frames
        for step in sorted(target_steps):
            # Convert step to 0-indexed frame number
            idx = step
            
            if idx < 0 or idx >= n_frames:
                print(f"⚠ Step {step} out of range (0-{n_frames-1}), skipping")
                continue
            
            # Seek to the frame and decode it properly
            im.seek(idx)
            
            # Convert to RGB for consistent PNG output
            frame = im.convert('RGB')
            
            # Save frame
            output_file = output_dir / f"{prefix}{step}.png"
            frame.save(output_file)
            
            extracted_frames.append(output_file)
            print(f"✓ Extracted Step {step} (frame {idx}) -> {output_file}")
    
    return extracted_frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific frames from an animated GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames at steps 0, 500, 1000, 1499
  python extract_gif_frames.py demo.gif --steps 0 500 1000 1499
  
  # Save to specific directory with custom prefix
  python extract_gif_frames.py demo.gif --steps 10 100 500 --output frames/ --prefix snapshot_
        """
    )
    
    parser.add_argument('gif_path', type=str,
                       help='Path to input GIF file')
    parser.add_argument('--steps', type=int, nargs='+', required=True,
                       help='Frame numbers to extract (0-indexed)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory (default: same as input GIF)')
    parser.add_argument('--prefix', '-p', type=str, default='frame_',
                       help='Filename prefix for extracted frames (default: frame_)')
    
    args = parser.parse_args()
    
    # Extract frames
    extracted = extract_frames(
        args.gif_path,
        args.steps,
        args.output,
        args.prefix
    )
    
    print(f"\n✓ Successfully extracted {len(extracted)} frames")


if __name__ == '__main__':
    main()
