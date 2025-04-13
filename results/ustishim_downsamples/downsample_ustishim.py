#!/usr/bin/env python3

"""
This module runs downsampling on the ustishim 23andme data at various percentages.
It creates both diploid and pseudo-haploid versions of the downsampled data
and collects statistics about the results.
"""
import os
import re
import subprocess
import shutil
import sys
import polars as pl

def run_downsampling(input_file, script_path, pct, pseudo_haploid=False):
    """Run the downsampling command with the specified parameters."""
    cmd = [
        "python", script_path,
        "-i", input_file,
        "-p", str(pct),
        "-s"  # Include stats
    ]

    if pseudo_haploid:
        cmd.append("-a")  # Enable pseudo-haploid mode
        mode_name = "pseudo-haploid"
    else:
        mode_name = "diploid"

    try:
        process = subprocess.run(
            cmd, check=True, text=True, capture_output=True
        )
        print(f"{mode_name.capitalize()} processing complete.")

        # Parse stats from output
        section_name = "Pseudo-haploid stats" if pseudo_haploid else "Downsampled stats"
        total_loci, missing_loci = extract_stats(process.stdout, section_name)

        # Create result dictionary
        result = {
            f'{mode_name.capitalize()} Total Loci': total_loci,
            f'{mode_name.capitalize()} Remaining Loci': total_loci - missing_loci,
            f'{mode_name.capitalize()} Missing Loci': missing_loci,
            f'{mode_name.capitalize()} Actual Missingness %': (missing_loci * 100 / total_loci
                                                                if total_loci > 0 else 0)
        }

        # Handle file copying
        handle_output_files(process.stdout, "./")

        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running {mode_name} downsampling with {pct}% missingness:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {e}")
        print(f"STDERR: {e.stderr}")
        return {}

def main():
    """Main function to run downsampling and collect statistics."""
    # Define paths
    input_file = "../../data/ustishim_23andme/2014_FuNature_ustishim_23andme.txt"
    output_dir = "./"  # Current directory is already results/ustishim_downsamples
    script_path = "../../downsample.py"

    # Ensure input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define missingness percentages to test
    percentages = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    # Initialize results list
    results = []

    # Process each percentage
    for pct in percentages:
        print(f"\n{'='*60}")
        print(f"Processing {pct}% missingness...")
        print(f"{'='*60}")

        row_data = {'Missingness %': pct}

        # Run diploid downsampling
        diploid_results = run_downsampling(input_file, script_path, pct)
        row_data.update(diploid_results)

        # Run pseudo-haploid downsampling
        pseudohap_results = run_downsampling(input_file, script_path, pct, pseudo_haploid=True)
        row_data.update(pseudohap_results)

        # Add the row data to our results
        results.append(row_data)

    # Create and save the stats DataFrame with polars
    stats_df = pl.DataFrame(results)
    stats_file = os.path.join(output_dir, "downsampling_stats.csv")
    stats_df.write_csv(stats_file)
    print(f"\nStatistics saved to {stats_file}")

    # Also display the table
    print("\nDownsampling Results:")
    print(stats_df)

    print("\nAll processing complete!")

def extract_stats(output_text, section_name=""):
    """Extract total and missing loci counts from the output text."""
    # Default values in case parsing fails
    total_loci = 0
    missing_loci = 0

    # Find the section with stats (original, downsampled, or pseudo-haploid)
    if section_name:
        # Find the section after the section_name
        parts = output_text.split(section_name)
        if len(parts) > 1:
            output_text = parts[1]

    # Extract total loci
    total_match = re.search(r"Total number of loci: (\d+)", output_text)
    if total_match:
        total_loci = int(total_match.group(1))

    # Extract missing loci
    missing_match = re.search(r"Number of missing loci: (\d+)", output_text)
    if missing_match:
        missing_loci = int(missing_match.group(1))

    return total_loci, missing_loci

def handle_output_files(output_text, output_dir):
    """Copy generated files to the output directory."""
    for line in output_text.split('\n'):
        if "written to" in line:
            source_file = line.split("written to ")[-1].strip()
            if os.path.exists(source_file):
                # Copy the generated file to our output directory
                filename = os.path.basename(source_file)
                dest_file = os.path.join(output_dir, filename)
                shutil.copy2(source_file, dest_file)
                print(f"Copied to {dest_file}")
            else:
                print(f"Warning: Expected output file '{source_file}' not found")

if __name__ == "__main__":
    main()
