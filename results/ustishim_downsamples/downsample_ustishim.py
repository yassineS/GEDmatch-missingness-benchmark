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

        # First run: downsampling only (diploid)
        cmd_diploid = [
            "python", script_path,
            "-i", input_file,
            "-p", str(pct),
            "-s"  # Include stats
        ]

        # Run the diploid command
        try:
            process_diploid = subprocess.run(
                cmd_diploid, check=True, text=True, capture_output=True
                )
            print("Diploid processing complete.")

            # Parse stats from output
            total_loci, missing_loci = extract_stats(process_diploid.stdout, "Downsampled stats")
            row_data['Diploid Total Loci'] = total_loci
            row_data['Diploid Remaining Loci'] = total_loci - missing_loci
            row_data['Diploid Missing Loci'] = missing_loci
            row_data['Diploid Actual Missingness %'] = (missing_loci * 100 / total_loci
														 if total_loci > 0 else 0)

            # Handle file copying for diploid
            handle_output_files(process_diploid.stdout, output_dir)

        except subprocess.CalledProcessError as e:
            print(f"Error running diploid downsampling with {pct}% missingness:")
            print(f"Command: {' '.join(cmd_diploid)}")
            print(f"Error: {e}")
            print(f"STDERR: {e.stderr}")

        # Second run: downsampling + pseudo-haploid
        cmd_pseudohap = [
            "python", script_path,
            "-i", input_file,
            "-p", str(pct),
            "-a",  # Enable pseudo-haploid mode
            "-s"   # Include stats
        ]

        # Run the pseudo-haploid command
        try:
            process_pseudohap = subprocess.run(
                cmd_pseudohap, check=True, text=True, capture_output=True
                )
            print("Pseudo-haploid processing complete.")

            # Parse stats from output
            total_loci, missing_loci = extract_stats(
                process_pseudohap.stdout, "Pseudo-haploid stats"
                )
            row_data['PseudoHaploid Total Loci'] = total_loci
            row_data['PseudoHaploid Remaining Loci'] = total_loci - missing_loci
            row_data['PseudoHaploid Missing Loci'] = missing_loci
            row_data['PseudoHaploid Actual Missingness %'] = ((missing_loci / total_loci * 100)
                                                            if total_loci > 0 else 0)

            # Handle file copying for pseudo-haploid
            handle_output_files(process_pseudohap.stdout, output_dir)

        except subprocess.CalledProcessError as e:
            print(f"Error running pseudo-haploid downsampling with {pct}% missingness:")
            print(f"Command: {' '.join(cmd_pseudohap)}")
            print(f"Error: {e}")
            print(f"STDERR: {e.stderr}")

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
