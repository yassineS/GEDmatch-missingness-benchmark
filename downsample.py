import argparse
import polars as pl
import random
import os

def main():
    parser = argparse.ArgumentParser(description="Process a 23andme file and downsample it.")
    parser.add_argument("-i", "--input_file", required=True, help="Path to the input 23andme file.")
    parser.add_argument("-s", "--calculate_stats", action="store_true", help="Calculate and print statistics about the file.")
    parser.add_argument("-p", "--percentage_to_remove", type=float, default=None, help="Percentage of loci to remove (default: None).")
    parser.add_argument("-a", "--pseudo_haploid", action="store_true", help="Generate pseudo-haploid genotypes by randomly selecting alleles.")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debugging information.")
    args = parser.parse_args()

    try:
        # First extract headers to later write them to output file
        headers = []
        column_names = None
        with open(args.input_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    headers.append(line)
                    if '# rsid chromosome position genotype' in line:
                        # Extract column names from this specific header line
                        column_names = line.strip('# \n').split()
                else:
                    break
        
        # Use infer_schema_length=0 to properly infer schema from first data rows
        df = pl.read_csv(
            args.input_file,
            separator="\t",
            has_header=False,
            comment_prefix="#",
            infer_schema_length=0,
            new_columns=column_names if column_names else ["rsid", "chromosome", "position", "genotype"],
        )
        
        # Extract ref and alt alleles from genotype column
        # For missing values ("--"), both ref and alt will be NaN
        genotype_col = "genotype" if "genotype" in df.columns else "column_4"
        df = df.with_columns([
            pl.when(pl.col(genotype_col) == "--")
            .then(pl.lit(None))
            .otherwise(pl.col(genotype_col).str.slice(0, 1))
            .alias("ref"),
            
            pl.when(pl.col(genotype_col) == "--")
            .then(pl.lit(None))
            .when(pl.col(genotype_col).str.slice(1, 1) == "")  # Check if second character exists
            .then(pl.lit(None))  # Handle single character genotypes
            .when(pl.col(genotype_col).str.slice(1, 1) == "-")
            .then(pl.lit(None))  # Handle cases like "A-"
            .otherwise(pl.col(genotype_col).str.slice(1, 1))
            .alias("alt")
        ])
        
        # Also handle "-C" type cases for the ref allele
        df = df.with_columns([
            pl.when(pl.col("ref") == "-")
            .then(pl.lit(None))
            .otherwise(pl.col("ref"))
            .alias("ref")
        ])
        
        if args.debug:
            print(df.head())
        
        # Calculate stats if requested
        if args.calculate_stats:
            print("Original stats:")
            display_stats(df)
        
        # Process the data according to the requested operations
        processed_df = df
        
        # Perform downsampling if requested
        if args.percentage_to_remove is not None:
            processed_df = remove_random_loci(processed_df, args.percentage_to_remove)
            
            if args.debug:
                print(processed_df.head())
            
            # If stats were requested, display them for the downsampled data
            if args.calculate_stats:
                print("\nDownsampled stats:")
                display_stats(processed_df)
            
            # Include the percentage in the output filename
            output_file = os.path.splitext(args.input_file)[0] + f"_downsampled_{int(args.percentage_to_remove)}pct.txt"
            
            # Write header comments from original file plus processing info
            processing_info = f"This file has been downsampled to introduce {args.percentage_to_remove}% missingness"
            write_with_headers(headers, output_file, processed_df, processing_info)
            
            print(f"Downsampled file written to {output_file}")
            
            # Only show stats here if not already shown above
            if not args.calculate_stats:
                display_stats(processed_df, prefix="Downsampled stats:")
        
        # Perform pseudo-haploidization if requested
        if args.pseudo_haploid:
            # Create a suffix for the filename based on operations performed
            file_suffix = ""
            downsampling_info = ""
            if args.percentage_to_remove is not None:
                file_suffix = f"_downsampled_{int(args.percentage_to_remove)}pct"
                downsampling_info = f" after downsampling to introduce {args.percentage_to_remove}% missingness"
            
            # Apply pseudo-haploidization
            pseudo_haploid_df = pseudo_haploidize_genotypes(processed_df)
            
            if args.debug:
                print("\nPseudo-haploid data:")
                print(pseudo_haploid_df.head())
            
            # Generate output filename
            pseudo_output_file = os.path.splitext(args.input_file)[0] + f"{file_suffix}_pseudohaploid.txt"
            
            # Write to file with processing info
            processing_info = f"This file has been pseudo-haploidized{downsampling_info}"
            write_with_headers(headers, pseudo_output_file, pseudo_haploid_df, processing_info)
            
            print(f"Pseudo-haploid file written to {pseudo_output_file}")
            
            if args.calculate_stats:
                print("\nPseudo-haploid stats:")
                display_stats(pseudo_haploid_df)
        
    except pl.exceptions.ComputeError as e:
        print(f"Error: Could not read file at {args.input_file}. Error details: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at {args.input_file}")

def pseudo_haploidize_genotypes(df):
    """
    Generates pseudo-haploid genotypes by randomly selecting between ref and alt alleles.
    For each SNP, chooses either ref or alt and creates a homozygous genotype (refref or altalt).
    """
    # Create a copy to avoid modifying the original
    df_modified = df.clone()
    
    # Determine the genotype column name
    genotype_col = "genotype" if "genotype" in df.columns else "column_4"
    
    # Generate random choices for each row (0 = use ref, 1 = use alt)
    # We need to convert this to a Polars expression
    n_rows = df.height
    random_choices = [random.randint(0, 1) for _ in range(n_rows)]
    
    # Convert to Polars series and add as a column
    df_modified = df_modified.with_columns([
        pl.Series(random_choices).alias("random_choice")
    ])
    
    # Apply the pseudo-haploidization
    df_modified = df_modified.with_columns([
        # For missing values, keep as "--"
        # For ref choice, create refref genotype
        # For alt choice, create altalt genotype
        pl.when(pl.col(genotype_col) == "--")
        .then(pl.lit("--"))
        .when(pl.col("random_choice") == 0)
        .then(pl.col("ref") + pl.col("ref"))
        .when(pl.col("random_choice") == 1)
        .then(pl.col("alt") + pl.col("alt"))
        .otherwise(pl.col(genotype_col))
        .alias(genotype_col)
    ])
    
    # Handle null values in ref or alt that could cause issues
    df_modified = df_modified.with_columns([
        pl.when(pl.col("ref").is_null() | pl.col("alt").is_null())
        .then(pl.lit("--"))
        .otherwise(pl.col(genotype_col))
        .alias(genotype_col)
    ])
    
    # Drop the temporary random choice column
    df_modified = df_modified.drop("random_choice")
    
    # Update ref and alt columns to match the new genotypes
    df_modified = df_modified.with_columns([
        pl.when(pl.col(genotype_col) == "--")
        .then(pl.lit(None))
        .otherwise(pl.col(genotype_col).str.slice(0, 1))
        .alias("ref"),
        
        pl.when(pl.col(genotype_col) == "--")
        .then(pl.lit(None))
        .otherwise(pl.col(genotype_col).str.slice(1, 1))
        .alias("alt")
    ])
    
    return df_modified

def write_with_headers(headers, output_file, data, processing_info=None):
    """
    Write the header comments and then the data.
    
    Parameters:
    headers -- List of header lines from the original file
    output_file -- Path to the output file
    data -- DataFrame to write
    processing_info -- Optional string describing processing performed on the data
    """
    # Select only the original columns (rsid, chromosome, position, genotype)
    # and exclude the processing columns (ref, alt)
    columns_to_keep = ["rsid", "chromosome", "position", "genotype"]
    if all(col in data.columns for col in columns_to_keep):
        data_to_write = data.select(columns_to_keep)
    else:
        # Fallback to first 4 columns if column names are different
        data_to_write = data.select(data.columns[:4])
    
    with open(output_file, 'w') as fout:
        # Write all header comments from the original file, but insert processing info
        # before the column header line (typically the last header line)
        for i, header in enumerate(headers):
            # If this is the last header line and we have processing info, add the info line first
            if i == len(headers) - 1 and processing_info:
                fout.write(f"# {processing_info}\n")
            fout.write(header)
        
        # Write the data without ref/alt columns
        for row in data_to_write.rows():
            fout.write("\t".join(map(str, row)) + "\n")

def display_stats(df, prefix=""):
    """Display statistics about the DataFrame."""
    total_loci = df.height
    missing_loci = df.filter(pl.col("genotype") == "--").height if "genotype" in df.columns else df.filter(pl.col("column_4") == "--").height
    
    if total_loci > 0:
        missingness_level = (missing_loci / total_loci) * 100
        if prefix:
            print(prefix)
        print(f"Total number of loci: {total_loci}")
        print(f"Number of missing loci: {missing_loci}")
        print(f"Missingness level: {missingness_level:.2f}%")
    else:
        print("No loci found in the file.")

def remove_random_loci(df, percentage_to_remove):
    """Sets genotype to '--' for a random percentage of loci in the DataFrame."""
    if not 0 <= percentage_to_remove <= 100:
        raise ValueError("Percentage to remove must be between 0 and 100.")

    # Create a copy of the dataframe to avoid modifying the original
    df_modified = df.clone()
    
    # Calculate number of rows to modify
    num_rows_to_modify = int(len(df) * (percentage_to_remove / 100))
    rows_to_modify = random.sample(range(len(df)), num_rows_to_modify)
    
    # Create a mask for rows to modify
    mask = pl.int_range(0, len(df)).is_in(rows_to_modify)
    
    # Determine the genotype column name
    genotype_col = "genotype" if "genotype" in df.columns else "column_4"
    
    # Update the genotype column and the ref/alt columns
    df_modified = df_modified.with_columns([
        pl.when(mask)
        .then(pl.lit("--"))
        .otherwise(pl.col(genotype_col))
        .alias(genotype_col),
        
        pl.when(mask)
        .then(pl.lit(None))
        .otherwise(pl.col("ref"))
        .alias("ref"),
        
        pl.when(mask)
        .then(pl.lit(None))
        .otherwise(pl.col("alt"))
        .alias("alt")
    ])
    
    return df_modified

if __name__ == "__main__":
    main()
