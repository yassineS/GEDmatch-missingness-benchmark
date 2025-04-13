#!/usr/bin/env python3
"""
test.py
Unit tests for the downsample module.
This script contains unit tests for the downsample module, which is designed to
downsample genetic data files. The tests cover various functionalities including
reading files, downsampling, pseudo-haploidization, and writing output files.
"""

import unittest
import os
import tempfile
from unittest.mock import patch
import polars as pl
import downsample
from downsample import extract_headers


class TestDownsample(unittest.TestCase):
    """Unit tests for the downsample module."""

    def setUp(self):
        """Set up test fixtures before each test method is run."""
        # Create a temporary test file with sample data
        # Important: Use explicit tab characters (\t) between fields
        self.test_data = """# This is a header
# rsid\tchromosome\tposition\tgenotype
rs123\t1\t1000\tAA
rs456\t1\t2000\tGC
rs789\t1\t3000\tTT
rs101\t2\t1500\tAG
rs202\t2\t2500\t--
"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
        self.temp_file.write(self.test_data)
        self.temp_file.close()
        self.test_file = self.temp_file.name

    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_file):
            os.unlink(self.test_file)

    def test_read_csv(self):
        """Test reading a genetic data file."""
        # Extract headers and column names using the extract_headers function
        _, column_names = extract_headers(self.test_file)

        df = pl.read_csv(
            self.test_file,
            separator="\t",
            has_header=False,
            comment_prefix="#",
            infer_schema_length=0,
            new_columns=column_names if column_names \
                else ["rsid", "chromosome", "position", "genotype"],
        )

        self.assertEqual(len(df), 5)  # 5 data rows in our test file
        self.assertEqual(df.columns, ["rsid", "chromosome", "position", "genotype"])

    def test_remove_random_loci(self):
        """Test the downsampling functionality."""
        # Read the test data
        df = pl.read_csv(
            self.test_file,
            separator="\t",
            has_header=False,
            comment_prefix="#",
            new_columns=["rsid", "chromosome", "position", "genotype"],
        )

        # Add ref and alt columns for processing
        df = df.with_columns([
            pl.when(pl.col("genotype") == "--")
            .then(pl.lit(None))
            .otherwise(pl.col("genotype").str.slice(0, 1))
            .alias("ref"),

            pl.when(pl.col("genotype") == "--")
            .then(pl.lit(None))
            .otherwise(pl.col("genotype").str.slice(1, 1))
            .alias("alt")
        ])

        # Test with 40% downsampling
        with patch('random.sample', return_value=[0, 2]):  # Mock to return predictable indices
            df_downsampled = downsample.remove_random_loci(df, 40)
            self.assertEqual(
                df_downsampled.filter(pl.col("genotype") == "--").height, 3
                )  # 2 new + 1 existing

    def test_pseudo_haploidize_genotypes(self):
        """Test the pseudo-haploidization functionality."""
        # Create test data
        df = pl.DataFrame({
            "rsid": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "chromosome": ["1", "1", "1", "2", "2"],
            "position": ["1000", "2000", "3000", "1000", "2000"],
            "genotype": ["AG", "TT", "CC", "GC", "--"],
            "ref": ["A", "T", "C", "G", None],
            "alt": ["G", "T", "C", "C", None]
        })

        # Test pseudo-haploidization with controlled randomness
        with patch('random.randint', side_effect=[0, 1, 0, 1, 0]):  # Control random choices
            pseudo_df = downsample.pseudo_haploidize_genotypes(df)

            # Verify genotypes are homozygous
            self.assertEqual(
                pseudo_df.filter(pl.col("rsid") == "rs1")["genotype"][0], "AA"
                )  # first allele
            self.assertEqual(
				pseudo_df.filter(pl.col("rsid") == "rs2")["genotype"][0], "TT"
			    )  # homozygous already
            self.assertEqual(
				pseudo_df.filter(pl.col("rsid") == "rs3")["genotype"][0], "CC"
			    )  # homozygous already
            self.assertEqual(
				pseudo_df.filter(pl.col("rsid") == "rs4")["genotype"][0], "CC"
			    )  # second allele
            self.assertEqual(
				pseudo_df.filter(pl.col("rsid") == "rs5")["genotype"][0], "--"
			    )  # missing remains missing

    def test_write_with_headers(self):
        """Test writing output with headers."""
        df = pl.DataFrame({
            "rsid": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "1", "2"],
            "position": ["1000", "2000", "1000"],
            "genotype": ["AA", "GC", "--"],
            "ref": ["A", "G", None],
            "alt": ["A", "C", None]
        })

        headers = ["# Header line 1\n", "# rsid\tchromosome\tposition\tgenotype\n"]
        output_file = tempfile.NamedTemporaryFile(delete=False).name

        try:
            downsample.write_with_headers(headers, output_file, df, "Test processing")

            # Verify the file exists and has content
            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(os.path.getsize(output_file) > 0)

            # Read the file and check content
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.readlines()

            # Check headers and processing info
            self.assertEqual(len(content), 6)  # 3 header lines + 3 data lines
            self.assertEqual(content[0], "# Header line 1\n")
            self.assertEqual(content[1], "# Test processing\n")
            self.assertEqual(content[2], "# rsid\tchromosome\tposition\tgenotype\n")

            # Check data lines
            self.assertTrue("rs1\t1\t1000\tAA" in content[3])
            self.assertTrue("rs2\t1\t2000\tGC" in content[4])
            self.assertTrue("rs3\t2\t1000\t--" in content[5])

            # Ensure no ref/alt columns in output
            for line in content[3:]:
                self.assertFalse("None" in line)  # No None values from ref/alt columns

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_display_stats(self):
        """Test display_stats function."""
        df = pl.DataFrame({
            "rsid": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "chromosome": ["1", "1", "1", "2", "2"],
            "position": ["1000", "2000", "3000", "1000", "2000"],
            "genotype": ["AG", "TT", "CC", "--", "--"],
        })

        with patch('builtins.print') as mock_print:
            downsample.display_stats(df)
            mock_print.assert_any_call("Total number of loci: 5")
            mock_print.assert_any_call("Number of missing loci: 2")
            mock_print.assert_any_call("Missingness level: 40.00%")

    def test_main_with_calculate_stats(self):
        """Test main function with --calculate_stats flag."""
        # Create a mock for the parser and args
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            # Configure mock
            mock_args.return_value.input_file = self.test_file
            mock_args.return_value.calculate_stats = True
            mock_args.return_value.percentage_to_remove = None
            mock_args.return_value.pseudo_haploid = False
            mock_args.return_value.debug = False

            # Mock print to check output
            with patch('builtins.print') as mock_print:
                downsample.main()
                mock_print.assert_any_call("Original stats:")

    def test_main_with_downsampling(self):
        """Test main function with downsampling."""
        # Create a temp file for output
        output_file = tempfile.NamedTemporaryFile(delete=False).name
        base_name = os.path.splitext(output_file)[0]

        # Mock the argument parser
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            # Configure mock
            mock_args.return_value.input_file = self.test_file
            mock_args.return_value.out = None
            mock_args.return_value.calculate_stats = False
            mock_args.return_value.percentage_to_remove = 50
            mock_args.return_value.pseudo_haploid = False
            mock_args.return_value.debug = False

            with patch('builtins.print') as mock_print:
                with patch('os.path.splitext', return_value=(base_name, '.txt')):
                    downsample.main()

                    # Instead of checking for exact message, check if any print call
                    # contains the substring "Downsampled file written to"
                    found = False
                    for call in mock_print.mock_calls:
                        # For each call args[0] is the first argument to print
                        if len(call.args) > 0 and "Downsampled file written to" in call.args[0]:
                            found = True
                            break

                    self.assertTrue(found, "No print call found with 'Downsampled file written to'")

    def test_main_with_pseudo_haploid(self):
        """Test main function with pseudo-haploidization."""
        # Create a temp file for output
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            base_name = os.path.splitext(temp_file.name)[0]

        # Mock the argument parser
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            # Configure mock
            mock_args.return_value.input_file = self.test_file
            mock_args.return_value.out = None  # Set to None to use default output path
            mock_args.return_value.calculate_stats = False
            mock_args.return_value.percentage_to_remove = None
            mock_args.return_value.pseudo_haploid = True
            mock_args.return_value.debug = False

            with patch('builtins.print') as mock_print:
                with patch('os.path.splitext', return_value=(base_name, '.txt')):
                    downsample.main()
                    # Fix the assert to match the actual print message
                    mock_print.assert_any_call(f"Pseudo-haploid file written \
                                               to {base_name}_pseudohaploid.txt")

        # Clean up the temp file if it exists
        if os.path.exists(f"{base_name}_pseudohaploid.txt"):
            os.unlink(f"{base_name}_pseudohaploid.txt")
        if os.path.exists(f"{base_name}_pseudohaploid.log"):
            os.unlink(f"{base_name}_pseudohaploid.log")

if __name__ == '__main__':
    unittest.main()
