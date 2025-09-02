#!/usr/bin/env python3
"""
Excel to JSONL Converter Script

This script converts the dataset-test2.xlsx file to JSONL format
for software requirements classification data.

Author: AI Engineer
Date: 2025
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any

class ExcelToJSONLConverter:
    """
    Converter class for transforming Excel data to JSONL format
    for software requirements classification.
    """
    
    def __init__(self):
        """Initialize the converter."""
        self.converted_data = []
    
    def load_excel_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            pd.DataFrame: Loaded Excel data
        """
        try:
            # Try to read the Excel file
            df = pd.read_excel(file_path)
            print(f"✅ Successfully loaded Excel file: {file_path}")
            print(f"📊 Data shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"❌ Error: Excel file {file_path} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return pd.DataFrame()
    
    def inspect_data_structure(self, df: pd.DataFrame):
        """
        Inspect the structure of the loaded data.
        
        Args:
            df (pd.DataFrame): The loaded DataFrame
        """
        print("\n🔍 Data Structure Inspection:")
        print("=" * 50)
        
        # Display basic info
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check for missing values
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Display unique values in each column
        for col in df.columns:
            unique_count = df[col].nunique()
            print(f"\nColumn '{col}' has {unique_count} unique values")
            if unique_count <= 10:  # Show unique values if not too many
                print(f"Unique values: {df[col].unique().tolist()}")
    
    def convert_to_jsonl_format(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to JSONL format matching the expected structure.
        
        Args:
            df (pd.DataFrame): The loaded DataFrame
            
        Returns:
            List[Dict]: List of JSONL-formatted records
        """
        jsonl_data = []
        
        # Determine the column mapping based on common patterns
        # This is a flexible approach that tries to identify the right columns
        requirement_col = None
        label_col = None
        
        # Try to identify requirement and label columns
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['requirement', 'text', 'content', 'description']):
                requirement_col = col
            elif any(keyword in col_lower for keyword in ['label', 'category', 'type', 'class']):
                label_col = col
        
        # If we can't identify columns automatically, use the first two columns
        if requirement_col is None or label_col is None:
            if len(df.columns) >= 2:
                requirement_col = df.columns[0]
                label_col = df.columns[1]
                print(f"⚠️  Using first two columns: '{requirement_col}' and '{label_col}'")
            else:
                print("❌ Error: Cannot identify requirement and label columns")
                return []
        
        print(f"📝 Using requirement column: '{requirement_col}'")
        print(f"🏷️  Using label column: '{label_col}'")
        
        # Convert each row to JSONL format
        for index, row in df.iterrows():
            requirement_text = str(row[requirement_col]).strip()
            label = str(row[label_col]).strip().lower()
            
            # Normalize label to match expected format
            if 'functional' in label and 'non' not in label:
                normalized_label = 'functional'
            elif 'non-functional' in label or 'nonfunctional' in label:
                normalized_label = 'non-functional'
            else:
                # Try to infer from context
                if 'non' in label:
                    normalized_label = 'non-functional'
                else:
                    normalized_label = 'functional'
            
            # Create JSONL record in the expected format
            jsonl_record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Software Engineer and need to categorize requirements into 'functional' or 'non-functional'. Your answer must be 'functional' or 'non-functional' only."
                    },
                    {
                        "role": "user",
                        "content": requirement_text
                    },
                    {
                        "role": "assistant",
                        "content": normalized_label
                    }
                ]
            }
            
            jsonl_data.append(jsonl_record)
        
        print(f"✅ Converted {len(jsonl_data)} records to JSONL format")
        return jsonl_data
    
    def save_jsonl(self, data: List[Dict], output_path: str):
        """
        Save data to JSONL file.
        
        Args:
            data (List[Dict]): List of JSONL records
            output_path (str): Path to save the JSONL file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"💾 Successfully saved {len(data)} records to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving JSONL file: {e}")
    
    def validate_conversion(self, data: List[Dict]):
        """
        Validate the converted data.
        
        Args:
            data (List[Dict]): List of converted records
        """
        print("\n🔍 Validation Results:")
        print("=" * 30)
        
        if not data:
            print("❌ No data to validate")
            return
        
        # Check structure
        valid_records = 0
        label_counts = {'functional': 0, 'non-functional': 0}
        
        for record in data:
            if 'messages' in record and len(record['messages']) == 3:
                valid_records += 1
                
                # Count labels
                label = record['messages'][2]['content']
                if label in label_counts:
                    label_counts[label] += 1
        
        print(f"✅ Valid records: {valid_records}/{len(data)}")
        print(f"📊 Label distribution:")
        print(f"   - Functional: {label_counts['functional']}")
        print(f"   - Non-functional: {label_counts['non-functional']}")
        
        # Show sample record
        if data:
            print(f"\n📋 Sample record:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False))

def main():
    """
    Main function to convert Excel to JSONL.
    """
    print("🔄 Starting Excel to JSONL Conversion")
    print("=" * 50)
    
    # Initialize converter
    converter = ExcelToJSONLConverter()
    
    # Load Excel data
    excel_file = 'data/dataset-test2.xlsx'
    df = converter.load_excel_data(excel_file)
    
    if df.empty:
        print("❌ No data loaded. Exiting.")
        return
    
    # Inspect data structure
    converter.inspect_data_structure(df)
    
    # Convert to JSONL format
    jsonl_data = converter.convert_to_jsonl_format(df)
    
    if not jsonl_data:
        print("❌ Conversion failed. Exiting.")
        return
    
    # Validate conversion
    converter.validate_conversion(jsonl_data)
    
    # Save to JSONL file
    output_file = 'data/dataset-test2-converted.jsonl'
    converter.save_jsonl(jsonl_data, output_file)
    
    print(f"\n✅ Conversion completed successfully!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 Total records: {len(jsonl_data)}")

if __name__ == "__main__":
    main()
