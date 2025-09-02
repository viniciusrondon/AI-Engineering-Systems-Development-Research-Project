#!/usr/bin/env python3
"""
Dataset Merger Script

This script merges the original dataset.jsonl with the converted dataset-test2.xlsx data
to create a larger test dataset for model evaluation.

Author: AI Engineer
Date: 2025
"""

import json
import os
from typing import List, Dict, Set
from collections import Counter

class DatasetMerger:
    """
    Class for merging multiple JSONL datasets into a single test dataset.
    """
    
    def __init__(self):
        """Initialize the dataset merger."""
        self.merged_data = []
        self.duplicate_count = 0
        self.total_original = 0
    
    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        """
        Load data from JSONL file.
        
        Args:
            file_path (str): Path to the JSONL file
            
        Returns:
            List[Dict]: List of loaded records
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            data.append(record)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}")
            
            print(f"✅ Loaded {len(data)} records from {file_path}")
            return data
        except FileNotFoundError:
            print(f"❌ Error: File {file_path} not found")
            return []
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return []
    
    def extract_requirement_text(self, record: Dict) -> str:
        """
        Extract the requirement text from a record.
        
        Args:
            record (Dict): JSONL record
            
        Returns:
            str: Extracted requirement text
        """
        try:
            if 'messages' in record and len(record['messages']) >= 2:
                return record['messages'][1]['content'].strip()
        except (KeyError, IndexError, TypeError):
            pass
        return ""
    
    def extract_label(self, record: Dict) -> str:
        """
        Extract the label from a record.
        
        Args:
            record (Dict): JSONL record
            
        Returns:
            str: Extracted label
        """
        try:
            if 'messages' in record and len(record['messages']) >= 3:
                return record['messages'][2]['content'].strip()
        except (KeyError, IndexError, TypeError):
            pass
        return ""
    
    def is_duplicate(self, record1: Dict, record2: Dict) -> bool:
        """
        Check if two records are duplicates based on requirement text.
        
        Args:
            record1 (Dict): First record
            record2 (Dict): Second record
            
        Returns:
            bool: True if records are duplicates
        """
        text1 = self.extract_requirement_text(record1)
        text2 = self.extract_requirement_text(record2)
        
        # Normalize texts for comparison (remove extra whitespace, convert to lowercase)
        text1_normalized = ' '.join(text1.lower().split())
        text2_normalized = ' '.join(text2.lower().split())
        
        return text1_normalized == text2_normalized
    
    def merge_datasets(self, dataset1_path: str, dataset2_path: str) -> List[Dict]:
        """
        Merge two datasets, removing duplicates.
        
        Args:
            dataset1_path (str): Path to first dataset
            dataset2_path (str): Path to second dataset
            
        Returns:
            List[Dict]: Merged dataset
        """
        print("🔄 Starting dataset merge process...")
        print("=" * 50)
        
        # Load both datasets
        dataset1 = self.load_jsonl_data(dataset1_path)
        dataset2 = self.load_jsonl_data(dataset2_path)
        
        if not dataset1 and not dataset2:
            print("❌ No data loaded from either dataset")
            return []
        
        print(f"📊 Dataset 1 ({dataset1_path}): {len(dataset1)} records")
        print(f"📊 Dataset 2 ({dataset2_path}): {len(dataset2)} records")
        
        # Start with first dataset
        merged_data = dataset1.copy()
        self.total_original = len(dataset1)
        
        # Add records from second dataset, checking for duplicates
        for record2 in dataset2:
            is_duplicate = False
            
            # Check against all existing records
            for existing_record in merged_data:
                if self.is_duplicate(record2, existing_record):
                    is_duplicate = True
                    self.duplicate_count += 1
                    break
            
            if not is_duplicate:
                merged_data.append(record2)
        
        self.merged_data = merged_data
        
        print(f"✅ Merge completed!")
        print(f"📈 Total records after merge: {len(merged_data)}")
        print(f"🔄 Duplicates removed: {self.duplicate_count}")
        print(f"📊 New records added: {len(merged_data) - self.total_original}")
        
        return merged_data
    
    def analyze_merged_dataset(self):
        """
        Analyze the merged dataset and provide statistics.
        """
        if not self.merged_data:
            print("❌ No merged data to analyze")
            return
        
        print("\n📊 Merged Dataset Analysis:")
        print("=" * 40)
        
        # Count labels
        label_counts = Counter()
        valid_records = 0
        
        for record in self.merged_data:
            label = self.extract_label(record)
            if label in ['functional', 'non-functional']:
                label_counts[label] += 1
                valid_records += 1
        
        print(f"📋 Total records: {len(self.merged_data)}")
        print(f"✅ Valid records: {valid_records}")
        print(f"🏷️  Label distribution:")
        print(f"   - Functional: {label_counts['functional']} ({label_counts['functional']/valid_records*100:.1f}%)")
        print(f"   - Non-functional: {label_counts['non-functional']} ({label_counts['non-functional']/valid_records*100:.1f}%)")
        
        # Check for data quality issues
        issues = []
        for i, record in enumerate(self.merged_data):
            requirement_text = self.extract_requirement_text(record)
            label = self.extract_label(record)
            
            if not requirement_text:
                issues.append(f"Record {i+1}: Missing requirement text")
            elif len(requirement_text) < 10:
                issues.append(f"Record {i+1}: Very short requirement text")
            
            if label not in ['functional', 'non-functional']:
                issues.append(f"Record {i+1}: Invalid label '{label}'")
        
        if issues:
            print(f"\n⚠️  Data quality issues found: {len(issues)}")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"   - {issue}")
            if len(issues) > 5:
                print(f"   - ... and {len(issues) - 5} more issues")
        else:
            print("✅ No data quality issues found")
    
    def save_merged_dataset(self, output_path: str):
        """
        Save the merged dataset to a JSONL file.
        
        Args:
            output_path (str): Path to save the merged dataset
        """
        if not self.merged_data:
            print("❌ No merged data to save")
            return
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in self.merged_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"💾 Merged dataset saved to: {output_path}")
            print(f"📊 Total records saved: {len(self.merged_data)}")
            
        except Exception as e:
            print(f"❌ Error saving merged dataset: {e}")
    
    def save_merge_report(self, output_path: str):
        """
        Save a detailed merge report.
        
        Args:
            output_path (str): Path to save the merge report
        """
        report = {
            "merge_summary": {
                "total_original_records": self.total_original,
                "duplicates_removed": self.duplicate_count,
                "final_record_count": len(self.merged_data),
                "new_records_added": len(self.merged_data) - self.total_original
            },
            "label_distribution": {},
            "data_quality": {
                "valid_records": 0,
                "issues": []
            }
        }
        
        # Count labels
        label_counts = Counter()
        for record in self.merged_data:
            label = self.extract_label(record)
            if label in ['functional', 'non-functional']:
                label_counts[label] += 1
                report["data_quality"]["valid_records"] += 1
        
        report["label_distribution"] = dict(label_counts)
        
        # Check for issues
        for i, record in enumerate(self.merged_data):
            requirement_text = self.extract_requirement_text(record)
            label = self.extract_label(record)
            
            if not requirement_text:
                report["data_quality"]["issues"].append(f"Record {i+1}: Missing requirement text")
            elif len(requirement_text) < 10:
                report["data_quality"]["issues"].append(f"Record {i+1}: Very short requirement text")
            
            if label not in ['functional', 'non-functional']:
                report["data_quality"]["issues"].append(f"Record {i+1}: Invalid label '{label}'")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Merge report saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving merge report: {e}")

def main():
    """
    Main function to merge datasets.
    """
    print("🔄 Starting Dataset Merge Process")
    print("=" * 50)
    
    # Initialize merger
    merger = DatasetMerger()
    
    # Define file paths
    dataset1_path = 'data/dataset.jsonl'
    dataset2_path = 'data/dataset-test2-converted.jsonl'
    
    # Check if files exist
    if not os.path.exists(dataset1_path):
        print(f"❌ Error: {dataset1_path} not found")
        return
    
    if not os.path.exists(dataset2_path):
        print(f"❌ Error: {dataset2_path} not found")
        print("💡 Please run convert_excel_to_jsonl.py first")
        return
    
    # Merge datasets
    merged_data = merger.merge_datasets(dataset1_path, dataset2_path)
    
    if not merged_data:
        print("❌ Merge failed. Exiting.")
        return
    
    # Analyze merged dataset
    merger.analyze_merged_dataset()
    
    # Save merged dataset
    output_path = 'data/merged_test_dataset.jsonl'
    merger.save_merged_dataset(output_path)
    
    # Save merge report
    report_path = 'merge_report.json'
    merger.save_merge_report(report_path)
    
    print(f"\n✅ Dataset merge completed successfully!")
    print(f"📁 Merged dataset: {output_path}")
    print(f"📊 Total records: {len(merged_data)}")

if __name__ == "__main__":
    main()
