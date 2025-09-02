#!/usr/bin/env python3
"""
Main Evaluation Runner Script

This script orchestrates the complete evaluation process for comparing
base and fine-tuned GPT-4.1-nano models on software requirements classification.

Author: AI Engineer
Date: 2025
"""

import os
import sys
import subprocess
import time
from typing import List, Dict

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

class EvaluationRunner:
    """
    Main class to orchestrate the complete evaluation process.
    """
    
    def __init__(self):
        """Initialize the evaluation runner."""
        self.scripts = [
            'convert_excel_to_jsonl.py',
            'merge_datasets.py',
            'test_base_model.py',
            'test_finetuned_model.py',
            'compare_models.py'
        ]
        self.results = {}
    
    def check_prerequisites(self) -> bool:
        """
        Check if all prerequisites are met.
        
        Returns:
            bool: True if all prerequisites are met
        """
        print("🔍 Checking prerequisites...")
        
        # Check if OpenAI API key is set
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            print("💡 Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            return False
        
        # Check if required files exist
        required_files = [
            'data/dataset-test.jsonl',
            'data/dataset-train.jsonl',
            'data/dataset.jsonl',
            'data/dataset-test2.xlsx'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Error: Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        # Check if image directory exists
        if not os.path.exists('image'):
            os.makedirs('image')
            print("📁 Created image directory for visualizations")
        
        print("✅ All prerequisites met")
        return True
    
    def run_script(self, script_name: str) -> bool:
        """
        Run a single evaluation script.
        
        Args:
            script_name (str): Name of the script to run
            
        Returns:
            bool: True if script ran successfully
        """
        print(f"\n🚀 Running {script_name}...")
        print("=" * 50)
        
        try:
            # Run the script
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {script_name} completed successfully")
                if result.stdout:
                    print("📋 Output:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ {script_name} failed with return code {result.returncode}")
                if result.stderr:
                    print("📋 Error output:")
                    print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {script_name} timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Error running {script_name}: {e}")
            return False
    
    def run_complete_evaluation(self) -> Dict:
        """
        Run the complete evaluation process.
        
        Returns:
            Dict: Results of the evaluation process
        """
        print("🎯 Starting Complete Model Evaluation Process")
        print("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return {"success": False, "error": "Prerequisites not met"}
        
        # Run each script in sequence
        for i, script in enumerate(self.scripts, 1):
            print(f"\n📋 Step {i}/{len(self.scripts)}: {script}")
            
            success = self.run_script(script)
            self.results[script] = success
            
            if not success:
                print(f"\n❌ Evaluation failed at step {i}: {script}")
                print("🛑 Stopping evaluation process")
                return {
                    "success": False,
                    "error": f"Failed at step {i}: {script}",
                    "completed_steps": i - 1,
                    "results": self.results
                }
            
            # Add delay between scripts to avoid rate limiting
            if i < len(self.scripts):
                print("⏳ Waiting 2 seconds before next step...")
                time.sleep(2)
        
        print("\n🎉 Complete evaluation process finished successfully!")
        return {
            "success": True,
            "completed_steps": len(self.scripts),
            "results": self.results
        }
    
    def generate_summary_report(self, evaluation_results: Dict):
        """
        Generate a summary report of the evaluation process.
        
        Args:
            evaluation_results (Dict): Results from the evaluation process
        """
        print("\n📊 EVALUATION SUMMARY REPORT")
        print("=" * 50)
        
        if evaluation_results["success"]:
            print("✅ Status: SUCCESS")
            print(f"📈 Completed Steps: {evaluation_results['completed_steps']}/{len(self.scripts)}")
            
            print("\n📋 Step Results:")
            for script, success in self.results.items():
                status = "✅" if success else "❌"
                print(f"   {status} {script}")
            
            print("\n📁 Generated Files:")
            generated_files = [
                "data/dataset-test2-converted.jsonl",
                "data/merged_test_dataset.jsonl",
                "base_model_results.json",
                "finetuned_model_results.json",
                "model_comparison_report.json",
                "merge_report.json"
            ]
            
            for file_path in generated_files:
                if os.path.exists(file_path):
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ❌ {file_path} (not found)")
            
            print("\n🖼️  Generated Visualizations:")
            visualization_files = [
                "image/confusion_matrix_base_model.png",
                "image/performance_base_model.png",
                "image/confusion_matrix_finetuned_model.png",
                "image/performance_finetuned_model.png",
                "image/model_comparison.png",
                "image/improvement_analysis.png"
            ]
            
            for file_path in visualization_files:
                if os.path.exists(file_path):
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ❌ {file_path} (not found)")
        
        else:
            print("❌ Status: FAILED")
            print(f"🛑 Error: {evaluation_results.get('error', 'Unknown error')}")
            print(f"📈 Completed Steps: {evaluation_results.get('completed_steps', 0)}/{len(self.scripts)}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files if needed."""
        temp_files = [
            "data/dataset-test2-converted.jsonl",
            "merge_report.json"
        ]
        
        print("\n🧹 Cleaning up temporary files...")
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"   🗑️  Removed {file_path}")
                except Exception as e:
                    print(f"   ⚠️  Could not remove {file_path}: {e}")

def main():
    """
    Main function to run the complete evaluation process.
    """
    print("🎯 GPT-4.1-Nano Fine-tuning Performance Evaluation")
    print("=" * 60)
    print("This script will run the complete evaluation process:")
    print("1. Convert Excel data to JSONL format")
    print("2. Merge datasets for comprehensive testing")
    print("3. Evaluate base model performance")
    print("4. Evaluate fine-tuned model performance")
    print("5. Compare models and generate visualizations")
    print("=" * 60)
    
    # Initialize runner
    runner = EvaluationRunner()
    
    # Run complete evaluation
    results = runner.run_complete_evaluation()
    
    # Generate summary report
    runner.generate_summary_report(results)
    
    # Ask user if they want to clean up temporary files
    if results["success"]:
        print("\n🧹 Cleanup Options:")
        print("1. Keep all files (recommended for analysis)")
        print("2. Clean up temporary files")
        
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == "2":
                runner.cleanup_temp_files()
        except KeyboardInterrupt:
            print("\n👋 Evaluation completed. Files preserved.")
    
    print("\n🎉 Evaluation process completed!")
    print("📖 Check README.md and REPORT.md for detailed analysis")
    print("🖼️  View generated visualizations in the image/ directory")

if __name__ == "__main__":
    main()
