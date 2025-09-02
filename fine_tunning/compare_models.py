#!/usr/bin/env python3
"""
Model Performance Comparison Script

This script compares the performance of base GPT-4.1-nano model
with its fine-tuned version for software requirements classification.

Author: AI Engineer
Date: 2025
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    """
    Class for comparing performance between base and fine-tuned models.
    """
    
    def __init__(self):
        """Initialize the model comparator."""
        self.base_results = None
        self.finetuned_results = None
        self.comparison_data = {}
    
    def load_results(self, base_results_file: str, finetuned_results_file: str):
        """
        Load results from both model evaluations.
        
        Args:
            base_results_file (str): Path to base model results
            finetuned_results_file (str): Path to fine-tuned model results
        """
        print("📊 Loading model evaluation results...")
        
        # Load base model results
        try:
            with open(base_results_file, 'r', encoding='utf-8') as f:
                self.base_results = json.load(f)
            print(f"✅ Loaded base model results from {base_results_file}")
        except FileNotFoundError:
            print(f"⚠️  Base model results file not found: {base_results_file}")
            self.base_results = None
        except Exception as e:
            print(f"❌ Error loading base model results: {e}")
            self.base_results = None
        
        # Load fine-tuned model results
        try:
            with open(finetuned_results_file, 'r', encoding='utf-8') as f:
                self.finetuned_results = json.load(f)
            print(f"✅ Loaded fine-tuned model results from {finetuned_results_file}")
        except FileNotFoundError:
            print(f"⚠️  Fine-tuned model results file not found: {finetuned_results_file}")
            self.finetuned_results = None
        except Exception as e:
            print(f"❌ Error loading fine-tuned model results: {e}")
            self.finetuned_results = None
    
    def calculate_detailed_metrics(self, results: Dict) -> Dict:
        """
        Calculate detailed performance metrics from results.
        
        Args:
            results (Dict): Model evaluation results
            
        Returns:
            Dict: Detailed metrics
        """
        if not results or 'detailed_results' not in results:
            return {}
        
        detailed_results = results['detailed_results']
        
        # Extract labels
        expected_labels = [r['expected'] for r in detailed_results]
        predicted_labels = [r['predicted'] for r in detailed_results]
        
        # Calculate confusion matrix
        cm = confusion_matrix(expected_labels, predicted_labels, 
                            labels=['functional', 'non-functional'])
        
        # Calculate metrics for each class
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': results.get('accuracy', 0),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision_functional': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall_functional': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'precision_non_functional': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'recall_non_functional': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_functional': 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))) if (tp + fp) > 0 and (tp + fn) > 0 else 0,
            'f1_non_functional': 2 * (tn / (tn + fn)) * (tn / (tn + fp)) / ((tn / (tn + fn)) + (tn / (tn + fp))) if (tn + fn) > 0 and (tn + fp) > 0 else 0
        }
        
        return metrics
    
    def generate_comparison_chart(self, save_path: str = None):
        """
        Generate comprehensive comparison chart.
        
        Args:
            save_path (str): Path to save the comparison chart
        """
        if not self.base_results or not self.finetuned_results:
            print("❌ Cannot generate comparison chart: missing results data")
            return
        
        # Calculate metrics for both models
        base_metrics = self.calculate_detailed_metrics(self.base_results)
        finetuned_metrics = self.calculate_detailed_metrics(self.finetuned_results)
        
        # Create comprehensive comparison visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy Comparison
        models = ['Base Model', 'Fine-tuned Model']
        accuracies = [base_metrics.get('accuracy', 0), finetuned_metrics.get('accuracy', 0)]
        colors = ['skyblue', 'lightgreen']
        
        bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.7)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Precision and Recall Comparison
        metrics_names = ['Precision\n(Functional)', 'Recall\n(Functional)', 
                        'Precision\n(Non-functional)', 'Recall\n(Non-functional)']
        base_values = [
            base_metrics.get('precision_functional', 0) * 100,
            base_metrics.get('recall_functional', 0) * 100,
            base_metrics.get('precision_non_functional', 0) * 100,
            base_metrics.get('recall_non_functional', 0) * 100
        ]
        finetuned_values = [
            finetuned_metrics.get('precision_functional', 0) * 100,
            finetuned_metrics.get('recall_functional', 0) * 100,
            finetuned_metrics.get('precision_non_functional', 0) * 100,
            finetuned_metrics.get('recall_non_functional', 0) * 100
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax2.bar(x - width/2, base_values, width, label='Base Model', alpha=0.7, color='skyblue')
        ax2.bar(x + width/2, finetuned_values, width, label='Fine-tuned Model', alpha=0.7, color='lightgreen')
        
        ax2.set_ylabel('Score (%)')
        ax2.set_title('Precision and Recall Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # 3. Confusion Matrix Comparison
        # Base model confusion matrix
        cm_base = np.array(base_metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
        sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['functional', 'non-functional'],
                   yticklabels=['functional', 'non-functional'])
        ax3.set_title('Base Model - Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Fine-tuned model confusion matrix
        cm_finetuned = np.array(finetuned_metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
        sns.heatmap(cm_finetuned, annot=True, fmt='d', cmap='Greens', ax=ax4,
                   xticklabels=['functional', 'non-functional'],
                   yticklabels=['functional', 'non-functional'])
        ax4.set_title('Fine-tuned Model - Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison chart saved to: {save_path}")
        
        plt.show()
    
    def generate_improvement_analysis(self, save_path: str = None):
        """
        Generate improvement analysis chart.
        
        Args:
            save_path (str): Path to save the improvement analysis
        """
        if not self.base_results or not self.finetuned_results:
            print("❌ Cannot generate improvement analysis: missing results data")
            return
        
        base_metrics = self.calculate_detailed_metrics(self.base_results)
        finetuned_metrics = self.calculate_detailed_metrics(self.finetuned_results)
        
        # Calculate improvements
        improvements = {
            'Accuracy': finetuned_metrics.get('accuracy', 0) - base_metrics.get('accuracy', 0),
            'Precision (Functional)': (finetuned_metrics.get('precision_functional', 0) - base_metrics.get('precision_functional', 0)) * 100,
            'Recall (Functional)': (finetuned_metrics.get('recall_functional', 0) - base_metrics.get('recall_functional', 0)) * 100,
            'Precision (Non-functional)': (finetuned_metrics.get('precision_non_functional', 0) - base_metrics.get('precision_non_functional', 0)) * 100,
            'Recall (Non-functional)': (finetuned_metrics.get('recall_non_functional', 0) - base_metrics.get('recall_non_functional', 0)) * 100,
            'F1 (Functional)': (finetuned_metrics.get('f1_functional', 0) - base_metrics.get('f1_functional', 0)) * 100,
            'F1 (Non-functional)': (finetuned_metrics.get('f1_non_functional', 0) - base_metrics.get('f1_non_functional', 0)) * 100
        }
        
        # Create improvement visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Improvement bar chart
        metrics = list(improvements.keys())
        values = list(improvements.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax1.barh(metrics, values, color=colors, alpha=0.7)
        ax1.set_xlabel('Improvement (%)')
        ax1.set_title('Fine-tuned Model Improvements Over Base Model')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_width() + (0.1 if value >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                    f'{value:+.2f}%', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
        
        # Performance summary
        summary_data = {
            'Metric': ['Accuracy', 'Precision (F)', 'Recall (F)', 'Precision (NF)', 'Recall (NF)'],
            'Base Model': [
                base_metrics.get('accuracy', 0),
                base_metrics.get('precision_functional', 0) * 100,
                base_metrics.get('recall_functional', 0) * 100,
                base_metrics.get('precision_non_functional', 0) * 100,
                base_metrics.get('recall_non_functional', 0) * 100
            ],
            'Fine-tuned Model': [
                finetuned_metrics.get('accuracy', 0),
                finetuned_metrics.get('precision_functional', 0) * 100,
                finetuned_metrics.get('recall_functional', 0) * 100,
                finetuned_metrics.get('precision_non_functional', 0) * 100,
                finetuned_metrics.get('recall_non_functional', 0) * 100
            ]
        }
        
        df = pd.DataFrame(summary_data)
        df.set_index('Metric', inplace=True)
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2, 
                   cbar_kws={'label': 'Score (%)'})
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xlabel('Model')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Improvement analysis saved to: {save_path}")
        
        plt.show()
        
        return improvements
    
    def generate_summary_report(self, save_path: str = None):
        """
        Generate a comprehensive summary report.
        
        Args:
            save_path (str): Path to save the summary report
        """
        if not self.base_results or not self.finetuned_results:
            print("❌ Cannot generate summary report: missing results data")
            return
        
        base_metrics = self.calculate_detailed_metrics(self.base_results)
        finetuned_metrics = self.calculate_detailed_metrics(self.finetuned_results)
        improvements = self.generate_improvement_analysis()
        
        report = {
            "evaluation_summary": {
                "base_model": {
                    "name": self.base_results.get('model_name', 'Unknown'),
                    "accuracy": base_metrics.get('accuracy', 0),
                    "total_examples": self.base_results.get('total_examples', 0),
                    "correct_predictions": self.base_results.get('correct_predictions', 0)
                },
                "finetuned_model": {
                    "name": self.finetuned_results.get('model_name', 'Unknown'),
                    "accuracy": finetuned_metrics.get('accuracy', 0),
                    "total_examples": self.finetuned_results.get('total_examples', 0),
                    "correct_predictions": self.finetuned_results.get('correct_predictions', 0)
                }
            },
            "detailed_metrics": {
                "base_model": base_metrics,
                "finetuned_model": finetuned_metrics
            },
            "improvements": improvements,
            "conclusion": {
                "overall_improvement": improvements.get('Accuracy', 0),
                "best_improvement_metric": max(improvements.items(), key=lambda x: x[1])[0] if improvements else "N/A",
                "worst_improvement_metric": min(improvements.items(), key=lambda x: x[1])[0] if improvements else "N/A"
            }
        }
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📋 Summary report saved to: {save_path}")
        
        return report

def main():
    """
    Main function to run model comparison.
    """
    print("🔄 Starting Model Performance Comparison")
    print("=" * 50)
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load results
    base_results_file = 'base_model_results.json'
    finetuned_results_file = 'finetuned_model_results.json'
    
    comparator.load_results(base_results_file, finetuned_results_file)
    
    if not comparator.base_results or not comparator.finetuned_results:
        print("❌ Cannot proceed with comparison: missing results files")
        print("💡 Please run test_base_model.py and test_finetuned_model.py first")
        return
    
    # Generate comparison visualizations
    print("\n📊 Generating comparison visualizations...")
    comparator.generate_comparison_chart('image/model_comparison.png')
    improvements = comparator.generate_improvement_analysis('image/improvement_analysis.png')
    
    # Generate summary report
    report = comparator.generate_summary_report('model_comparison_report.json')
    
    # Print summary
    print("\n📈 COMPARISON SUMMARY")
    print("=" * 40)
    print(f"Base Model Accuracy: {comparator.base_results.get('accuracy', 0):.2f}%")
    print(f"Fine-tuned Model Accuracy: {comparator.finetuned_results.get('accuracy', 0):.2f}%")
    print(f"Overall Improvement: {improvements.get('Accuracy', 0):+.2f}%")
    
    if improvements.get('Accuracy', 0) > 0:
        print("✅ Fine-tuning improved model performance!")
    else:
        print("⚠️  Fine-tuning did not improve model performance")
    
    print("\n✅ Model comparison completed successfully!")

if __name__ == "__main__":
    main()
