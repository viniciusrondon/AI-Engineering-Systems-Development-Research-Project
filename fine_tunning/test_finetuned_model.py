#!/usr/bin/env python3
"""
Fine-tuned Model Performance Evaluation Script

This script evaluates the performance of the fine-tuned GPT-4.1-nano model
on software requirements classification task (functional vs non-functional).

Author: AI Engineer
Date: 2025
"""

import os
import json
import time
from typing import List, Dict, Tuple
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Set up OpenAI API
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class FineTunedModelEvaluator:
    """
    Evaluator class for testing fine-tuned GPT-4.1-nano model performance
    on software requirements classification.
    """
    
    def __init__(self, model_name: str = "ft:gpt-4.1-nano-2025-04-14:rondon:viniciusrondon:CBQ8G05I"):
        """
        Initialize the evaluator with the fine-tuned model.
        
        Args:
            model_name (str): Name of the fine-tuned OpenAI model to use
        """
        self.model_name = model_name
        self.results = []
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def load_test_data(self, file_path: str) -> List[Dict]:
        """
        Load test data from JSONL file.
        
        Args:
            file_path (str): Path to the JSONL test file
            
        Returns:
            List[Dict]: List of test examples
        """
        test_data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        test_data.append(json.loads(line))
            print(f"✅ Loaded {len(test_data)} test examples from {file_path}")
            return test_data
        except FileNotFoundError:
            print(f"❌ Error: Test file {file_path} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            return []
    
    def predict_requirement_type(self, requirement_text: str) -> str:
        """
        Get prediction from the fine-tuned model for a given requirement.
        
        Args:
            requirement_text (str): The software requirement text
            
        Returns:
            str: Model prediction ('functional' or 'non-functional')
        """
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a Software Engineer and need to categorize requirements into 'functional' or 'non-functional'. Your answer must be 'functional' or 'non-functional' only."
                    },
                    {
                        "role": "user", 
                        "content": requirement_text
                    }
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            prediction = response.choices[0].message.content.strip().lower()
            print(f"🔍 Raw prediction: '{prediction}'")
            
            # Normalize prediction to match expected format
            if 'functional' in prediction and 'non' not in prediction:
                return 'functional'
            elif 'non-functional' in prediction or 'nonfunctional' in prediction:
                return 'non-functional'
            else:
                # Fallback: try to extract from the response
                print(f"⚠️  Unexpected prediction format: '{prediction}'")
                if 'non' in prediction:
                    return 'non-functional'
                else:
                    return 'functional'
                    
        except Exception as e:
            print(f"❌ Error getting prediction: {e}")
            print(f"❌ Error type: {type(e)}")
            return 'error'
    
    def evaluate_model(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate the fine-tuned model on the test dataset.
        
        Args:
            test_data (List[Dict]): List of test examples
            
        Returns:
            Dict: Evaluation results
        """
        print(f"\n🔍 Evaluating fine-tuned model: {self.model_name}")
        print("=" * 60)
        
        self.results = []
        self.correct_predictions = 0
        self.total_predictions = len(test_data)
        
        for i, example in enumerate(test_data, 1):
            # Extract requirement text and expected label
            user_message = example['messages'][1]['content']
            expected_label = example['messages'][2]['content']
            
            print(f"Processing example {i}/{self.total_predictions}...", end=" ")
            
            # Get model prediction
            predicted_label = self.predict_requirement_type(user_message)
            
            # Check if prediction is correct
            is_correct = predicted_label == expected_label
            if is_correct:
                self.correct_predictions += 1
            
            # Store result
            result = {
                'example_id': i,
                'requirement': user_message,
                'expected': expected_label,
                'predicted': predicted_label,
                'correct': is_correct
            }
            self.results.append(result)
            
            # Print result
            status = "✅" if is_correct else "❌"
            print(f"{status} Expected: {expected_label}, Predicted: {predicted_label}")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Calculate accuracy
        accuracy = (self.correct_predictions / self.total_predictions) * 100
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total examples: {self.total_predictions}")
        print(f"Correct predictions: {self.correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        return {
            'total_examples': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': accuracy,
            'results': self.results
        }
    
    def generate_confusion_matrix(self, save_path: str = None):
        """
        Generate and save confusion matrix visualization.
        
        Args:
            save_path (str): Path to save the confusion matrix image
        """
        if not self.results:
            print("❌ No results available for confusion matrix")
            return
        
        # Create confusion matrix data
        expected_labels = [r['expected'] for r in self.results]
        predicted_labels = [r['predicted'] for r in self.results]
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(expected_labels, predicted_labels, 
                            labels=['functional', 'non-functional'])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['functional', 'non-functional'],
                   yticklabels=['functional', 'non-functional'])
        plt.title(f'Confusion Matrix - Fine-tuned Model ({self.model_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def generate_performance_chart(self, save_path: str = None):
        """
        Generate performance visualization chart.
        
        Args:
            save_path (str): Path to save the performance chart
        """
        if not self.results:
            print("❌ No results available for performance chart")
            return
        
        # Calculate metrics
        accuracy = (self.correct_predictions / self.total_predictions) * 100
        
        # Count predictions by type
        expected_counts = Counter([r['expected'] for r in self.results])
        predicted_counts = Counter([r['predicted'] for r in self.results])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy bar chart
        ax1.bar(['Fine-tuned Model'], [accuracy], color='lightgreen', alpha=0.7)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(f'Fine-tuned Model Accuracy: {accuracy:.2f}%')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        ax1.text(0, accuracy + 1, f'{accuracy:.2f}%', 
                ha='center', va='bottom', fontweight='bold')
        
        # Distribution comparison
        categories = ['functional', 'non-functional']
        expected_values = [expected_counts.get(cat, 0) for cat in categories]
        predicted_values = [predicted_counts.get(cat, 0) for cat in categories]
        
        x = range(len(categories))
        width = 0.35
        
        ax2.bar([i - width/2 for i in x], expected_values, width, 
               label='Expected', alpha=0.7, color='lightgreen')
        ax2.bar([i + width/2 for i in x], predicted_values, width,
               label='Predicted', alpha=0.7, color='lightcoral')
        
        ax2.set_xlabel('Requirement Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Expected vs Predicted Distribution')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance chart saved to: {save_path}")
        
        plt.show()
    
    def save_detailed_results(self, file_path: str):
        """
        Save detailed evaluation results to JSON file.
        
        Args:
            file_path (str): Path to save the results
        """
        results_summary = {
            'model_name': self.model_name,
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_examples': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': (self.correct_predictions / self.total_predictions) * 100,
            'detailed_results': self.results
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results saved to: {file_path}")

def main():
    """
    Main function to run the fine-tuned model evaluation.
    """
    print("🚀 Starting Fine-tuned Model Performance Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = FineTunedModelEvaluator()
    
    # Load test data (use merged dataset if available, otherwise use original test dataset)
    test_files = [
        'data/merged_test_dataset.jsonl',
        'data/dataset-test.jsonl'
    ]
    
    test_data = []
    for test_file in test_files:
        if os.path.exists(test_file):
            test_data = evaluator.load_test_data(test_file)
            break
    
    if not test_data:
        print("❌ No test data found. Please ensure at least one test file exists.")
        return
    
    # Evaluate model
    results = evaluator.evaluate_model(test_data)
    
    # Generate visualizations
    print("\n📊 Generating performance visualizations...")
    evaluator.generate_confusion_matrix('image/confusion_matrix_finetuned_model.png')
    evaluator.generate_performance_chart('image/performance_finetuned_model.png')
    
    # Save detailed results
    evaluator.save_detailed_results('finetuned_model_results.json')
    
    print("\n✅ Fine-tuned model evaluation completed successfully!")
    print(f"📈 Final Accuracy: {results['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
