"""
MEDICAL TRIAGE RAG EVALUATION SYSTEM
Batch test the improved LLM comparison with evaluation metrics
"""

import json
import os
import time
from typing import Dict, List
from improved_llm_compare import (
    load_rag_components,
    compare_models_with_rag,
    print_comparison_report,
    save_comparison_result,
)


class TriageEvaluator:
    """Evaluate triage performance against gold standard"""
    
    def __init__(self):
        self.results = []
        self.summary = {
            "total_queries": 0,
            "mistral_correct": 0,
            "groq_correct": 0,
            "mistral_avg_rag_score": 0,
            "groq_avg_rag_score": 0,
            "model_similarity_avg": 0,
        }
    
    def extract_triage_level(self, text: str) -> str:
        """Extract triage level from text"""
        text_upper = text.upper()
        for level in ["RED", "YELLOW", "GREEN"]:
            if level in text_upper:
                return level
        return "UNCLASSIFIED"
    
    def evaluate_query(self, test_case: Dict, comparison_result: Dict) -> Dict:
        """Evaluate single query result"""
        
        expected = test_case.get("expected_triage")
        
        mistral_triage = comparison_result["mistral"]["triage_level"]
        groq_triage = comparison_result["groq"]["triage_level"]
        
        mistral_correct = mistral_triage == expected
        groq_correct = groq_triage == expected
        
        evaluation = {
            "query_id": test_case.get("query_id"),
            "expected": expected,
            "mistral": {
                "predicted": mistral_triage,
                "correct": mistral_correct,
                "rag_score": comparison_result["mistral"]["rag_adherence_score"],
                "latency": comparison_result["mistral"]["latency_s"],
            },
            "groq": {
                "predicted": groq_triage,
                "correct": groq_correct,
                "rag_score": comparison_result["groq"]["rag_adherence_score"],
                "latency": comparison_result["groq"]["latency_s"],
            },
            "model_alignment": comparison_result.get("model_similarity", {}).get("composite_score", 0),
        }
        
        return evaluation
    
    def add_result(self, evaluation: Dict):
        """Add evaluation result"""
        self.results.append(evaluation)
        
        # Update summary
        self.summary["total_queries"] += 1
        if evaluation["mistral"]["correct"]:
            self.summary["mistral_correct"] += 1
        if evaluation["groq"]["correct"]:
            self.summary["groq_correct"] += 1
        
        self.summary["mistral_avg_rag_score"] += evaluation["mistral"]["rag_score"]
        self.summary["groq_avg_rag_score"] += evaluation["groq"]["rag_score"]
        self.summary["model_similarity_avg"] += evaluation["model_alignment"]
    
    def finalize_summary(self):
        """Compute final metrics"""
        n = max(1, self.summary["total_queries"])
        
        self.summary["mistral_accuracy"] = round(
            self.summary["mistral_correct"] / n * 100, 2
        )
        self.summary["groq_accuracy"] = round(
            self.summary["groq_correct"] / n * 100, 2
        )
        self.summary["mistral_avg_rag_score"] = round(
            self.summary["mistral_avg_rag_score"] / n, 4
        )
        self.summary["groq_avg_rag_score"] = round(
            self.summary["groq_avg_rag_score"] / n, 4
        )
        self.summary["model_similarity_avg"] = round(
            self.summary["model_similarity_avg"] / n, 4
        )
    
    def print_summary(self):
        """Print evaluation summary"""
        print(f"\n{'='*80}")
        print("📊 BATCH EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Total Queries Tested: {self.summary['total_queries']}")
        print(f"\n🔵 MISTRAL PERFORMANCE:")
        print(f"  ✅ Correct: {self.summary['mistral_correct']}/{self.summary['total_queries']}")
        print(f"  📈 Accuracy: {self.summary['mistral_accuracy']}%")
        print(f"  📌 Avg RAG Adherence: {self.summary['mistral_avg_rag_score']}")
        
        print(f"\n🟡 GROQ PERFORMANCE:")
        print(f"  ✅ Correct: {self.summary['groq_correct']}/{self.summary['total_queries']}")
        print(f"  📈 Accuracy: {self.summary['groq_accuracy']}%")
        print(f"  📌 Avg RAG Adherence: {self.summary['groq_avg_rag_score']}")
        
        print(f"\n🔄 MODEL ALIGNMENT:")
        print(f"  Avg Similarity Score: {self.summary['model_similarity_avg']}")
        
        # Winner
        if self.summary["mistral_accuracy"] > self.summary["groq_accuracy"]:
            print(f"\n🏆 WINNER: Mistral (by {round(self.summary['mistral_accuracy'] - self.summary['groq_accuracy'], 2)}%)")
        elif self.summary["groq_accuracy"] > self.summary["mistral_accuracy"]:
            print(f"\n🏆 WINNER: Groq (by {round(self.summary['groq_accuracy'] - self.summary['mistral_accuracy'], 2)}%)")
        else:
            print(f"\n🏆 TIE: Both models equal accuracy")
        
        print(f"\n{'='*80}\n")
    
    def export_results(self, filename: str = "evaluation_results.json"):
        """Export results to JSON"""
        export_data = {
            "summary": self.summary,
            "detailed_results": self.results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Results exported to {filename}")


def load_test_dataset(filename: str = "medical_triage_dataset.json") -> List[Dict]:
    """Load test dataset"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        
        return data.get("medical_triage_dataset", [])
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {filename}")
        return []
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in dataset: {filename}")
        return []


def format_vital_signs(vital_signs: Dict) -> str:
    """Format vital signs for RAG context"""
    if not vital_signs:
        return ""
    
    formatted = "\n📊 VITAL SIGNS:\n"
    for key, value in vital_signs.items():
        formatted += f"  • {key.replace('_', ' ').title()}: {value}\n"
    
    return formatted


def run_batch_evaluation(
    dataset: List[Dict],
    use_rag: bool = True,
    sample_size: int = None,
    verbose: bool = True
):
    """Run batch evaluation on dataset"""
    
    print(f"\n{'='*80}")
    print("🏥 MEDICAL TRIAGE RAG SYSTEM - BATCH EVALUATION")
    print(f"{'='*80}\n")
    
    # Load RAG components
    rag_loaded = load_rag_components()
    if use_rag and not rag_loaded:
        print("⚠ RAG components not available, running models without RAG context")
        use_rag = False
    
    print(f"📝 Test Configuration:")
    print(f"  • Use RAG: {use_rag}")
    print(f"  • Total test cases: {len(dataset)}")
    if sample_size:
        print(f"  • Sample size: {sample_size}")
    print(f"\n{'='*80}\n")
    
    # Limit to sample size if specified
    test_cases = dataset[:sample_size] if sample_size else dataset
    
    evaluator = TriageEvaluator()
    
    for i, test_case in enumerate(test_cases, 1):
        query_id = test_case.get("query_id", f"QUERY_{i}")
        query = test_case.get("query", "")
        base_answer = test_case.get("base_answer")
        
        print(f"\n[{i}/{len(test_cases)}] {query_id}")
        print(f"{'-'*80}")
        
        try:
            # Add vital signs to query if available
            vital_signs = test_case.get("vital_signs", {})
            if vital_signs:
                query_with_vitals = query + format_vital_signs(vital_signs)
            else:
                query_with_vitals = query
            
            # Run comparison
            comparison_result = compare_models_with_rag(
                query=query_with_vitals,
                base_answer=base_answer,
                use_rag=use_rag
            )
            
            # Evaluate
            evaluation = evaluator.evaluate_query(test_case, comparison_result)
            evaluator.add_result(evaluation)
            
            # Print results for this query
            if verbose:
                print(f"Expected: {evaluation['expected']}")
                print(f"Mistral: {evaluation['mistral']['predicted']} {'✅' if evaluation['mistral']['correct'] else '❌'}")
                print(f"Groq:    {evaluation['groq']['predicted']} {'✅' if evaluation['groq']['correct'] else '❌'}")
                print(f"RAG Adherence - Mistral: {evaluation['mistral']['rag_score']}, Groq: {evaluation['groq']['rag_score']}")
            
        except Exception as e:
            print(f"❌ Error processing {query_id}: {str(e)}")
            continue
    
    # Finalize and print summary
    evaluator.finalize_summary()
    evaluator.print_summary()
    
    # Export results
    timestamp = int(time.time())
    evaluator.export_results(f"evaluation_results_{timestamp}.json")
    
    return evaluator


def interactive_single_test():
    """Test single query interactively"""
    
    print(f"\n{'='*80}")
    print("🔬 SINGLE QUERY TEST MODE")
    print(f"{'='*80}\n")
    
    # Load RAG
    rag_loaded = load_rag_components()
    
    query = input("📝 Enter medical query: ").strip()
    if not query:
        print("❌ No query provided")
        return
    
    base_answer = input("📖 Enter base answer (optional): ").strip()
    use_rag = input("🔍 Use RAG context? (y/n, default=y): ").strip().lower() != "n"
    
    if use_rag and not rag_loaded:
        print("⚠ RAG not available, proceeding without context")
        use_rag = False
    
    print(f"\n{'='*80}")
    print("🔄 Running comparison...")
    print(f"{'='*80}\n")
    
    result = compare_models_with_rag(
        query=query,
        base_answer=base_answer if base_answer else None,
        use_rag=use_rag
    )
    
    print_comparison_report(result)
    
    # Option to save
    save = input("💾 Save result to JSON? (y/n): ").strip().lower() == "y"
    if save:
        timestamp = int(time.time())
        filename = f"single_test_{timestamp}.json"
        save_comparison_result(result, filename)


def main():
    """Main entry point"""
    
    print(f"\n{'='*80}")
    print("🏥 MEDICAL TRIAGE RAG EVALUATION SYSTEM")
    print(f"{'='*80}\n")
    
    print("Select test mode:")
    print("1. Batch evaluation (all test cases)")
    print("2. Batch evaluation (sample)")
    print("3. Single query test")
    print("4. Exit")
    
    choice = input("\n👉 Enter choice (1-4): ").strip()
    
    if choice == "1":
        dataset = load_test_dataset()
        if dataset:
            run_batch_evaluation(dataset, use_rag=True)
    
    elif choice == "2":
        dataset = load_test_dataset()
        if dataset:
            try:
                sample_size = int(input("How many test cases? "))
                run_batch_evaluation(dataset, use_rag=True, sample_size=sample_size)
            except ValueError:
                print("❌ Invalid number")
    
    elif choice == "3":
        interactive_single_test()
    
    elif choice == "4":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
