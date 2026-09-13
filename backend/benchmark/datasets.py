"""
Benchmark datasets for Local AI Arena.

Provides standard, reproducible task suites for evaluating multi-agent collaboration:
1. Coding & Algorithmic Problem Solving
2. Logical & Analytical Reasoning
3. Multi-Step Mathematical Deduction
4. Knowledge Synthesis & Analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkTask:
    id: str
    category: str
    title: str
    prompt: str
    evaluation_criteria: str
    expected_keywords: list[str]


BUILTIN_DATASETS: dict[str, list[BenchmarkTask]] = {
    "coding": [
        BenchmarkTask(
            id="code-01",
            category="coding",
            title="Longest Palindromic Substring",
            prompt=(
                "Write an optimal Python function `longest_palindromic_substring(s: str) -> str` "
                "that returns the longest palindromic substring of s. "
                "Include unit tests covering edge cases (empty string, single character, even/odd lengths) "
                "and explain the time and space complexity."
            ),
            evaluation_criteria="Correct expansion or Manacher's algorithm, edge-case coverage, complexity analysis.",
            expected_keywords=["def", "palindrome", "return", "complexity", "O("],
        ),
        BenchmarkTask(
            id="code-02",
            category="coding",
            title="LRU Cache with O(1) Operations",
            prompt=(
                "Implement an LRU (Least Recently Used) Cache class in Python with `get(key)` and `put(key, value)` "
                "methods operating in O(1) average time complexity. "
                "Use a doubly linked list combined with a hash map. Include explanations and code."
            ),
            evaluation_criteria="Doubly linked list + dict, correct eviction of least recently used, O(1) ops.",
            expected_keywords=["class LRUCache", "def get", "def put", "Node", "head", "tail"],
        ),
    ],
    "reasoning": [
        BenchmarkTask(
            id="reason-01",
            category="reasoning",
            title="Knights, Knaves, and Spies",
            prompt=(
                "On an island, inhabitants are either Knights (who always tell the truth) or Knaves "
                "(who always lie). You meet three people: A, B, and C.\n"
                "A says: 'All of us are knaves.'\n"
                "B says: 'Exactly one of us is a knight.'\n"
                "Determine the identity of each person (Knight or Knave). Walk step-by-step through your logic."
            ),
            evaluation_criteria="A must be a Knave; B must be a Knight; C must be a Knave.",
            expected_keywords=["A is a knave", "B is a knight", "C is a knave", "contradiction"],
        ),
        BenchmarkTask(
            id="reason-02",
            category="reasoning",
            title="River Crossing Constraint Puzzle",
            prompt=(
                "A farmer needs to cross a river with a wolf, a goat, and a cabbage. "
                "His boat can only carry himself and one item at a time. "
                "If left alone together without the farmer, the wolf will eat the goat, or the goat will eat the cabbage. "
                "Provide the minimal sequence of boat crossings to get all three safely across, explaining why each step avoids danger."
            ),
            evaluation_criteria="7-step sequence crossing goat first, then returning with goat after bringing cabbage/wolf.",
            expected_keywords=["cross", "goat", "wolf", "cabbage", "return"],
        ),
    ],
    "math": [
        BenchmarkTask(
            id="math-01",
            category="math",
            title="Modular Arithmetic & Divisibility",
            prompt=(
                "Find the remainder when 7^2025 is divided by 100. Show every step using Euler's totient theorem "
                "or powers of 7 modulo 100."
            ),
            evaluation_criteria="phi(100)=40, 2025 mod 40 = 25, 7^25 mod 100 calculation.",
            expected_keywords=["totient", "modulo", "remainder", "7^"],
        ),
    ],
}


def get_available_datasets() -> list[dict]:
    """List available benchmark dataset suites."""
    return [
        {
            "id": "coding",
            "name": "Algorithmic & Code Synthesis (2 tasks)",
            "description": "Evaluates Python algorithmic correctness, edge case handling, and complexity rigor.",
            "task_count": len(BUILTIN_DATASETS["coding"]),
        },
        {
            "id": "reasoning",
            "name": "Logical & Constraint Reasoning (2 tasks)",
            "description": "Multi-step logic puzzles requiring deductive rigor and contradiction checks.",
            "task_count": len(BUILTIN_DATASETS["reasoning"]),
        },
        {
            "id": "math",
            "name": "Multi-Step Mathematical Deduction (1 task)",
            "description": "Modular arithmetic and discrete mathematics with explicit proofs.",
            "task_count": len(BUILTIN_DATASETS["math"]),
        },
    ]
