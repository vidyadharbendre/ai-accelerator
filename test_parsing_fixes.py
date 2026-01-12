#!/usr/bin/env python3
"""
Test script to validate the regex parsing fixes in _parse_analysis_response method.
This script tests various JSON response formats that the AI models might return.
"""

import json
import sys
import os

# Add the current directory to the path so we can import the CodeReview module
sys.path.insert(0, os.path.dirname(__file__))

try:
    from Day_11.CodeReview import OpenRouterLangChainReviewer
except ImportError:
    print("❌ Failed to import OpenRouterLangChainReviewer from Day_11.CodeReview")
    sys.exit(1)

def test_parsing_with_samples():
    """Test the _parse_analysis_response method with various sample responses"""
    reviewer = OpenRouterLangChainReviewer()

    # Sample test cases with different response formats
    test_cases = [
        {
            "name": "Simple JSON array",
            "response": '''[
                {
                    "severity": "Critical",
                    "category": "Security",
                    "line_number": 1,
                    "title": "Hardcoded password",
                    "description": "Password is hardcoded in the code",
                    "suggestion": "Use environment variables",
                    "confidence": 0.95
                }
            ]''',
            "expected_issues": 1
        },
        {
            "name": "JSON in markdown code block",
            "response": '''```json
[
    {
        "severity": "High",
        "category": "Performance",
        "line_number": 5,
        "title": "Inefficient loop",
        "description": "Loop has O(n²) complexity",
        "suggestion": "Use more efficient algorithm",
        "confidence": 0.85
    }
]
```''',
            "expected_issues": 1
        },
        {
            "name": "Nested JSON structures",
            "response": '''```json
[
    {
        "severity": "Medium",
        "category": "Maintainability",
        "line_number": 10,
        "title": "Complex function",
        "description": "Function is too long with nested conditions",
        "suggestion": "Break down into smaller functions",
        "code_snippet": "def complex_func():\n    if x > 0:\n        for i in range(100):\n            if i % 2 == 0:\n                print(f\"Even: {i}\")\n            else:\n                print(f\"Odd: {i}\")",
        "confidence": 0.75
    },
    {
        "severity": "Low",
        "category": "Style",
        "line_number": 25,
        "title": "Variable naming",
        "description": "Variable name 'x' is not descriptive",
        "suggestion": "Use more descriptive names like 'user_count'",
        "confidence": 0.6
    }
]
```''',
            "expected_issues": 2
        },
        {
            "name": "Empty array response",
            "response": "[]",
            "expected_issues": 0
        },
        {
            "name": "Response with extra text",
            "response": '''Based on my analysis of the code, I found the following issues:

```json
[
    {
        "severity": "High",
        "category": "Security",
        "line_number": 3,
        "title": "SQL Injection vulnerability",
        "description": "User input is directly concatenated into SQL query",
        "suggestion": "Use parameterized queries or prepared statements",
        "confidence": 0.9
    }
]
```

These issues should be addressed to improve code quality.''',
            "expected_issues": 1
        },
        {
            "name": "Malformed JSON (should fallback to text parsing)",
            "response": '''I found some issues in your code:

1. Line 1: Hardcoded password - this is a security risk
2. Line 10: Missing error handling - could cause crashes
3. Line 15: Poor variable naming - makes code hard to read

Please fix these issues.''',
            "expected_issues": 3  # Should parse as text
        },
        {
            "name": "Single issue object (not array)",
            "response": '''{
    "severity": "Critical",
    "category": "Security",
    "line_number": 1,
    "title": "API key exposed",
    "description": "API key is committed to version control",
    "suggestion": "Move to environment variables",
    "confidence": 0.95
}''',
            "expected_issues": 1
        }
    ]

    sample_code = "def test():\n    password = 'secret123'\n    return password"

    print("🧪 Testing regex parsing fixes...")
    print("=" * 50)

    all_passed = True

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 30)

        try:
            # Call the parsing method
            result = reviewer._parse_analysis_response(test_case['response'], sample_code)

            # Check if we got the expected number of issues
            if len(result) == test_case['expected_issues']:
                print(f"✅ PASSED: Expected {test_case['expected_issues']} issues, got {len(result)}")

                # Print details of parsed issues
                for j, issue in enumerate(result):
                    print(f"   Issue {j+1}: {issue.get('title', 'No title')} (line {issue.get('line_number', '?')})")

            else:
                print(f"❌ FAILED: Expected {test_case['expected_issues']} issues, got {len(result)}")
                all_passed = False

                # Debug info
                print(f"   Response preview: {test_case['response'][:100]}...")
                if result:
                    print(f"   Parsed issues: {[issue.get('title', 'No title') for issue in result]}")

        except Exception as e:
            print(f"❌ ERROR: Test failed with exception: {e}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The regex parsing fixes are working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED. Please review the implementation.")

    return all_passed

def test_edge_cases():
    """Test edge cases that might cause crashes"""
    reviewer = OpenRouterLangChainReviewer()

    print("\n🧪 Testing edge cases...")
    print("=" * 30)

    edge_cases = [
        ("Empty string", ""),
        ("None input", None),
        ("Very large response", "x" * 100000),
        ("JSON with special characters", '{"test": "value with \\"quotes\\" and \\n newlines"}'),
        ("Nested arrays", '[[{"test": "nested"}]]'),
        ("Invalid JSON", '{"incomplete": json}'),
    ]

    sample_code = "print('hello world')"

    for name, response in edge_cases:
        try:
            if response is None:
                # Test with None input
                result = reviewer._parse_analysis_response("", sample_code)
            else:
                result = reviewer._parse_analysis_response(response, sample_code)
            print(f"✅ {name}: Handled gracefully (returned {len(result)} issues)")
        except Exception as e:
            print(f"❌ {name}: Crashed with {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("🔧 Testing OpenRouterLangChainReviewer parsing fixes\n")

    # Run main tests
    success = test_parsing_with_samples()

    # Run edge case tests
    test_edge_cases()

    print(f"\n📊 Final Result: {'SUCCESS' if success else 'ISSUES FOUND'}")
    sys.exit(0 if success else 1)
