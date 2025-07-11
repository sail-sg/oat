import os
from oat.oracles.cal_oracle import CALOracle
import time

test_cases = [
    {
        "question": "A regular octagon is inscribed in a circle of radius 1. Find the area of the octagon.",
        "correct_solution": "The octagon can be divided into 8 congruent isosceles triangles... the total area is 2*sqrt(2).",
        "incorrect_solution": "An octagon has 8 sides... The area is simply the perimeter times the radius, which is 8s.",
        "expected_segment": "The area is simply the perimeter times the radius, which is 8s."
    },
    {
        "question": "What is the remainder when $1! + 2! + 3! + ... + 100!$ is divided by 12?",
        "correct_solution": "For any n >= 4, n! will be a multiple of 12... We only need to sum the first three terms: 1 + 2 + 6 = 9. So the remainder is 9.",
        "incorrect_solution": "We can see that starting from 5!, all terms end in 0. The sum of the last digits will be 1+2+6+4+0=13. The last digit is 3. This means the remainder when divided by 12 must also be 3.",
        "expected_segment": "This means the remainder when divided by 12 must also be 3."
    },
    {
        "question": "If a car travels at 60 miles per hour, how far does it travel in 30 minutes?",
        "correct_solution": "First, we must work in consistent units. We convert 30 minutes to hours, which is 0.5 hours. Then, we use the formula Distance = Speed * Time. So, Distance = 60 mph * 0.5 hours = 30 miles.",
        "incorrect_solution": "To find the distance, we multiply the speed by the time. The speed is 60 and the time is 30. So the distance is 60 * 30 = 1800 miles.",
        "expected_segment": "So the distance is 60 * 30 = 1800 miles."
    },
    {
        "question": "What is the probability of rolling a 7 on a standard six-sided die?",
        "correct_solution": "A standard six-sided die has faces numbered 1, 2, 3, 4, 5, and 6. It is impossible to roll a 7 as it is not one of the possible outcomes. Therefore, the probability of an impossible event is 0.",
        "incorrect_solution": "There are 6 possible outcomes when rolling a die. We are interested in one specific outcome, which is rolling a 7. Therefore, the probability is 1 out of 6, or 1/6.",
        "expected_segment": "Therefore, the probability is 1 out of 6, or 1/6."
    },
    {
        "question": "If $f(x) = 3x-2$ and $g(x) = x^2+1$, what is $f(g(2))$?",
        "correct_solution": "First, we evaluate the inner function, g(2). We plug 2 into g(x): $g(2) = (2)^2 + 1 = 4 + 1 = 5$. Now we take this result, 5, and plug it into f(x). $f(5) = 3(5) - 2 = 15 - 2 = 13$. Therefore, f(g(2)) = 13.",
        "incorrect_solution": "To solve this, we find f(2) and g(2) and combine them. f(2) is $3(2)-2=4$. g(2) is $(2)^2+1=5$. To find f(g(2)), we can multiply these results, which gives $4 \\times 5 = 20$.",
        "expected_segment": "To find f(g(2)), we can multiply these results, which gives $4 \\times 5 = 20$."
    }
]

def run_all_tests():
    print("--- Starting CALOracle Final Test (with Gemini) ---")

    if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
        print("FATAL: GOOGLE_API_KEY environment variable not set.")
        return

    try:
        print("Instantiating CALOracle...")
        cal_oracle = CALOracle(
            cal_model_name="gemini-1.5-flash-latest",
            few_shot_path="scripts/cal_few_shot_examples.json"
        )
        print("Oracle instantiated successfully.\n")
    except Exception as e:
        print(f"FAILED to instantiate CALOracle: {e}")
        return

    passed_count = 0
    for i, case in enumerate(test_cases):

        print(f"--- Running Test {i+1}/{len(test_cases)} ---")
        print(f"  Question: {case['question']}")
        
        try:
            expected_segment = case['expected_segment']
            
            _, metric_output = cal_oracle.get_reward(
                inputs=[case['question']],
                responses=[case['incorrect_solution']],
                references=[case['correct_solution']]
            )
            error_segment = metric_output["cal_outputs"][0]["error_segment"]

        except KeyError:
            print(f"FAILED: Test case {i+1} is missing a required key.")
            continue
        except Exception as e:
            print(f"FAILED during API call for test {i+1}: {e}")
            continue

        clean_expected = expected_segment.strip().rstrip(',.')
        clean_actual = error_segment.strip().rstrip(',.')
        
        print(f"  Expected: '{clean_expected}'")
        print(f"  Actual:   '{clean_actual}'")

        if clean_expected == clean_actual:
            print("  Result:   PASSED\n")
            passed_count += 1
        else:
            print("  Result:   FAILED\n")
        
        time.sleep(1) 

    print("--- Test Summary ---")
    print(f"  {passed_count} / {len(test_cases)} tests passed.")
    if passed_count == len(test_cases):
        print("All tests passed successfully! Phase 1 is complete.")
    else:
        print("Review the failed tests.")

if __name__ == "__main__":
    run_all_tests()