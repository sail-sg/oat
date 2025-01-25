# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slice LLM outputs into reasoning steps."""

from typing import List


def get_slicer(name: str):
    if "gsm8k" in name:
        return slice_gsm8k
    else:
        raise NotImplementedError


def slice_gsm8k(
    solution: str, delimiter: str = "\n", answer_prefix: str = "\n#### "
) -> List[int]:
    """
    Reference to VinePPO codebase: https://github.com/McGill-NLP/VinePPO.

    Args:
        solution: The solution text.

    Returns:
        A list of indices where each index corresponds to the start of a reasoning step.
        Example:
        >>> solution = '...'
        >>> indices = slice_gsm8k(solution)
        >>> steps = [solution[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]
    """

    if answer_prefix is None:
        sol_without_answer, answer = solution, None
    else:
        try:
            solution_parts = solution.split(answer_prefix)
            if len(solution_parts) < 2:
                sol_without_answer, answer = solution, None
            else:
                sol_without_answer, answer = solution_parts
        except Exception:
            print(solution)
            raise

    steps = sol_without_answer.split(delimiter)

    # Merge first empty steps to the first non-empty step
    first_non_empty_step_idx = None
    for i, step in enumerate(steps):
        if step.strip() != "":
            first_non_empty_step_idx = i
            break

    if first_non_empty_step_idx is not None and first_non_empty_step_idx > 0:
        new_first_step = delimiter.join(steps[: first_non_empty_step_idx + 1])

        steps = [new_first_step] + steps[first_non_empty_step_idx + 1 :]

    if answer is not None:
        # We want to merge the last step with the answer

        # Find last non-empty step index
        last_non_empty_step_idx = None
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].strip() != "":
                last_non_empty_step_idx = i
                break

        if last_non_empty_step_idx is None:
            # Then it means the entire solution is a single step
            last_non_empty_step_idx = 0

        new_last_step = delimiter.join(steps[last_non_empty_step_idx:])
        # Also merge the last step with the answer
        new_last_step = f"{new_last_step}{answer_prefix}{answer}"
        steps = steps[:last_non_empty_step_idx] + [new_last_step]

    reconstructed_solution = delimiter.join(steps)
    assert reconstructed_solution == solution, f"{reconstructed_solution} != {solution}"

    # Find the indices of the reasoning steps
    indices = [0]
    for i, step in enumerate(steps):
        if i == 0:
            indices.append(indices[-1] + len(step))
        else:
            indices.append(indices[-1] + len(step) + len(delimiter))

    assert indices[-1] == len(solution), f"{indices[-1]} != {len(solution)}"

    return indices
