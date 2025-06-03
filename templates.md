# Templates

## Sliding Window

- [Reference](https://leetcode.com/problems/frequency-of-the-most-frequent-element/solutions/1175088/C++-Maximum-Sliding-Window-Cheatsheet-Template/)
- The sliding window algorithm is used to solve subarray/substring problems efficiently in O(N) time
- Key insight: Use two pointers (left/right) to maintain a window, expand/shrink as needed
- Considerations:
    - When should you expand the window (move `right`)?
    - When should you shrink the window (move `left`)?
    - When should you update the result?
- Questions that can use sliding window:
    - Find longest/shortest subarray/substring that meets certain conditions
    - Find all subarrays/substrings of fixed length that meet conditions
    - Count number of valid subarrays/substrings

```python
def sliding_window(s: str) -> int:
    def shrink_condition(window) -> bool:
        # Define when window needs to shrink
        pass
    
    # Use appropriate data structure to track window contents
    # - For character frequency: use dict/Counter
    # - For sum/count: use int variables
    window = {}
    left = right = 0
    result = 0
    while right < len(s):
        c = s[right] # character entering the window
        right += 1 # expand window
        window[c] = window.get(c, 0) + 1 # update window data

        while shrink_condition(window):
            d = s[left] # character leaving the window
            left += 1 # shrink window
            window[d] -= 1 # update window data
            if window[d] == 0:
                del window[d]

        # Update result (can be inside shrink loop for some problems)
        result = max(result, right - left)

    return result
```

## Two Pointers

- [Reference](https://leetcode.com/discuss/post/1688903/solved-all-two-pointers-problems-in-100-z56cn/)
- Pattern for finding pairs or elements in arrays/strings that meet specific criteria
- Considerations:
    - How to initialize the pointers (start/end vs both at start)?
    - What condition determines pointer movement?
    - When to update the result?
- Common variants:
    - **Opposite ends**: Two pointers from start and end (sorted arrays)
    - **Same direction**: Both pointers move in same direction (slow/fast)
    - **Two inputs**: One pointer per input array

```python
# Variant 1: Opposite ends (for sorted arrays)
def two_pointers_opposite(nums: list, target: int) -> list:
    def process(left_val, right_val):
        # Define how to process the two values
        pass
    
    left, right = 0, len(nums) - 1

    while left < right:
        result = process(nums[left], nums[right])

        if result == target:
            return [left, right]
        elif result < target:
            left += 1
        else:
            right -= 1

    return []

# Variant 2: Same direction (slow/fast pointers)
def two_pointers_same_direction(nums: list) -> int:
    def condition(value) -> bool:
        # Define condition for including element
        pass
    
    slow = fast = 0

    while fast < len(nums):
        if condition(nums[fast]):
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    
    return slow # new length

# Variant 3: Two inputs
def two_pointers_two_inputs(arr1: list, arr2: list) -> list:
    def compare(val1, val2) -> bool:
        # Define comparison logic
        pass
    
    i = j = 0
    result = []

    while i < len(arr1) and j < len(arr2):
        if compare(arr1[i], arr2[j]):
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # Handle remaining elements
    result.extend(arr1[i:])
    result.extend(arr2[j:])

    return result
```

## Binary Search

- [Reference](https://leetcode.com/discuss/post/786126/python-powerful-ultimate-binary-search-t-rwv8/)
- Most binary search problems can be solved with the following template.
  - Considerations:
    - How to properly define `left` and `right`?
    - How to properly define `condition`?
- Questions that might not appear to be binary search problems but if you can rephase as:
  - if `condition(k) is True` then `condition(k+1) is True`
  - alternative view: `condition` function is monotonic
    - Search space halves until `left == right` which is the minimum value that satisfies the condition

```python
def binary_search(array) -> int:
    def condition(value) -> bool:
        pass

    left, right = min(search_space), max(search_space) # could be [0, n], [1, n] etc. Depends on problem
    while left < right:
        mid = left + (right - left) // 2
        if condition(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

## Monotonic Stack

- [Reference](https://leetcode.com/discuss/post/2347639/a-comprehensive-guide-and-template-for-m-irii/)
- Stack that maintains elements in monotonic order (increasing or decreasing)
- Key insight: Find "next/previous greater/smaller" elements efficiently in O(N) time
- Considerations:
    - What monotonic order to maintain (increasing/decreasing, strict/weak)?
    - When to pop elements from stack?
    - What to do with popped elements?
- Questions that can use monotonic stack:
    - Next/previous greater/smaller element problems
    - Find elements that can "see" each other
    - Rectangle problems (largest rectangle, building heights)

```python
# Template for NEXT greater/smaller elements
def monotonic_stack_next(arr: list, cmp_fn) -> list:
    """
    Find next element that satisfies comparison function.
    
    Examples:
    - Next greater: cmp_fn = lambda curr, top: curr > top
    - Next smaller: cmp_fn = lambda curr, top: curr < top
    - Next greater/equal: cmp_fn = lambda curr, top: curr >= top

    Main difference between Next and Previous is Assignment position (inside vs outside while loop)
    """
    n = len(arr)
    result = [-1] * n
    stack = []  # stores indices
    
    for i in range(n):
        while stack and cmp_fn(arr[i], arr[stack[-1]]):
            j = stack.pop()
            # Option 1: Next greater/smaller
            # result[j] = i  # or arr[i] for values instead of indices
        # Option 2: Previous greater/smaller
        # if stack:
        #     result[i] = stack[-1]
        stack.append(i)
    
    return result
```
