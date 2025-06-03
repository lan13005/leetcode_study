# References

This is based on https://blog.algomaster.io/p/15-leetcode-patterns but with additional information:

# Montonic Stack

Pattern for problems that require finding next greater/smaller element

## Explanation:

1. Use a stack to keep track of elements for which we haven't found the next greater element yet.
2. Iterate through the array, and for each element, pop elements from the stack until you find a greater element.
3. If the stack is not empty, set the result for index at the top of the stack to current element.
4. Push the current element onto the stack.

## LeetCode Problems:

- Next Greater Element I (LeetCode #496)
- Daily Temperatures (LeetCode #739)
- Largest Rectangle in Histogram (LeetCode #84)

## Example:

```text
Question: Next greater for [3, 1, 4]
Answer: [4, 4, -1]

# Initialize stack and result
stack = []
result = [-1, -1, -1]

# Walk left to right
i=0; nums[0] = 3 nothing to compare; stack = [0] # push onto stack
i=1; nums[1] = 1 < nums[0] = 3; stack = [0, 1]   # push onto stack
i=2; nums[2] = 4 > nums[1] = 1; pop 1 from stack -> stack = [0] set result[1] = 4
i=2; nums[2] = 4 > nums[0] = 3; pop 0 from stack -> stack = []  set result[0] = 4
i=2; nums[2] = 4 nothing to compare; stack = [2] # push onto stack
result = [4, 4, -1]
```

```python
def next_greater(nums):
    res = [-1] * len(nums)
    stack = []

    for i in range(len(nums)):
        # pop all smaller elements from the stack
        while stack and nums[i] > nums[stack[-1]]:
            prev_index = stack.pop()
            res[prev_index] = nums[i]
        stack.append(i)

    return res

print(next_greater([3, 1, 4]))  # Output: [4, 4, -1]
```
