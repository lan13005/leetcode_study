# References

This is based on https://blog.algomaster.io/p/15-leetcode-patterns but with additional information:

# Prefix Sum

Pattern for problems involving multiple sum queries on subarrays or calculating cumulative sums

## Explanation:

1. Preprocess the array to create a new array where each element at index `i` represents the sum from start up to `i`
2. To find sum between indices `i` and `j`, use formula: `P[j] - P[i-1]`
3. Handle edge cases for queries starting at index 0

## LeetCode Problems:

- Range Sum Query - Immutable (LeetCode [303](https://leetcode.com/problems/range-sum-query-immutable/))
- Contiguous Array (LeetCode [525](https://leetcode.com/problems/contiguous-array/))
- Subarray Sum Equals K (LeetCode [560](https://leetcode.com/problems/subarray-sum-equals-k/))

## Example:

```text
Question: Find sum between indices 1 and 3 for [1, 2, 3, 4, 5, 6]
Answer: 9 (2 + 3 + 4)

# Create prefix sum array
nums = [1, 2, 3, 4, 5, 6]
prefix = [1, 3, 6, 10, 15, 21]

# Query sum from index 1 to 3
sum = prefix[3] - prefix[0] = 10 - 1 = 9
```

```python
def range_sum_query(nums, queries):
    # Question: Given an integer array nums and multiple queries asking for the sum of elements between indices left and right (inclusive), efficiently answer all queries.
    # Key Insight: 
    #  - Precompute prefix sums to avoid recalculating subarray sums
    #  - sum(left, right) = prefix[right+1] - prefix[left]
    #  - Time: O(n) preprocessing + O(1) per query
    # Build prefix sum array
    prefix = [0] * (len(nums) + 1)
    for i in range(len(nums)):
        prefix[i + 1] = prefix[i] + nums[i]
    
    results = []
    for left, right in queries:
        # Sum from left to right (inclusive)
        sum_range = prefix[right + 1] - prefix[left]
        results.append(sum_range)
    
    return results

nums = [1, 2, 3, 4, 5, 6]
queries = [(1, 3), (0, 2), (2, 5)]
print(range_sum_query(nums, queries))  # Output: [9, 6, 18]
```

# Two Pointers

Pattern for finding pairs or elements in sorted arrays that meet specific criteria.
This is a very common pattern in leetcode:
- [List of two pointers problems](https://leetcode.com/discuss/post/1688903/solved-all-two-pointers-problems-in-100-z56cn/)
- See [[pattern-templates#Two Pointers]]

## Explanation:

1. Initialize two pointers, one at start (`left`) and one at end (`right`) of array
2. Check condition with elements at both pointers
3. Move left pointer right if sum/condition is too small
4. Move right pointer left if sum/condition is too large
5. Continue until pointers meet or condition is satisfied

## LeetCode Problems:

- Two Sum II - Input Array is Sorted (LeetCode [167](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/))
- 3Sum (LeetCode [15](https://leetcode.com/problems/3sum/))
- Container With Most Water (LeetCode [11](https://leetcode.com/problems/container-with-most-water/))

## Example:

```text
Question: Find two numbers that add up to target 6 in [1, 2, 3, 4, 6]
Answer: [1, 3] (indices of numbers 2 and 4)

nums = [1, 2, 3, 4, 6], target = 6
left = 0, right = 4
nums[0] + nums[4] = 1 + 6 = 7 > 6, move right left
left = 0, right = 3  
nums[0] + nums[3] = 1 + 4 = 5 < 6, move left right
left = 1, right = 3
nums[1] + nums[3] = 2 + 4 = 6 = target, found!
```

```python
def two_sum_sorted(nums, target):
    # Question: Given a sorted array and a target sum, find two numbers that add up to the target. Return their indices.
    # Key Insight: 
    #  - Use two pointers from start and end
    #  - If sum too small, move left pointer right
    #  - If sum too large, move right pointer left
    #  - Time: O(n), Space: O(1)
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

nums = [1, 2, 3, 4, 6]
target = 6
print(two_sum_sorted(nums, target))  # Output: [1, 3]
```

# Fast & Slow Pointers

Another example of [Two Pointers](#two-pointers). Pattern for detecting cycles using two pointers moving at different speeds (Tortoise and Hare)

## Explanation:

1. Initialize slow pointer moving one step at a time
2. Initialize fast pointer moving two steps at a time
3. If there's a cycle, fast pointer will eventually meet slow pointer
4. If fast pointer reaches end/null, no cycle exists
5. Can be used to find cycle start, middle of list, etc.

## LeetCode Problems:

- Linked List Cycle (LeetCode [141](https://leetcode.com/problems/linked-list-cycle/))
- Happy Number (LeetCode [202](https://leetcode.com/problems/happy-number/))
- Find the Duplicate Number (LeetCode [287](https://leetcode.com/problems/find-the-duplicate-number/))

## Example:

```text
Question: Detect cycle in linked list 1->2->3->4->2 (4 points back to 2)
Answer: True (cycle exists)

Step 1: slow=1, fast=1
Step 2: slow=2, fast=3  
Step 3: slow=3, fast=2 (fast wrapped around)
Step 4: slow=4, fast=4 (they meet - cycle detected!)
```

```python
def has_cycle(head):
    # Question: Given the head of a linked list, determine if the linked list has a cycle in it.
    # Key Insight: 
    #  - Use Floyd's cycle detection (tortoise and hare)
    #  - Slow pointer moves 1 step, fast pointer moves 2 steps
    #  - If cycle exists, fast will eventually meet slow
    #  - Time: O(n), Space: O(1)
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

# Example usage
# Create linked list with cycle: 1->2->3->4->2
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = head.next  # Create cycle

print(has_cycle(head))  # Output: True
```

# Sliding Window

Pattern for finding optimal subarray or substring by maintaining a window of elements
- [[pattern-templates#Sliding Window]]

## Explanation:

1. Initialize window with first k elements (for fixed size) or expand until condition met
2. Slide window by removing leftmost element and adding new rightmost element
3. Update optimal result as window slides
4. For variable size: expand right to include elements, shrink left when condition violated

## LeetCode Problems:

- Maximum Average Subarray I (LeetCode [643](https://leetcode.com/problems/maximum-average-subarray-i/))
- Longest Substring Without Repeating Characters (LeetCode [3](https://leetcode.com/problems/longest-substring-without-repeating-characters/))
- Minimum Window Substring (LeetCode [76](https://leetcode.com/problems/minimum-window-substring/))

## Example:

```text
Question: Find maximum sum of subarray of size 3 in [2, 1, 5, 1, 3, 2]
Answer: 9 (subarray [5, 1, 3])

nums = [2, 1, 5, 1, 3, 2], k = 3

# Initial window [2, 1, 5]
window_sum = 2 + 1 + 5 = 8, max_sum = 8

# Slide to [1, 5, 1]: remove 2, add 1
window_sum = 8 - 2 + 1 = 7, max_sum = 8

# Slide to [5, 1, 3]: remove 1, add 3  
window_sum = 7 - 1 + 3 = 9, max_sum = 9

# Slide to [1, 3, 2]: remove 5, add 2
window_sum = 9 - 5 + 2 = 6, max_sum = 9
```

```python
def max_sum_subarray(nums, k):
    # Question: Given an array and integer k, find the maximum sum of any contiguous subarray of size k.
    # Key Insight: 
    #  - Use sliding window technique
    #  - Calculate first window sum, then slide by removing left element and adding right element
    #  - Time: O(n), Space: O(1)
    if len(nums) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

nums = [2, 1, 5, 1, 3, 2]
k = 3
print(max_sum_subarray(nums, k))  # Output: 9
```

# LinkedList In-place Reversal

Pattern for reversing parts of a linked list without using extra space
## Explanation:

1. Identify the start and end positions of reversal
2. Keep track of nodes before and after the reversal section
3. Reverse the section by adjusting next pointers
4. Connect the reversed section back to the rest of the list

## LeetCode Problems:

- Reverse Linked List (LeetCode [206](https://leetcode.com/problems/reverse-linked-list/))
- Reverse Linked List II (LeetCode [92](https://leetcode.com/problems/reverse-linked-list-ii/))
- Swap Nodes in Pairs (LeetCode [24](https://leetcode.com/problems/swap-nodes-in-pairs/))

## Example:

```text
Question: Reverse sublist from position 2 to 4 in [1,2,3,4,5]
Answer: [1,4,3,2,5]

Original: 1 -> 2 -> 3 -> 4 -> 5
After:    1 -> 4 -> 3 -> 2 -> 5

Steps:
1. Find position 2 (node with value 2)
2. Reverse section 2->3->4 to get 4->3->2
3. Connect: 1 -> 4->3->2 -> 5
```

![[linked-list-between.png]]

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_between(head, left, right):
    # Question: Reverse nodes between positions left and right (1-indexed).
    # Key Insight:
    #   - Use a dummy node to handle edge cases
    #   - Locate node just before the reversal region
    #   - Reverse nodes by front-inserting into the sublist
    #   - Reconnect the reversed sublist into the main list
	if not head or left == right:
		return head

	dummy = ListNode(0, head)
	prev = dummy

	for _ in range(left - 1):
		prev = prev.next

	cur = prev.next
	for _ in range(right - left):
		temp = cur.next
		cur.next = temp.next
		temp.next = prev.next
		prev.next = temp

	return dummy.next
```

# Monotonic Stack

Pattern for problems that require finding next greater/smaller element
- See [[pattern-templates#Monotonic Stack]]

## Explanation:

1. Use a stack to keep track of elements for which we haven't found the next greater element yet.
2. Iterate through the array, and for each element, pop elements from the stack until you find a greater element.
3. If the stack is not empty, set the result for index at the top of the stack to current element.
4. Push the current element onto the stack.

## LeetCode Problems:

- Next Greater Element I (LeetCode [496](https://leetcode.com/problems/next-greater-element-i/))
- Daily Temperatures (LeetCode [739](https://leetcode.com/problems/daily-temperatures/))
- Largest Rectangle in Histogram (LeetCode [84](https://leetcode.com/problems/largest-rectangle-in-histogram/))

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
    # Question: Given an array, find the next greater element for each element. Return -1 if no greater element exists.
    # Key Insight: 
    #  - Use monotonic stack to track indices without next greater element yet
    #  - For each element, pop smaller elements from stack and set their next greater to current
    #  - Push current index to stack
    #  - Time: O(n), Space: O(n)
    res = [-1] * len(nums)
    stack = []

    for i in range(len(nums)):
        # pop all smaller elements from the stack 
        # ensuring the values at the stack indices are monotonic decreasing
        while stack and nums[i] > nums[stack[-1]]:
            prev_index = stack.pop() # previous index looking for next greater
            res[prev_index] = nums[i]
        stack.append(i)

    return res

print(next_greater([3, 1, 4]))  # Output: [4, 4, -1]
```

# Top 'K' Elements

Pattern for finding k largest/smallest elements using heaps or sorting

## Explanation:

1. Use a min-heap of size k to keep track of k largest elements (or max-heap for k smallest)
2. Iterate through array, adding elements to heap
3. If heap size exceeds k, remove the smallest/largest element
4. Root of heap will be the k-th largest/smallest element

## LeetCode Problems:

- Kth Largest Element in an Array (LeetCode [215](https://leetcode.com/problems/kth-largest-element-in-an-array/))
- Top K Frequent Elements (LeetCode [347](https://leetcode.com/problems/top-k-frequent-elements/))
- Find K Pairs with Smallest Sums (LeetCode [373](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/))

## Example:

```text
Question: Find 2nd largest element in [3, 2, 1, 5, 6, 4]
Answer: 5

Using min-heap of size k=2:
Process 3: heap = [3]
Process 2: heap = [2, 3]  
Process 1: heap = [2, 3] (1 < 2, don't add)
Process 5: heap = [3, 5] (remove 2, add 5)
Process 6: heap = [5, 6] (remove 3, add 6)
Process 4: heap = [5, 6] (4 < 5, don't add)

Result: heap[0] = 5 (2nd largest)
```

```python
def find_kth_largest(nums, k):
    # Question: Find the kth largest element in an unsorted array.
    # Key Insight: 
    #  - Use min-heap of size k to track k largest elements
    #  - Only keep elements larger than the smallest in heap
    #  - Root of heap is the kth largest element
    #  - Time: O(n log k), Space: O(k)
    # Use min heap of size k
    import heapq
    heap = []
    
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num) # push onto heap
        elif num > heap[0]:
            heapq.heapreplace(heap, num) # push onto heap then pop and return smallest
    
    return heap[0]

def top_k_frequent(nums, k):
    # Question: Given an integer array nums and an integer k, return the k most frequent elements.
    # Key Insight: 
    #  - Count frequency of each element
    #  - Use heap to get k elements with highest frequency
    #  - Time: O(n log k), Space: O(n)
    from collections import Counter
    count = Counter(nums)
    
    # Use heap to get k most frequent
    return heapq.nlargest(k, count.keys(), key=count.get)

nums = [3, 2, 1, 5, 6, 4]
print(find_kth_largest(nums, 2))  # Output: 5

nums = [1, 1, 1, 2, 2, 3]
print(top_k_frequent(nums, 2))   # Output: [1, 2]
```

# Overlapping Intervals

Pattern for merging or handling overlapping intervals in an array

## Explanation:

1. Sort intervals by start time
2. Initialize result with first interval
3. For each subsequent interval, check if it overlaps with last interval in result
4. If overlapping (start ≤ end of previous), merge them
5. If not overlapping, add new interval to result

## LeetCode Problems:

- Merge Intervals (LeetCode [56](https://leetcode.com/problems/merge-intervals/))
- Insert Interval (LeetCode [57](https://leetcode.com/problems/insert-interval/))
- Non-overlapping Intervals (LeetCode [435](https://leetcode.com/problems/non-overlapping-intervals/))

## Example:

```text
Question: Merge overlapping intervals [[1,3],[2,6],[8,10],[15,18]]
Answer: [[1,6],[8,10],[15,18]]

Sort by start: [[1,3],[2,6],[8,10],[15,18]] (already sorted)

Process [1,3]: result = [[1,3]]
Process [2,6]: 2 ≤ 3, so merge -> result = [[1,6]]
Process [8,10]: 8 > 6, no overlap -> result = [[1,6],[8,10]]  
Process [15,18]: 15 > 10, no overlap -> result = [[1,6],[8,10],[15,18]]
```

```python
def merge_intervals(intervals):
    # Question: Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals.
    # Key Insight: 
    #  - Sort intervals by start time
    #  - Iterate through sorted intervals, merging overlapping ones
    #  - Two intervals overlap if current.start <= previous.end
    #  - Time: O(n log n), Space: O(1)
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # Check if current overlaps with last interval
        if current[0] <= last[1]:
            # Merge intervals
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # No overlap, add current interval
            merged.append(current)
    
    return merged

intervals = [[1,3],[2,6],[8,10],[15,18]]
print(merge_intervals(intervals))  # Output: [[1,6],[8,10],[15,18]]
```


# Modified Binary Search

Pattern for adapting binary search to solve problems with rotated or modified sorted arrays

## Explanation:

1. Perform binary search with additional checks to determine which half is sorted
2. Check if target is within range of the sorted half
3. If target in sorted half, search that half; otherwise search the other half
4. Handle edge cases like duplicates or rotated arrays

## LeetCode Problems:

- Search in Rotated Sorted Array (LeetCode #33)
- Find Minimum in Rotated Sorted Array (LeetCode #153)
- Search a 2D Matrix II (LeetCode #240)

## Example:

```text
Question: Find target 0 in rotated sorted array [4,5,6,7,0,1,2]
Answer: 4 (index of target 0)

nums = [4,5,6,7,0,1,2], target = 0

Left half:
- left_idx = 0, right_idx = 6, mid_idx = 3, nums[mid_idx] = 7
- Check if left half [4,5,6,7] is sorted (nums[0] ≤ nums[3])
- Target 0 not in range [4,7], so search right half (move left_idx to mid_idx + 1)
Right half:
- left_idx = 4, right_idx = 6, mid_idx = 5, nums[mid_idx] = 1  
- Check if right half [1,2] is sorted (nums[4] ≤ nums[5])
- Target 0 not in range [1,2], so search left half (move right_idx to mid_idx - 1)
Left half:
- left_idx = 4, right_idx = 4, mid_idx = 4, nums[mid_idx] = 0 = target
Found at index 4!
```

```python
def search_rotated_array(nums, target):
    # Question: Given a rotated sorted array and a target value, search for the target. Return its index or -1 if not found.
    # Key Insight: 
    #  - Use modified binary search
    #  - At least one half of the array is always sorted
    #  - Determine which half is sorted, then check if target is in that range
    #  - Time: O(log n), Space: O(1)
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(search_rotated_array(nums, target))  # Output: 4
```

# Binary Tree Traversal

Pattern for visiting all nodes in a binary tree in specific orders
- See [[pattern-templates#Binary Search]]

## Explanation:

1. **PreOrder**: Visit root → left subtree → right subtree
2. **InOrder**: Visit left subtree → root → right subtree  
3. **PostOrder**: Visit left subtree → right subtree → root
4. Use recursion or stack for implementation

## LeetCode Problems:

- Binary Tree Paths (LeetCode #257) - PreOrder
- Kth Smallest Element in a BST (LeetCode #230) - InOrder
- Binary Tree Maximum Path Sum (LeetCode #124) - PostOrder

## Example:

```text
Question: Perform inorder traversal of tree:
    1
     \
      2
     /
    3

Flattened graph: [1, null, 2, 3]
Answer: [1, 3, 2]

InOrder traversal: Left → Root → Right
- Start at Root=1
  - Check Left -> None
  - Add Root=1 to result -> result = [1]
  - Check Right -> 2, go right
- Now at Root=2
  - Check Left -> 3, go left, Root=3 now
    - Check Left -> None
    - Add Root=3 to result -> result = [1, 3]
    - Check Right -> None
  - Add Root=2 to result -> result = [1, 3, 2]
  - Check Right -> None
```

```python
def inorder_traversal(root):
    # Question: Given the root of a binary tree, return the inorder traversal of its nodes' values.
    # Key Insight: 
    #  - Inorder: Left -> Root -> Right
    #  - Use recursion or stack for implementation
    #  - For BST, inorder gives sorted sequence
    #  - Time: O(n), Space: O(h) where h is height
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)    # Left
            result.append(node.val)  # Root
            inorder(node.right)   # Right
    
    inorder(root)
    return result

def preorder_traversal(root):
    # Question: Given the root of a binary tree, return the preorder traversal of its nodes' values.
    # Key Insight: 
    #  - Preorder: Root -> Left -> Right
    #  - Process root first, then recursively traverse subtrees
    #  - Useful for creating copy of tree or prefix expressions
    #  - Time: O(n), Space: O(h) where h is height
    result = []
    
    def preorder(node):
        if node:
            result.append(node.val)  # Root
            preorder(node.left)      # Left
            preorder(node.right)     # Right
    
    preorder(root)
    return result

# Example tree: 1 -> null, 2 -> 3, null
root = TreeNode(1)
root.right = TreeNode(2)
root.right.left = TreeNode(3)

print(inorder_traversal(root))   # Output: [1, 3, 2]
print(preorder_traversal(root))  # Output: [1, 2, 3]
```

# Depth-First Search (DFS)

Pattern for exploring all paths or branches in graphs/trees by going deep before backtracking

## Explanation:

1. Use recursion or stack to traverse as far as possible down each branch
2. Mark visited nodes to avoid cycles in graphs
3. Backtrack when reaching dead end or target found
4. Explore all possible paths from current node before moving to next

## LeetCode Problems:

- Clone Graph (LeetCode [133](https://leetcode.com/problems/clone-graph/))
- Path Sum II (LeetCode [113](https://leetcode.com/problems/path-sum-ii/))
- Course Schedule II (LeetCode [210](https://leetcode.com/problems/course-schedule-ii/))

## Example:

```text
Question: Find all root-to-leaf paths in binary tree:
    1
   / \
  2   3
   \
    5

Flattened graph: [1, 2, 3, null, 5]
Answer: ["1->2->5", "1->3"]

DFS traversal:
- Start at 1, path = "1"
- Go left to 2, path = "1->2"  
- Go right to 5 (leaf), path = "1->2->5" → add to result
- Backtrack to 1
- Go right to 3 (leaf), path = "1->3" → add to result
```

```python
def binary_tree_paths(root):
    # Question: Given the root of a binary tree, return all root-to-leaf paths in any order.
    # Key Insight: 
    #  - Use DFS to explore all paths from root to leaves
    #  - Build path string as we traverse down
    #  - When reaching leaf, add complete path to result
    #  - Time: O(n), Space: O(n) for recursion and paths
    if not root:
        return []
    
    paths = []
    
    def dfs(node, path):
        if not node.left and not node.right:  # Leaf node
            paths.append(path)
            return
        
        if node.left:
            dfs(node.left, path + "->" + str(node.left.val))
        
        if node.right:
            dfs(node.right, path + "->" + str(node.right.val))
    
    dfs(root, str(root.val))
    return paths

# Example tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.right = TreeNode(5)

print(binary_tree_paths(root))  # Output: ['1->2->5', '1->3']
```

# Breadth-First Search (BFS)

Pattern for exploring nodes level by level using a queue

## Explanation:

1. Use queue to keep track of nodes at current level
2. Process all nodes at current level before moving to next level
3. Add children of current nodes to queue for next level
4. Continue until queue is empty or target found

## LeetCode Problems:

- Binary Tree Level Order Traversal (LeetCode [102](https://leetcode.com/problems/binary-tree-level-order-traversal/))
- Rotting Oranges (LeetCode [994](https://leetcode.com/problems/rotting-oranges/))
- Word Ladder (LeetCode [127](https://leetcode.com/problems/word-ladder/))

## Example:

```text
Question: Level order traversal of tree:
    3
   / \
  9   20
     /  \
    15   7

Flattened graph: [3, 9, 20, null, null, 15, 7]
Answer: [[3], [9, 20], [15, 7]]

BFS process:
Level 0: queue = [3] → process 3 → level = [3]
Level 1: queue = [9, 20] → process both → level = [9, 20]  
Level 2: queue = [15, 7] → process both → level = [15, 7]
```

```python
def level_order_traversal(root):
    # Question: Given the root of a binary tree, return the level order traversal of its nodes' values (i.e., from left to right, level by level).
    # Key Insight: 
    #  - Use BFS with queue to process nodes level by level
    #  - Track level size to separate levels in result
    #  - Process all nodes at current level before moving to next
    #  - Time: O(n), Space: O(w) where w is maximum width
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result

# Example tree
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

print(level_order_traversal(root))  # Output: [[3], [9, 20], [15, 7]]
```

# Matrix Traversal

Pattern for traversing 2D grids using DFS, BFS, or other techniques

## Explanation:

1. Use DFS or BFS to traverse matrix starting from given cell
2. Check bounds and visited cells to avoid going out of matrix
3. Explore adjacent cells (up, down, left, right, and sometimes diagonals)
4. Mark visited cells or use them to solve specific problems

## LeetCode Problems:

- Flood Fill (LeetCode [733](https://leetcode.com/problems/flood-fill/))
- Number of Islands (LeetCode [200](https://leetcode.com/problems/number-of-islands/))
- Surrounded Regions (LeetCode [130](https://leetcode.com/problems/surrounded-regions/))

## Example:

```text
(n, m) = (row, col)
Question: Flood fill starting at (1,1) with new color=2 in:
[[1,1,1],
 [1,1,0], 
 [1,0,1]]

Answer:
[[2,2,2],
 [2,2,0],
 [2,0,1]]

DFS from (1,1):
- Change (1,1) from 1 to 2
- Check neighbors: (0,1), (2,1), (1,0), (1,2)
- (0,1) has value 1 → change to 2, continue DFS
- Continue until all connected 1s become 2s
```

```python
def flood_fill(image, sr, sc, new_color): # start_row, start_col, new_color
    # Question: Given a 2D image, starting position, and new color, perform flood fill by changing all connected pixels of the same color to the new color.
    # Key Insight: 
    #  - Use DFS to explore all connected pixels with same original color
    #  - Change color as we visit each pixel
    #  - Use bounds checking and color checking to stop recursion
    #  - Time: O(n*m), Space: O(n*m) for recursion
    if not image or image[sr][sc] == new_color:
        return image
    
    original_color = image[sr][sc]
    rows, cols = len(image), len(image[0])
    
    def dfs(r, c):
        # if out of bounds or not the original color (already visited), return
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            image[r][c] != original_color):
            return
        
        image[r][c] = new_color
        
        # Explore 4 directions
        dfs(r + 1, c)  # down
        dfs(r - 1, c)  # up
        dfs(r, c + 1)  # right
        dfs(r, c - 1)  # left
    
    dfs(sr, sc)
    return image

def num_islands(grid):
    # Question: Given a 2D grid map of '1's (land) and '0's (water), count the number of islands.
    # Key Insight: 
    #  - Use DFS to explore each connected component of land
    #  - Mark visited land as water to avoid recounting
    #  - Each DFS call from unvisited land represents one island
    #  - Time: O(n*m), Space: O(n*m) for recursion
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] != '1'):
            return
        
        grid[r][c] = '0'  # Mark as visited
        
        # Explore 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)
    
    return islands

# Example
image = [[1,1,1],[1,1,0],[1,0,1]]
result = flood_fill(image, 1, 1, 2)
print(result)  # Output: [[2,2,2],[2,2,0],[2,0,1]]

grid = [["1","1","0"],["1","0","0"],["0","0","1"]]
print(num_islands(grid))  # Output: 2
```

# Backtracking

Pattern for exploring all possible solutions by trying partial solutions and abandoning them if they can't lead to valid solution

## Explanation:

1. Build solution incrementally, one piece at a time
2. At each step, try all possible choices
3. If choice leads to invalid solution, backtrack (undo choice)
4. If choice leads to valid solution, add to result
5. Continue until all possibilities explored

## LeetCode Problems:

- Permutations (LeetCode [46](https://leetcode.com/problems/permutations/))
- Subsets (LeetCode [78](https://leetcode.com/problems/subsets/))
- N-Queens (LeetCode [51](https://leetcode.com/problems/n-queens/))

## Example:

```text
Question: Generate all permutations of [1, 2, 3]
Answer: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

Backtracking process:
Start with []
- Try 1: [1]
  - Try 2: [1,2]
    - Try 3: [1,2,3] → valid, add to result
    - Backtrack to [1,2]
  - Backtrack to [1]
  - Try 3: [1,3]
    - Try 2: [1,3,2] → valid, add to result
- Continue with 2, then 3...
```

```text
Start with []
- Add 1 -> [1]
- backtrack([1])
  - Add 2 -> [1, 2]
  - backtrack([1, 2])
    - Add 3 -> [1, 2, 3]
    - backtrack([1, 2, 3]) -> add to result + return
    - pop 3 -> [1, 2]
  - pop 2 -> [1]
  - Add 3 -> [1, 3]
  - backtrack([1, 3])
    - Add 2 -> [1, 3, 2]
    - backtrack([1, 3, 2]) -> add to result + return
    - pop 2 -> [1, 3]
  - pop 3 -> [1]
- pop 1 -> []
- Add 2 -> [2]
- ... and so on
```

```python
def permutations(nums):
    # Question: Given an array nums of distinct integers, return all possible permutations in any order.
    # Key Insight: 
    #  - Use backtracking to build permutations incrementally
    #  - At each step, try all remaining unused numbers
    #  - Backtrack when permutation is complete or no more choices
    #  - Time: O(n! * n), Space: O(n) for recursion
    result = []
    
    def backtrack(current_perm):
        # Base case: permutation is complete
        if len(current_perm) == len(nums):
            result.append(current_perm[:])  # Make a copy
            return

        for num in nums:
            if num not in current_perm:     # Try only remaining numbers
                current_perm.append(num)    # Choose
                backtrack(current_perm)     # Explore
                current_perm.pop()          # Backtrack
    
    backtrack([])
    return result

def subsets(nums):
    # Question: Given an integer array nums of unique elements, return all possible subsets (the power set).
    # Key Insight: 
    #  - Use backtracking to build subsets incrementally
    #  - At each position, choose to include or exclude the element
    #  - Add current subset to result at each step
    #  - Time: O(2^n * n), Space: O(n) for recursion
    result = []
    
    def backtrack(start, current_subset):
        # Add current subset to result
        result.append(current_subset[:])
        
        # Try adding each remaining element
        for i in range(start, len(nums)):
            current_subset.append(nums[i])     # Choose
            backtrack(i + 1, current_subset)   # Explore
            current_subset.pop()               # Backtrack
    
    backtrack(0, [])
    return result

def solve_n_queens(n):
    # Question: The n-queens puzzle is to place n queens on an n×n chessboard such that no two queens attack each other.
    # Key Insight: 
    #  - Use backtracking to place queens row by row
    #  - Check if current position is safe (no conflicts with previous queens)
    #  - Backtrack if no safe position in current row
    #  - Time: O(n!), Space: O(n^2) for board
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonals
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'       # Choose
                backtrack(row + 1)          # Explore  
                board[row][col] = '.'       # Backtrack
    
    backtrack(0)
    return result

print(permutations([1, 2, 3]))  # Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
print(subsets([1, 2, 3]))       # Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

# Dynamic Programming Patterns

Pattern for solving problems with overlapping subproblems and optimal substructure

## Explanation:

1. Break problem into smaller overlapping subproblems
2. Store results of subproblems to avoid recomputation
   1. top-down memoization: recursive caching
   2. bottom-up tabulation: iterative with table
3. Build solution bottom-up or top-down
4. Common patterns: Fibonacci, Knapsack, Longest Common Subsequence, Longest Increasing Subsequence, Subset Sum

## LeetCode Problems:

- Climbing Stairs (LeetCode [70](https://leetcode.com/problems/climbing-stairs/)) - Fibonacci
- House Robber (LeetCode [198](https://leetcode.com/problems/house-robber/)) - Linear DP
- Coin Change (LeetCode [322](https://leetcode.com/problems/coin-change/)) - Unbounded Knapsack
- Longest Common Subsequence (LeetCode [1143](https://leetcode.com/problems/longest-common-subsequence/)) - 2D DP
- Longest Increasing Subsequence (LeetCode [300](https://leetcode.com/problems/longest-increasing-subsequence/)) - LIS
- Partition Equal Subset Sum (LeetCode [416](https://leetcode.com/problems/partition-equal-subset-sum/)) - 0/1 Knapsack

## Example:

```text
Question: Calculate 5th Fibonacci number
Answer: 5 (sequence: 0, 1, 1, 2, 3, 5)

Bottom-up approach:
dp[0] = 0
dp[1] = 1  
dp[2] = dp[1] + dp[0] = 1 + 0 = 1
dp[3] = dp[2] + dp[1] = 1 + 1 = 2
dp[4] = dp[3] + dp[2] = 2 + 1 = 3
dp[5] = dp[4] + dp[3] = 3 + 2 = 5
```

```python
def fibonacci(n):
    # Question: Calculate the nth Fibonacci number where F(0) = 0, F(1) = 1, and F(n) = F(n-1) + F(n-2) for n > 1.
    # Key Insight: 
    #  - Each number is sum of previous two numbers
    #  - Use bottom-up DP to avoid recomputation
    #  - Can optimize space to O(1) by keeping only last two values
    #  - Time: O(n), Space: O(n) or O(1) with optimization
    if n <= 1:
        return n
    
    # Bottom-up approach
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

def climbing_stairs(n):
    # Question: You are climbing a staircase. It takes n steps to reach the top. You can either climb 1 or 2 steps at a time. How many distinct ways can you climb to the top?
    # Key Insight: 
    #  - Take 1 step from step `n-1` -> f(n-1)
    #  - Take 2 step from step `n-2` -> f(n-2)
    #  - f(n) = f(n-1) + f(n-2)
    #  - Same as Fibonacci sequence
    # Same as Fibonacci - ways to reach step n
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

def coin_change(coins, amount):
    # Question: Given an integer array representing coins of different denominations and an integer amount, return the fewest number of coins needed to make up that amount.
    # Key Insight: 
    #  - Use DP where dp[i] = minimum coins to make amount i
    #  - For each coin, try using it: dp[i] = min(dp[i], dp[i-coin] + 1)
    #  - Build solution bottom-up from 0 to amount
    #  - Time: O(amount * coins), Space: O(amount)
    # Minimum coins to make amount
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def longest_common_subsequence(text1, text2):
    # Question: Given two strings text1 and text2, return the length of their longest common subsequence.
    # Key Insight: 
    #  - Use 2D DP where dp[i][j] = LCS length of text1[0:i] and text2[0:j]
    #  - If characters match: dp[i][j] = dp[i-1][j-1] + 1
    #  - If not match: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    #  - Time: O(m*n), Space: O(m*n)
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def can_partition(nums):
    # Question: Given a non-empty array nums containing only positive integers, determine if the array can be partitioned into two subsets with equal sum.
    # Key Insight: 
    #  - This is 0/1 knapsack problem: can we make sum = total_sum/2?
    #  - Use DP where dp[i] = true if sum i is achievable
    #  - For each number, update dp array backwards to avoid using same number twice
    #  - Time: O(n * sum), Space: O(sum)
    # 0/1 Knapsack - can partition into equal sum subsets
    total_sum = sum(nums)
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]

print(fibonacci(5))                    # Output: 5
print(climbing_stairs(5))              # Output: 8
print(coin_change([1, 3, 4], 6))      # Output: 2
print(longest_common_subsequence("abcde", "ace"))  # Output: 3
print(can_partition([1, 5, 11, 5]))   # Output: True
```
