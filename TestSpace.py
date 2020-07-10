
l1 = [1, 2, 3, 4]
l2 = [2, 3, 4, 5]
l3 = [3, 4, 5, 6]
combined = [l1, l2, l3]

print([i-k for i,k in zip(l1, l2)])
