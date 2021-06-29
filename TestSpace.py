
l1 = [1, 2, 3, 4]
l2 = [2, 3, 4, 5]
l3 = [3, 4, 5, 6]
combined = [l1, l2, l3]
combined2 = [l3, l2, l1]

# print([i-k for i,k in zip(l1, l2)])

for i, (c1, c2) in enumerate(zip(combined, combined2)):
    print(i)
    print(c1)
    print(c2)

