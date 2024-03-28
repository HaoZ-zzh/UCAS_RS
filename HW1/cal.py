Hit = [1, 4, 6, 9, 12, 18, 20]
# Hit = [1, 4, 6, 9, 12, 18, 20, 21, 22, 23]
# Hit = [1, 4, 6, 9, 12, 18, 20, 9998, 9999, 10000]
sum = 0
len = len(Hit)
Need = 10
for i in range(1, len + 1):
    sum += 1.0 * i / Hit[i-1]
MAP = sum / Need
print(MAP)