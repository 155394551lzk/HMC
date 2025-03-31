import pandas as pd

# 假设我们有一个DataFrame：df = pd.DataFrame({'A': [1, 2, 3],  'B': [4, 5, 6] }) ，
# 目的df = pd.DataFrame({'A': [1, 4, 9],  'B': [4, 5, 6] , 'X': [1, 4, 9],  'Y'=[5,9,15]}) 。其中X列是A平方后的值， Y列是A + B的值

def precess():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    # 计算新列 X 和 Y 的值
    df['X'] = df['A'] ** 2  # 计算 A 列的平方值，存储到 X 列
    df['Y'] = df['A'] + df['B']  # 计算 A 列和 B 列的和，存储到 Y 列

    # 输出目标 DataFrame
    print(df)

    A = df.apply("A")
    B = df.apply("B")
    x = []
    y = []

    for i, j in zip(A, B):
        x.append(i ** 2)
        y.append(i + j)

    df = pd.DataFrame([A, B, x, y])

    print(df)

# 给定两个字符串 a b, 要求判断a字符串是否可由b字符串中的字符构成，其中b中的字符要求只能使用一次
# def judgment(a,b)  return True or False
# Example:
# judgment('xx','xy')   false
# judgment('xx','xxy')   true
# judgment('abcdefghi','hgfedcbai')   true
# judgment('mississippi' ,'misisppi')   false

def judgment(a,b):
    count = {}
    for i in b:
        if i not in count:
            count[i] = 0
        count[i] += 1
    for i in a:
        if i not in count:
            return False
        if count[i] > 0:
            count[i] -= 1
        else:
            return False
    return True

judgment('xx','xy')
judgment('xx','xxy')
judgment('abcdefghi','hgfedcbai')
judgment('mississippi' ,'misisppi')