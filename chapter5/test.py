my_dict = dict([('layer1', 10), ('layer2', 20)])

# 2. 开一扇玻璃窗（获取视图）
my_view = my_dict.values()
print(my_view)
# 输出: odict_values([10, 20])

# 3. 重点来了！我们在原字典里加一个新层
my_dict['layer3'] = 30

# 4. 再次看一眼这扇玻璃窗（注意：我们根本没有重新获取 my_view）
print(my_view)
# 输出: odict_values([10, 20, 30]) —— 它居然自己变了！