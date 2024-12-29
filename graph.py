from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='CoT 过程中的熵优化流程')

# 设置全局字体属性
dot.attr(
    encoding='UTF-8',  # 确保使用 UTF-8 编码
    fontname='Microsoft YaHei',  # 设置全局字体为微软雅黑
)

# 设置图的整体布局，使用树形布局（从上到下）
dot.attr(dpi='300', rankdir='TB', nodesep='0.8', ranksep='1.2')  # 设置从上到下的布局，并增加节点间距

# 调整节点样式，使其更加美观
dot.attr('node', shape='ellipse', style='rounded,filled', fontname='Microsoft YaHei', 
         fontsize='14', width='1.5', height='0.8', margin='0.1', 
         fixedsize='true', color='black', fontcolor='black')

# 使用柔和的渐变颜色
dot.node('A', '输入问题', color='lightblue', fontcolor='black', fillcolor='lightblue')
dot.node('B', '初始思考（高熵）', color='lightgreen', fontcolor='black', fillcolor='lightgreen')
dot.node('C', '逐步推理并减少熵', color='yellow', fontcolor='black', fillcolor='yellow')
dot.node('D', '熵计算与更新（再推理）', color='orange', fontcolor='black', fillcolor='orange')
dot.node('E', '多步骤推理与熵整合', color='pink', fontcolor='black', fillcolor='pink')
dot.node('F', '生成最终结果', color='lightgrey', fontcolor='black', fillcolor='lightgrey')
dot.node('G', '结果反馈与优化', color='cyan', fontcolor='black', fillcolor='cyan')
dot.node('H', '结束或继续优化', color='lightcoral', fontcolor='black', fillcolor='lightcoral')

# 设置边样式，使其看起来更加精致
dot.attr('edge', fontname='Microsoft YaHei', fontsize='12', color='gray', fontcolor='black', style='solid')

# 添加边（连接节点）
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')

# 渲染图形，保存为 PNG 文件
output_file = 'cot_entropy_optimization_flowchart_tree'
dot.render(output_file, format='png', view=True)  # 自动打开渲染好的图

# 输出成功提示
print(f"流程图已保存为 {output_file}.png")
