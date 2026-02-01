import numpy as np
#import matplotlib.pyplot as plt

phone=[[6.06,6.4,7.15,6.3,6.42],
       [5.14,4.01,4.7,4.77,5.21],
       [6.02,5.24,6.72,5.82,5.39],
       [4.2,3.96,4.18,3.61,3.8],
       [50.2,52.2,51.7,52.6,49.5]]

data=[[6.1,6.2,6.8,6.0,6.5],
      [4.2,3.7,4.5,4.2,4.6],
      [5.5,5.1,6.25,6.0,5.1],
      [3.5,3.1,4.0,3.2,3.4],
      [50.4,51.6,52.1,53.3,50.3]]

row_names = ['视频', '游戏', '导航', '低温视频', '空闲']

print("="*80)
print("模型拟合效果分析 (Model vs Experimental Data)")
print("="*80)

# 转换为numpy数组
phone_array = np.array(phone)
data_array = np.array(data)

# 逐行分析
all_metrics = []

for i in range(len(phone)):
    print(f"\n【单元 {i+1}: {row_names[i]}】")
    print(f"  实验数据: {phone[i]}")
    print(f"  模型数据: {data[i]}")
    
    # 计算逐个差值
    differences = [phone[i][j] - data[i][j] for j in range(5)]
    abs_differences = [abs(d) for d in differences]
    
    # 计算相对误差（百分比）
    relative_errors = [(abs(phone[i][j] - data[i][j]) / data[i][j]) * 100 for j in range(5)]
    
    # 计算统计指标
    mae = np.mean(abs_differences)  # 平均绝对误差
    rmse = np.sqrt(np.mean([(phone[i][j] - data[i][j])**2 for j in range(5)]))  # 均方根误差
    mape = np.mean(relative_errors)  # 平均绝对百分比误差
    
    # R² (决定系数)
    ss_res = sum((phone[i][j] - data[i][j])**2 for j in range(5))
    ss_tot = sum((phone[i][j] - np.mean(phone[i]))**2 for j in range(5))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 最大误差
    max_error_idx = np.argmax(abs_differences)
    max_error = abs_differences[max_error_idx]
    max_rel_error = relative_errors[max_error_idx]
    
    print(f"\n  逐点差值: {[f'{d:+.4f}' for d in differences]}")
    print(f"  相对误差: {[f'{re:.2f}%' for re in relative_errors]}")
    print(f"\n  统计指标:")
    print(f"  ├─ MAE (平均绝对误差): {mae:.4f}")
    print(f"  ├─ RMSE (均方根误差): {rmse:.4f}")
    print(f"  ├─ MAPE (平均相对误差): {mape:.2f}%")
    print(f"  ├─ R² (决定系数): {r_squared:.4f}")
    print(f"  └─ 最大误差: {max_error:.4f} ({max_rel_error:.2f}%) [第{max_error_idx+1}个点]")
    
    # 拟合质量评价
    if mape < 5 and r_squared > 0.95:
        quality = "优秀"
        score = "A"
    elif mape < 10 and r_squared > 0.85:
        quality = "良好"
        score = "B"
    elif mape < 15 and r_squared > 0.70:
        quality = "一般"
        score = "C"
    else:
        quality = "较差"
        score = "D"
    
    print(f"\n  拟合质量: {quality} (评级: {score})")
    
    all_metrics.append({
        'unit': row_names[i],
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r_squared,
        'score': score
    })

# 综合评估
print("\n" + "="*80)
print("综合评估报告")
print("="*80)

avg_mape = np.mean([m['mape'] for m in all_metrics])
avg_r2 = np.mean([m['r2'] for m in all_metrics])
avg_mae = np.mean([m['mae'] for m in all_metrics])
avg_rmse = np.mean([m['rmse'] for m in all_metrics])

print(f"\n整体平均指标:")
print(f"  平均MAPE: {avg_mape:.2f}%")
print(f"  平均R²: {avg_r2:.4f}")
print(f"  平均MAE: {avg_mae:.4f}")
print(f"  平均RMSE: {avg_rmse:.4f}")

# 找出最好和最差的单元
best_unit = min(all_metrics, key=lambda x: x['mape'])
worst_unit = max(all_metrics, key=lambda x: x['mape'])

print(f"\n拟合最好的单元: {best_unit['unit']} (MAPE: {best_unit['mape']:.2f}%, R²: {best_unit['r2']:.4f})")
print(f"拟合最差的单元: {worst_unit['unit']} (MAPE: {worst_unit['mape']:.2f}%, R²: {worst_unit['r2']:.4f})")

# 总体评级
a_count = sum(1 for m in all_metrics if m['score'] == 'A')
b_count = sum(1 for m in all_metrics if m['score'] == 'B')
c_count = sum(1 for m in all_metrics if m['score'] == 'C')
d_count = sum(1 for m in all_metrics if m['score'] == 'D')

print(f"\n评级分布: A={a_count}, B={b_count}, C={c_count}, D={d_count}")

if avg_mape < 5 and avg_r2 > 0.95:
    print("\n总体拟合质量: 优秀")
elif avg_mape < 10 and avg_r2 > 0.85:
    print("\n总体拟合质量: 良好")
elif avg_mape < 15 and avg_r2 > 0.70:
    print("\n总体拟合质量: 一般")
else:
    print("\n总体拟合质量: 需要改进")

print("="*80)
