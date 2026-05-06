import sys
for i, line in enumerate(open('app_v1.py', encoding='utf-8')): 
    if 'Detection Details' in line or 'Quick Metrics' in line: print(f'{i+1}: {line.strip()}')
