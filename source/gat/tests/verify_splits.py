import pandas as pd
import numpy as np

def verify_splits():
    # Load splits
    train = pd.read_csv('data/splits/train.csv')
    val = pd.read_csv('data/splits/val.csv')
    test = pd.read_csv('data/splits/test.csv')
    
    print("="*60)
    print("SPLIT VERIFICATION")
    print("="*60)
    
    # 1. Check sizes
    print(f"\n1. Split Sizes:")
    total = len(train) + len(val) + len(test)
    print(f"   Train: {len(train):,} ({len(train)/total*100:.1f}%)")
    print(f"   Val:   {len(val):,} ({len(val)/total*100:.1f}%)")
    print(f"   Test:  {len(test):,} ({len(test)/total*100:.1f}%)")
    print(f"   Total: {total:,}")
    
    # 2. Check for duplicates using content hashing
    print(f"\n2. Data Leakage Check:")
    
    def row_to_tuple(df):
        """Convert rows to tuples for comparison."""
        return [tuple(row) for row in df.values]
    
    train_rows = set(row_to_tuple(train))
    val_rows = set(row_to_tuple(val))
    test_rows = set(row_to_tuple(test))
    
    overlap_train_val = len(train_rows & val_rows)
    overlap_train_test = len(train_rows & test_rows)
    overlap_val_test = len(val_rows & test_rows)
    
    print(f"   Train/Val overlap:  {overlap_train_val} rows")
    print(f"   Train/Test overlap: {overlap_train_test} rows")
    print(f"   Val/Test overlap:   {overlap_val_test} rows")
    
    if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
        print("   ✅ No data leakage detected!")
    else:
        print("   ❌ WARNING: Data leakage detected!")
        return False
    
    # 3. Platform balance
    print(f"\n3. Platform Distribution:")
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        counts = df['platform'].value_counts().sort_index()
        rubikpi = counts.get(0, 0)
        tx2 = counts.get(1, 0)
        total_split = len(df)
        print(f"   {name:5s}: RubikPi={rubikpi} ({rubikpi/total_split*100:.1f}%), "
              f"TX2={tx2} ({tx2/total_split*100:.1f}%)")
    
    # 4. Benchmark diversity
    print(f"\n4. Benchmark Coverage:")
    train_benchmarks = set(train['benchmark'].unique())
    val_benchmarks = set(val['benchmark'].unique())
    test_benchmarks = set(test['benchmark'].unique())
    all_benchmarks = train_benchmarks | val_benchmarks | test_benchmarks
    
    print(f"   Total unique: {len(all_benchmarks)}")
    print(f"   Train: {len(train_benchmarks)} ({len(train_benchmarks)/len(all_benchmarks)*100:.0f}%)")
    print(f"   Val:   {len(val_benchmarks)} ({len(val_benchmarks)/len(all_benchmarks)*100:.0f}%)")
    print(f"   Test:  {len(test_benchmarks)} ({len(test_benchmarks)/len(all_benchmarks)*100:.0f}%)")
    
    # 5. Target distribution
    print(f"\n5. Target Distribution (normalized):")
    print(f"   {'':5s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        target = df['time_elapsed']
        print(f"   {name:5s}  {target.mean():8.4f}  {target.std():8.4f}  "
              f"{target.min():8.4f}  {target.max():8.4f}")
    
    print("\n" + "="*60)
    print("✅ All checks passed!")
    print("="*60)
    return True

if __name__ == '__main__':
    verify_splits()