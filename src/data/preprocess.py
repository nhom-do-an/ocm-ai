"""
Data Preprocessing Pipeline - Step 1
Validate and clean raw data from CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Paths
RAW_DATA_DIR = Path("d:/ai-feature/data/processed")
CLEANED_DATA_DIR = Path("d:/ai-feature/data/processed/cleaned")
REPORT_DIR = Path("d:/ai-feature/reports/data")

# Create directories
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    """Load all raw CSV files"""
    print("\n📂 Loading raw data...")
    
    data = {}
    files = [
        'stores', 'customers', 'addresses',
        'products', 'variants', 'orders', 'line_items', 'carts',
        'product_types', 'vendors', 'collections', 'tags',
        'product_collections', 'product_tags',
        'inventory_items', 'inventory_levels', 'locations',
        'payment_methods', 'payment_method_lines',
        'shipping_lines', 'sources'
    ]
    
    for file in files:
        try:
            df = pd.read_csv(RAW_DATA_DIR / f"{file}.csv")
            data[file] = df
            print(f"  ✅ {file}: {len(df):,} rows")
        except FileNotFoundError:
            print(f"  ⚠️  {file}.csv not found")
            data[file] = pd.DataFrame()
    
    return data


def validate_data(data):
    """Validate data quality"""
    print("\n🔍 Validating data quality...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'tables': {}
    }
    
    for table_name, df in data.items():
        if df.empty:
            continue
            
        table_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'unique_counts': {},
            'issues': []
        }
        
        # Missing values
        missing = df.isnull().sum()
        table_report['missing_values'] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        
        # Data types
        table_report['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Unique counts for key columns
        if 'id' in df.columns:
            table_report['unique_counts']['id'] = df['id'].nunique()
            if df['id'].nunique() != len(df):
                table_report['issues'].append('Duplicate IDs found')
        
        # Soft deletes
        if 'deleted_at' in df.columns:
            deleted_count = df['deleted_at'].notna().sum()
            table_report['deleted_records'] = int(deleted_count)
            if deleted_count > 0:
                table_report['issues'].append(f'{deleted_count} soft-deleted records')
        
        report['tables'][table_name] = table_report
        
        # Print summary
        print(f"\n  📊 {table_name}:")
        print(f"     Rows: {len(df):,}")
        if table_report['missing_values']:
            print(f"     Missing values: {sum(table_report['missing_values'].values()):,}")
        if table_report['issues']:
            print(f"     ⚠️  Issues: {', '.join(table_report['issues'])}")
    
    # Save report
    report_path = REPORT_DIR / "data_validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Validation report saved: {report_path}")
    return report


def clean_data(data):
    """Clean and standardize data"""
    print("\n🧹 Cleaning data...")
    
    cleaned = {}
    
    for table_name, df in data.items():
        if df.empty:
            cleaned[table_name] = df
            continue
        
        df_clean = df.copy()
        
        # Remove soft-deleted records
        if 'deleted_at' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['deleted_at'].isna()]
            removed = before - len(df_clean)
            if removed > 0:
                print(f"  ✂️  {table_name}: Removed {removed:,} deleted records")
        
        # Convert datetime columns
        datetime_cols = [col for col in df_clean.columns if 
                        any(x in col.lower() for x in ['_at', '_on', 'date'])]
        for col in datetime_cols:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                except:
                    pass
        
        # Standardize string columns
        string_cols = df_clean.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col not in datetime_cols:
                df_clean[col] = df_clean[col].fillna('').astype(str).str.strip()
        
        # Fill numeric NaN with appropriate values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                # For price/amount columns, fill with 0
                if any(x in col.lower() for x in ['price', 'amount', 'total', 'cost']):
                    df_clean[col] = df_clean[col].fillna(0)
                # For count columns, fill with 0
                elif any(x in col.lower() for x in ['quantity', 'count', 'num']):
                    df_clean[col] = df_clean[col].fillna(0)
        
        cleaned[table_name] = df_clean
        
        if len(df_clean) < len(df):
            print(f"  ✅ {table_name}: {len(df):,} → {len(df_clean):,} rows")
    
    return cleaned


def save_cleaned_data(data):
    """Save cleaned data to CSV"""
    print("\n💾 Saving cleaned data...")
    
    for table_name, df in data.items():
        if df.empty:
            continue
        
        output_path = CLEANED_DATA_DIR / f"{table_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✅ Saved: {output_path}")
    
    print(f"\n✅ All cleaned data saved to: {CLEANED_DATA_DIR}")


def generate_summary_stats(data):
    """Generate summary statistics"""
    print("\n📊 Generating summary statistics...")
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'tables': {}
    }
    
    # Key tables statistics
    if not data['stores'].empty:
        stats['stores'] = {
            'total': len(data['stores']),
            'by_category': data['stores'].groupby('_category').size().to_dict() if '_category' in data['stores'].columns else {}
        }
    
    if not data['customers'].empty:
        stats['customers'] = {
            'total': len(data['customers']),
            'by_store': len(data['customers'].groupby('store_id')),
            'avg_per_store': len(data['customers']) / len(data['stores']) if not data['stores'].empty else 0,
            'verified_email_pct': (data['customers']['verified_email'].sum() / len(data['customers']) * 100) if 'verified_email' in data['customers'].columns else 0
        }
    
    if not data['products'].empty:
        stats['products'] = {
            'total': len(data['products']),
            'active': len(data['products'][data['products']['status'] == 'active']) if 'status' in data['products'].columns else 0,
            'avg_per_store': len(data['products']) / len(data['stores']) if not data['stores'].empty else 0
        }
    
    if not data['orders'].empty:
        stats['orders'] = {
            'total': len(data['orders']),
            'by_status': data['orders']['status'].value_counts().to_dict() if 'status' in data['orders'].columns else {},
            'by_financial_status': data['orders']['financial_status'].value_counts().to_dict() if 'financial_status' in data['orders'].columns else {},
            'avg_per_customer': len(data['orders']) / len(data['customers']) if not data['customers'].empty else 0
        }
    
    if not data['line_items'].empty:
        stats['line_items'] = {
            'total': len(data['line_items']),
            'by_reference_type': data['line_items']['reference_type'].value_counts().to_dict() if 'reference_type' in data['line_items'].columns else {}
        }
    
    # Save stats
    stats_path = REPORT_DIR / "data_summary_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Summary stats saved: {stats_path}")
    
    # Print key metrics
    print("\n" + "="*60)
    print("📈 KEY METRICS")
    print("="*60)
    if 'stores' in stats:
        print(f"Stores: {stats['stores']['total']:,}")
    if 'customers' in stats:
        print(f"Customers: {stats['customers']['total']:,}")
    if 'products' in stats:
        print(f"Products: {stats['products']['total']:,}")
    if 'orders' in stats:
        print(f"Orders: {stats['orders']['total']:,}")
    if 'line_items' in stats:
        print(f"Line Items: {stats['line_items']['total']:,}")
    print("="*60)
    
    return stats


def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*60)
    print("🚀 DATA PREPROCESSING PIPELINE - STEP 1")
    print("="*60)
    
    # Load data
    data = load_raw_data()
    
    # Validate
    validation_report = validate_data(data)
    
    # Clean
    cleaned_data = clean_data(data)
    
    # Save
    save_cleaned_data(cleaned_data)
    
    # Summary stats
    stats = generate_summary_stats(cleaned_data)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nCleaned data: {CLEANED_DATA_DIR}")
    print(f"Reports: {REPORT_DIR}")
    print("\n💡 Next step: python src/data/create_interactions.py")


if __name__ == "__main__":
    main()
