"""Validate generated data against database schema"""
import pandas as pd
import re

# Define expected schema from ocm-tables.sql
DB_SCHEMA = {
    'stores': ['id', 'name', 'trade_name', 'email', 'phone', 'domain', 'alias', 'address',
               'currency', 'timezone', 'money_format', 'money_with_currency_format', 
               'weight_unit', 'created_at', 'updated_at', 'deleted_at'],
    'customers': ['id', 'email', 'phone', 'first_name', 'last_name', 'password', 'gender', 
                  'dob', 'note', 'status', 'verified_email', 'created_at', 'updated_at', 'deleted_at', 'store_id'],
    'addresses': ['id', 'customer_id', 'address', 'phone', 'email', 'zip', 'default_address', 
                  'created_at', 'updated_at', 'deleted_at', 'is_new_region', 'first_name', 'last_name'],
    'products': ['id', 'name', 'alias', 'product_type_id', 'meta_title', 'meta_description', 
                 'summary', 'published_on', 'content', 'status', 'type', 'created_at', 'updated_at', 
                 'deleted_at', 'vendor_id', 'store_id'],
    'variants': ['id', 'name', 'title', 'product_id', 'sku', 'barcode', 'compare_at_price', 
                 'price', 'position', 'option1', 'option2', 'option3', 'taxable', 'requires_shipping', 
                 'weight', 'weight_unit', 'image_id', 'created_at', 'updated_at', 'deleted_at'],
    'orders': ['id', 'cancel_reason', 'canceled_on', 'confirmed_on', 'checkout_token', 'cart_token', 
               'closed_on', 'customer_id', 'assignee_id', 'created_user_id', 'note', 'order_number', 
               'name', 'fulfillment_status', 'financial_status', 'return_status', 'processed_on', 
               'completed_on', 'billing_address_id', 'shipping_address_id', 'location_id', 'store_id', 
               'source_id', 'edited', 'expected_delivery_date', 'created_at', 'updated_at', 'deleted_at', 
               'channel_id', 'status'],
    'line_items': ['id', 'variant_id', 'reference_id', 'reference_type', 'quantity', 'note', 
                   'price', 'product_name', 'variant_title', 'requires_shipping', 'grams'],
    'carts': ['id', 'token', 'customer_id', 'status', 'store_id', 'created_at', 'updated_at', 'deleted_at'],
    'shipping_lines': ['id', 'shipping_rate_id', 'order_id', 'title', 'price', 'type'],
    'payment_method_lines': ['id', 'order_id', 'payment_method_id', 'payment_method_name', 'amount'],
    'payment_methods': ['id', 'name', 'description', 'status', 'auto_posting_receipt', 'provider_id', 
                        'beneficiary_account_id', 'store_id', 'created_at', 'updated_at', 'deleted_at'],
    'inventory_items': ['id', 'variant_id', 'tracked', 'requires_shipping', 'cost_price', 
                        'created_at', 'updated_at', 'deleted_at', 'lot_management'],
    'inventory_levels': ['id', 'inventory_item_id', 'location_id', 'store_id', 'on_hand', 
                         'available', 'committed', 'incoming', 'created_at', 'updated_at', 'deleted_at'],
    'locations': ['id', 'code', 'name', 'email', 'phone', 'address', 'zip', 'fulfill_order', 
                  'inventory_management', 'default_location', 'status', 'store_id', 'created_at', 
                  'updated_at', 'deleted_at'],
    'product_types': ['id', 'name', 'alias', 'created_at', 'updated_at', 'deleted_at', 'store_id'],
    'vendors': ['id', 'name', 'alias', 'created_at', 'updated_at', 'deleted_at', 'store_id'],
    'tags': ['id', 'name', 'alias', 'created_at', 'updated_at', 'deleted_at', 'store_id'],
    'collections': ['id', 'name', 'alias', 'description', 'meta_title', 'meta_description', 
                    'sort_order', 'type', 'disjunctive', 'store_id', 'created_at', 'updated_at', 'deleted_at'],
    'sources': ['id', 'name', 'description'],
    'product_collections': ['id', 'product_id', 'collection_id', 'auto', 'position'],
    'product_tags': ['id', 'product_id', 'tag_id', 'position'],
}

def validate_table(table_name, csv_path):
    """Validate a single table against its schema"""
    expected_cols = DB_SCHEMA.get(table_name, [])
    
    try:
        df = pd.read_csv(csv_path)
        actual_cols = df.columns.tolist()
        
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)
        
        return {
            'table': table_name,
            'status': 'OK' if not missing and not extra else 'ERROR',
            'expected_count': len(expected_cols),
            'actual_count': len(actual_cols),
            'missing': list(missing),
            'extra': list(extra),
            'rows': len(df)
        }
    except Exception as e:
        return {
            'table': table_name,
            'status': 'FILE_ERROR',
            'error': str(e)
        }

print("\n" + "="*80)
print("VALIDATION: Generated CSV vs Database Schema")
print("="*80)

base_path = 'd:/ai-feature/data/processed'
tables_to_check = [
    'stores', 'customers', 'addresses', 'products', 'variants', 'orders',
    'line_items', 'carts', 'shipping_lines', 'payment_method_lines',
    'payment_methods', 'inventory_items', 'inventory_levels', 'locations',
    'product_types', 'vendors', 'tags', 'collections', 'sources',
    'product_collections', 'product_tags'
]

results = []
for table in tables_to_check:
    result = validate_table(table, f'{base_path}/{table}.csv')
    results.append(result)

# Print summary
errors = []
for r in results:
    if r['status'] == 'OK':
        print(f"✅ {r['table']:25} - {r['rows']:,} rows, {r['actual_count']} columns")
    else:
        print(f"❌ {r['table']:25} - {r['status']}")
        if 'missing' in r and r['missing']:
            print(f"   Missing columns: {', '.join(r['missing'])}")
        if 'extra' in r and r['extra']:
            print(f"   Extra columns: {', '.join(r['extra'])}")
        errors.append(r)

print("\n" + "="*80)
if errors:
    print(f"❌ VALIDATION FAILED: {len(errors)} table(s) with errors")
    print("\nDETAILED ERRORS:")
    for err in errors:
        print(f"\n📋 {err['table']}:")
        if 'missing' in err and err['missing']:
            print(f"  Missing: {err['missing']}")
        if 'extra' in err and err['extra']:
            print(f"  Extra: {err['extra']}")
else:
    print("✅ ALL TABLES MATCH DATABASE SCHEMA PERFECTLY!")
print("="*80 + "\n")
