"""
Script để generate thêm orders và line_items realistic cho database
Giúp đủ data để train AI models (cần tối thiểu 100 orders)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker(['vi_VN'])

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'testocm',
    'user': 'zalolog',
    'password': '123456'
}

def connect_db():
    """Connect to PostgreSQL"""
    return psycopg2.connect(**DB_CONFIG)

def get_all_stores(conn):
    """Lấy danh sách tất cả stores"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, name, deleted_at 
            FROM stores 
            WHERE deleted_at IS NULL
            ORDER BY id
        """)
        stores = cur.fetchall()
        return stores

def get_existing_data(conn, store_id):
    """Lấy dữ liệu hiện có từ database cho một store cụ thể"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Get customers
        cur.execute("SELECT id FROM customers WHERE store_id = %s AND deleted_at IS NULL", (store_id,))
        customers = [row['id'] for row in cur.fetchall()]
        
        # Get variants with prices
        cur.execute("""
            SELECT v.id, v.price, v.product_id, p.name as product_name, v.title
            FROM variants v
            JOIN products p ON v.product_id = p.id
            WHERE p.store_id = %s AND v.price > 0 
                AND v.deleted_at IS NULL AND p.deleted_at IS NULL
        """, (store_id,))
        variants = cur.fetchall()
        
        # Get locations
        cur.execute("SELECT id FROM locations WHERE store_id = %s AND deleted_at IS NULL LIMIT 1", (store_id,))
        location = cur.fetchone()
        location_id = location['id'] if location else None
        
        # Get current max order_number
        cur.execute("SELECT COALESCE(MAX(order_number), 0) as max_num FROM orders WHERE store_id = %s", (store_id,))
        max_order = cur.fetchone()
        start_order_num = max_order['max_num'] + 1 if max_order else 1
        
        print(f"   📊 Store {store_id}:")
        print(f"      - Customers: {len(customers)}")
        print(f"      - Variants: {len(variants)}")
        print(f"      - Start order number: {start_order_num}")
        
        return customers, variants, location_id, start_order_num

def generate_orders(conn, num_orders=100, store_id=None):
    """Generate orders với distribution realistic"""
    stores = get_all_stores(conn)
    
    if not stores:
        print("❌ Không có stores trong DB.")
        return
    
    # Nếu chỉ định store_id, chỉ gen cho store đó
    if store_id:
        stores = [s for s in stores if s['id'] == store_id]
        if not stores:
            print(f"❌ Không tìm thấy store_id={store_id}")
            return
    
    print(f"\n🔨 Generating orders for {len(stores)} store(s)...")
    
    total_orders_created = 0
    total_line_items_created = 0
    
    # Order statuses với distribution realistic
    statuses = ['completed'] * 70 + ['processing'] * 20 + ['pending'] * 7 + ['cancelled'] * 3
    
    # Date range: 90 ngày gần đây
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    for store in stores:
        current_store_id = store['id']
        store_name = store.get('name', f'Store {current_store_id}')
        
        print(f"\n   🏪 Processing: {store_name} (ID: {current_store_id})")
        
        customers, variants, location_id, start_order_num = get_existing_data(conn, current_store_id)
        
        if not customers:
            print(f"      ⚠️  No customers for this store, skipping...")
            continue
        
        if not variants:
            print(f"      ⚠️  No variants for this store, skipping...")
            continue
        
        with conn.cursor() as cur:
            orders_created = 0
            line_items_created = 0
            
            for i in range(num_orders):
                # Random date trong 90 ngày
                days_ago = random.randint(0, 90)
                order_date = end_date - timedelta(days=days_ago)
                
                # Random customer
                customer_id = random.choice(customers)
            # Random date trong 90 ngày
            days_ago = random.randint(0, 90)
            order_date = end_date - timedelta(days=days_ago)
            
            # Random customer
            customer_id = random.choice(customers)
            
            # Random status
            status = random.choice(statuses)
            
            # Fulfillment status
            if status == 'completed':
                fulfillment_status = 'fulfilled'
            elif status == 'cancelled':
                fulfillment_status = 'unfulfilled'
            else:
                fulfillment_status = random.choice(['fulfilled', 'partial', 'unfulfilled'])
            
            # Financial status
            if status == 'completed':
                financial_status = 'paid'
            elif status == 'cancelled':
                financial_status = random.choice(['paid', 'refunded', 'unpaid'])
            else:
                financial_status = random.choice(['paid', 'pending', 'partially_paid'])
            
            order_number = start_order_num + i
            name = f"#{order_number}"
            
            # Create order
            cur.execute("""
                INSERT INTO orders (
                    customer_id, store_id, location_id, order_number, name,
                    status, fulfillment_status, financial_status,
                    created_at, updated_at, processed_on, confirmed_on, completed_on
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                customer_id, current_store_id, location_id, order_number, name,
                status, fulfillment_status, financial_status,
                order_date, order_date, 
                order_date if status != 'pending' else None,
                order_date if status != 'pending' else None,
                order_date if status == 'completed' else None
            ))
            
            order_id = cur.fetchone()[0]
            orders_created += 1
            
            # Create line_items (1-5 items per order)
            num_items = random.choices([1, 2, 3, 4, 5], weights=[30, 35, 20, 10, 5])[0]
            selected_variants = random.sample(variants, min(num_items, len(variants)))
            
            for variant in selected_variants:
                quantity = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3])[0]
                
                # Price có thể có discount 0-30%
                discount = random.uniform(0, 0.3) if random.random() < 0.3 else 0
                price = float(variant['price']) * (1 - discount)
                
                cur.execute("""
                    INSERT INTO line_items (
                        variant_id, reference_id, reference_type, quantity,
                        price, product_name, variant_title, requires_shipping, grams
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    variant['id'], order_id, 'order', quantity,
                    price, variant['product_name'], variant['title'],
                    True, random.randint(100, 2000)
                ))
                
                line_items_created += 1
            
                if (i + 1) % 20 == 0:
                    print(f"      ✓ Generated {i + 1}/{num_orders} orders...")
        
            conn.commit()
            
            total_orders_created += orders_created
            total_line_items_created += line_items_created
            
            print(f"      ✅ Store {current_store_id}: {orders_created} orders, {line_items_created} line items")
    
    print(f"\n✅ Hoàn thành tất cả!")
    print(f"   - Total orders created: {total_orders_created}")
    print(f"   - Total line items created: {total_line_items_created}")
    if total_orders_created > 0:
        print(f"   - Avg items/order: {total_line_items_created/total_orders_created:.1f}")

def generate_customers(conn, num_customers=50, store_id=None):
    """Generate thêm customers"""
    stores = get_all_stores(conn)
    
    if not stores:
        print("❌ Không có stores trong DB.")
        return
    
    # Nếu chỉ định store_id, chỉ gen cho store đó
    if store_id:
        stores = [s for s in stores if s['id'] == store_id]
        if not stores:
            print(f"❌ Không tìm thấy store_id={store_id}")
            return
    
    print(f"\n👥 Generating customers for {len(stores)} store(s)...")
    
    total_created = 0
    
    for store in stores:
        current_store_id = store['id']
        store_name = store.get('name', f'Store {current_store_id}')
        
        print(f"   🏪 {store_name} (ID: {current_store_id}): Creating {num_customers} customers...")
        
        with conn.cursor() as cur:
            for i in range(num_customers):
                first_name = fake.first_name()
                last_name = fake.last_name()
                email = f"{fake.user_name()}_{current_store_id}_{i}@example.com"  # Unique email
                phone = fake.phone_number()
                
                cur.execute("""
                    INSERT INTO customers (
                        first_name, last_name, email, phone, store_id,
                        status, verified_email, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    first_name, last_name, email, phone, current_store_id,
                    'active', True, datetime.now(), datetime.now()
                ))
            
            conn.commit()
            total_created += num_customers
            print(f"      ✅ Created {num_customers} customers")
    
    print(f"\n✅ Total customers created: {total_created}")

def check_data_readiness(conn, store_id=None):
    """Kiểm tra xem đã đủ data để train AI chưa"""
    stores = get_all_stores(conn)
    
    if not stores:
        print("❌ Không có stores trong DB.")
        return False
    
    # Nếu chỉ định store_id, chỉ check store đó
    if store_id:
        stores = [s for s in stores if s['id'] == store_id]
        if not stores:
            print(f"❌ Không tìm thấy store_id={store_id}")
            return False
    
    print(f"\n📈 Thống kê dữ liệu cho {len(stores)} store(s):")
    print("=" * 80)
    
    all_ready = True
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for store in stores:
            current_store_id = store['id']
            store_name = store.get('name', f'Store {current_store_id}')
            
            # Count orders
            cur.execute("SELECT COUNT(*) as count FROM orders WHERE store_id = %s", (current_store_id,))
            total_orders = cur.fetchone()['count']
            
            # Count customers
            cur.execute("SELECT COUNT(*) as count FROM customers WHERE store_id = %s", (current_store_id,))
            total_customers = cur.fetchone()['count']
            
            # Count products
            cur.execute("SELECT COUNT(*) as count FROM products WHERE store_id = %s", (current_store_id,))
            total_products = cur.fetchone()['count']
            
            # Count distinct variants in orders
            cur.execute("""
                SELECT COUNT(DISTINCT li.variant_id) as count
                FROM line_items li
                JOIN orders o ON li.reference_id = o.id AND li.reference_type = 'order'
                WHERE o.store_id = %s
            """, (current_store_id,))
            variants_ordered = cur.fetchone()['count']
            
            # Check readiness
            is_ready = total_orders >= 100 and total_customers >= 50 and total_products >= 20
            status_icon = '✅' if is_ready else '❌'
            
            print(f"\n{status_icon} {store_name} (ID: {current_store_id}):")
            print(f"   - Orders: {total_orders:,} {'✅' if total_orders >= 100 else f'❌ (need {100 - total_orders} more)'}")
            print(f"   - Customers: {total_customers:,} {'✅' if total_customers >= 50 else f'❌ (need {50 - total_customers} more)'}")
            print(f"   - Products: {total_products:,} {'✅' if total_products >= 20 else f'❌ (need {20 - total_products} more)'}")
            print(f"   - Variants ordered: {variants_ordered:,}")
            
            if not is_ready:
                all_ready = False
    
    print("\n" + "=" * 80)
    if all_ready:
        print("🎉 Tất cả stores đã đủ dữ liệu để train AI models!")
    else:
        print("⚠️  Một số stores chưa đủ dữ liệu để train AI")
    
    return all_ready

def main():
    print("=" * 80)
    print("🚀 Data Generator for AI Training - Multi-Store Support")
    print("=" * 80)
    
    conn = connect_db()
    
    try:
        # Show available stores
        stores = get_all_stores(conn)
        if not stores:
            print("❌ Không có stores trong DB.")
            return
        
        print(f"\n📍 Available Stores ({len(stores)}):")
        for store in stores:
            print(f"   - ID: {store['id']}, Name: {store.get('name', 'N/A')}")
        
        # Check current status
        check_data_readiness(conn)
        
        print("\n" + "=" * 80)
        print("\nChọn action:")
        print("1. Generate thêm customers")
        print("2. Generate thêm orders")
        print("3. Cả hai (customers + orders)")
        print("4. Chỉ check status")
        
        choice = input("\nLựa chọn (1-4): ")
        
        # Ask for store scope
        store_id = None
        if choice in ['1', '2', '3']:
            scope = input("\nGenerate cho:\n  a. Tất cả stores\n  b. Một store cụ thể\nLựa chọn (a/b): ").lower()
            if scope == 'b':
                store_id = int(input(f"Nhập Store ID (1-{len(stores)}): "))
        
        if choice == '1':
            num = int(input("Số lượng customers mỗi store (default 50): ") or "50")
            generate_customers(conn, num, store_id)
        elif choice == '2':
            num = int(input("Số lượng orders mỗi store (default 100): ") or "100")
            generate_orders(conn, num, store_id)
        elif choice == '3':
            num_customers = int(input("Số lượng customers mỗi store (default 50): ") or "50")
            num_orders = int(input("Số lượng orders mỗi store (default 100): ") or "100")
            generate_customers(conn, num_customers, store_id)
            generate_orders(conn, num_orders, store_id)
        
        # Check final status
        print("\n" + "=" * 80)
        check_data_readiness(conn, store_id)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    main()
