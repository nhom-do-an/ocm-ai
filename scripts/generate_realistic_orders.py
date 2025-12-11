"""
REALISTIC Order Generator for AI Training
Generates user behavior patterns based on real e-commerce dynamics:
- User personas with preferences
- Item popularity (power-law distribution)
- Co-purchase patterns
- Temporal trends
- Cart interactions (browsing)
- Sequential behavior
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

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


def get_store_data(conn, store_id):
    """Lấy dữ liệu store và phân loại products"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Get customers
        cur.execute("""
            SELECT id, first_name, last_name 
            FROM customers 
            WHERE store_id = %s AND deleted_at IS NULL
        """, (store_id,))
        customers = cur.fetchall()
        
        # Get variants with product info
        cur.execute("""
            SELECT 
                v.id, v.price, v.product_id, v.title as variant_title,
                p.name as product_name, p.vendor_id,
                pt.name as product_type,
                vn.name as vendor_name
            FROM variants v
            JOIN products p ON v.product_id = p.id
            LEFT JOIN product_types pt ON p.product_type_id = pt.id
            LEFT JOIN vendors vn ON p.vendor_id = vn.id
            WHERE p.store_id = %s 
                AND v.price > 0 
                AND v.deleted_at IS NULL 
                AND p.deleted_at IS NULL
            ORDER BY v.id
        """, (store_id,))
        variants = cur.fetchall()
        
        # Get location
        cur.execute("""
            SELECT id 
            FROM locations 
            WHERE store_id = %s AND deleted_at IS NULL 
            LIMIT 1
        """, (store_id,))
        location = cur.fetchone()
        location_id = location['id'] if location else None
        
        # Get max order number
        cur.execute("""
            SELECT COALESCE(MAX(order_number), 1000) as max_num 
            FROM orders 
            WHERE store_id = %s
        """, (store_id,))
        start_order_num = cur.fetchone()['max_num'] + 1
        
        return customers, variants, location_id, start_order_num


def cluster_variants_by_category(variants):
    """Phân loại variants theo product_type và vendor"""
    clusters = defaultdict(list)
    
    for v in variants:
        # Primary cluster by product_type
        ptype = v.get('product_type') or 'Other'
        clusters[ptype].append(v)
    
    return dict(clusters)


def assign_user_personas(customers, clusters):
    """Gán persona cho mỗi user dựa trên clusters available"""
    personas = {}
    cluster_names = list(clusters.keys())
    
    for customer in customers:
        cid = customer['id']
        
        # 40% users có 1 primary interest (focused buyers)
        # 40% users có 2-3 interests (diverse buyers)
        # 20% users mua random (explorers)
        
        rand = random.random()
        if rand < 0.4:
            # Focused buyer - 1 category, weight = 0.8
            primary = random.choice(cluster_names)
            personas[cid] = {
                'type': 'focused',
                'interests': {primary: 0.8, 'random': 0.2}
            }
        elif rand < 0.8:
            # Diverse buyer - 2-3 categories
            num_interests = random.randint(2, min(3, len(cluster_names)))
            selected = random.sample(cluster_names, num_interests)
            weights = np.random.dirichlet(np.ones(num_interests) * 2)  # Generate weights summing to 1
            personas[cid] = {
                'type': 'diverse',
                'interests': {cat: float(w) for cat, w in zip(selected, weights)}
            }
        else:
            # Explorer - mua random
            personas[cid] = {
                'type': 'explorer',
                'interests': {'random': 1.0}
            }
    
    return personas


def create_item_popularity_scores(variants):
    """Tạo popularity scores theo power-law distribution"""
    n = len(variants)
    
    # Power-law: 20% items chiếm 80% popularity
    # Zipf distribution with alpha = 1.5
    ranks = np.arange(1, n + 1)
    scores = 1.0 / (ranks ** 1.5)
    scores = scores / scores.sum()  # Normalize to sum = 1
    
    # Shuffle để không bias theo order
    indices = list(range(n))
    random.shuffle(indices)
    
    popularity = {}
    for i, idx in enumerate(indices):
        popularity[variants[idx]['id']] = float(scores[i])
    
    return popularity


def select_items_for_user(persona, clusters, all_variants, popularity, num_items=3):
    """Chọn items cho user dựa trên persona và popularity"""
    selected = []
    interests = persona['interests']
    
    for _ in range(num_items):
        # Choose category based on interests
        if 'random' in interests and interests['random'] == 1.0:
            # Explorer: choose from all variants weighted by popularity
            variant = random.choices(all_variants, weights=[popularity[v['id']] for v in all_variants])[0]
        else:
            # Choose category weighted by interests
            categories = [k for k in interests.keys() if k != 'random']
            weights = [interests[k] for k in categories]
            
            if random.random() < interests.get('random', 0):
                # Sometimes buy outside interests
                variant = random.choices(all_variants, weights=[popularity[v['id']] for v in all_variants])[0]
            else:
                category = random.choices(categories, weights=weights)[0]
                category_variants = clusters.get(category, all_variants)
                if not category_variants:
                    category_variants = all_variants
                
                # Weight by popularity within category
                cat_weights = [popularity[v['id']] for v in category_variants]
                variant = random.choices(category_variants, weights=cat_weights)[0]
        
        selected.append(variant)
    
    return selected


def create_temporal_trends(variants, days=90):
    """Tạo temporal trends - một số items trending up, một số down"""
    trends = {}
    
    # 30% items trending up (hot items)
    # 50% items stable
    # 20% items trending down (declining)
    
    n_up = int(len(variants) * 0.3)
    n_stable = int(len(variants) * 0.5)
    
    shuffled = list(variants)
    random.shuffle(shuffled)
    
    for i, v in enumerate(shuffled):
        vid = v['id']
        if i < n_up:
            # Trending up: popularity increases linearly over 90 days
            trends[vid] = {'type': 'up', 'start': 0.2, 'end': 1.5}
        elif i < n_up + n_stable:
            # Stable
            trends[vid] = {'type': 'stable', 'start': 1.0, 'end': 1.0}
        else:
            # Trending down
            trends[vid] = {'type': 'down', 'start': 1.2, 'end': 0.3}
    
    return trends


def get_trend_multiplier(variant_id, day, total_days, trends):
    """Get popularity multiplier for a variant on a given day"""
    trend = trends.get(variant_id, {'type': 'stable', 'start': 1.0, 'end': 1.0})
    
    # Linear interpolation from start to end
    progress = day / total_days
    multiplier = trend['start'] + (trend['end'] - trend['start']) * progress
    
    return max(0.1, multiplier)  # Minimum 0.1 to avoid zero probability


def clear_store_data(conn, store_id, clear_customers=False):
    """Xóa dữ liệu cũ của store"""
    print(f"\n🗑️  Clearing old data for store {store_id}...")
    
    with conn.cursor() as cur:
        # Delete line_items first (foreign key constraints)
        cur.execute("""
            DELETE FROM line_items 
            WHERE reference_id IN (
                SELECT id FROM orders WHERE store_id = %s
            ) AND reference_type = 'order'
        """, (store_id,))
        order_items_deleted = cur.rowcount
        
        cur.execute("""
            DELETE FROM line_items 
            WHERE reference_id IN (
                SELECT id FROM carts WHERE store_id = %s
            ) AND reference_type = 'cart'
        """, (store_id,))
        cart_items_deleted = cur.rowcount
        
        # Delete orders
        cur.execute("DELETE FROM orders WHERE store_id = %s", (store_id,))
        orders_deleted = cur.rowcount
        
        # Delete carts
        cur.execute("DELETE FROM carts WHERE store_id = %s", (store_id,))
        carts_deleted = cur.rowcount
        
        # Optionally delete customers
        customers_deleted = 0
        if clear_customers:
            cur.execute("DELETE FROM customers WHERE store_id = %s", (store_id,))
            customers_deleted = cur.rowcount
        
        conn.commit()
    
    print(f"   ✅ Deleted {orders_deleted:,} orders")
    print(f"   ✅ Deleted {order_items_deleted + cart_items_deleted:,} line items")
    print(f"   ✅ Deleted {carts_deleted:,} carts")
    if clear_customers:
        print(f"   ✅ Deleted {customers_deleted:,} customers")


def generate_realistic_orders(conn, store_id, num_orders_per_user_avg=8, days_back=90, cart_ratio=0.3):
    """
    Generate realistic orders với user behavior patterns
    
    Args:
        store_id: Store ID to generate for
        num_orders_per_user_avg: Average orders per user (will vary)
        days_back: Number of days to spread orders over
        cart_ratio: Ratio of users with active carts (0.0-1.0, default 0.3 = 30%)
    """
    print(f"\n🎯 REALISTIC DATA GENERATION FOR STORE {store_id}")
    print("=" * 80)
    
    # Load store data
    print("📥 Loading store data...")
    customers, variants, location_id, start_order_num = get_store_data(conn, store_id)
    
    if not customers:
        print("❌ No customers found for this store")
        return
    
    if not variants:
        print("❌ No variants found for this store")
        return
    
    print(f"   ✅ Customers: {len(customers)}")
    print(f"   ✅ Variants: {len(variants)}")
    print(f"   ✅ Starting order #: {start_order_num}")
    
    # Step 1: Cluster variants by category
    print("\n🏷️  Clustering variants by category...")
    clusters = cluster_variants_by_category(variants)
    print(f"   ✅ Found {len(clusters)} categories:")
    for cat, items in clusters.items():
        print(f"      - {cat}: {len(items)} items")
    
    # Step 2: Assign user personas
    print("\n👥 Assigning user personas...")
    personas = assign_user_personas(customers, clusters)
    persona_types = defaultdict(int)
    for p in personas.values():
        persona_types[p['type']] += 1
    print(f"   ✅ Persona distribution:")
    for ptype, count in persona_types.items():
        print(f"      - {ptype}: {count} users ({count/len(customers)*100:.1f}%)")
    
    # Step 3: Create item popularity scores
    print("\n📊 Creating item popularity scores (power-law)...")
    popularity = create_item_popularity_scores(variants)
    top_items = sorted(popularity.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   ✅ Top 5 popular items:")
    for vid, score in top_items:
        v = next(v for v in variants if v['id'] == vid)
        print(f"      - {v['product_name']}: {score:.4f}")
    
    # Step 4: Create temporal trends
    print("\n📈 Creating temporal trends...")
    trends = create_temporal_trends(variants, days_back)
    trending_up = sum(1 for t in trends.values() if t['type'] == 'up')
    trending_down = sum(1 for t in trends.values() if t['type'] == 'down')
    print(f"   ✅ Trending up: {trending_up} items")
    print(f"   ✅ Stable: {len(variants) - trending_up - trending_down} items")
    print(f"   ✅ Trending down: {trending_down} items")
    
    # Step 5: Generate orders
    print(f"\n🛒 Generating orders...")
    print(f"   Target: ~{num_orders_per_user_avg} orders/user × {len(customers)} users = ~{num_orders_per_user_avg * len(customers)} orders")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    orders_created = 0
    line_items_created = 0
    carts_created = 0
    
    with conn.cursor() as cur:
        for customer in customers:
            customer_id = customer['id']
            persona = personas[customer_id]
            
            # Number of orders for this user (Poisson distribution around average)
            num_orders = max(1, int(np.random.poisson(num_orders_per_user_avg)))
            
            # Generate orders spread over time
            for order_idx in range(num_orders):
                # Day in the period (earlier users started earlier)
                day_in_period = random.randint(0, days_back - 1)
                order_date = start_date + timedelta(days=day_in_period)
                
                # 85% orders completed, 10% processing, 5% cancelled
                rand = random.random()
                if rand < 0.85:
                    status = 'completed'
                    fulfillment_status = 'fulfilled'
                    financial_status = 'paid'
                elif rand < 0.95:
                    status = 'processing'
                    fulfillment_status = random.choice(['partial', 'unfulfilled'])
                    financial_status = random.choice(['paid', 'partially_paid'])
                else:
                    status = 'cancelled'
                    fulfillment_status = 'unfulfilled'
                    financial_status = 'refunded'
                
                order_number = start_order_num + orders_created
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
                    customer_id, store_id, location_id, order_number, name,
                    status, fulfillment_status, financial_status,
                    order_date, order_date,
                    order_date if status != 'pending' else None,
                    order_date if status in ['completed', 'processing'] else None,
                    order_date if status == 'completed' else None
                ))
                
                order_id = cur.fetchone()[0]
                orders_created += 1
                
                # Number of items in order (weighted: 1-3 items most common)
                num_items = random.choices([1, 2, 3, 4, 5], weights=[35, 35, 20, 7, 3])[0]
                
                # Select items based on persona and popularity with temporal trends
                selected_variants = select_items_for_user(
                    persona, clusters, variants, popularity, num_items
                )
                
                # Apply temporal trend multiplier
                adjusted_variants = []
                for v in selected_variants:
                    trend_mult = get_trend_multiplier(v['id'], day_in_period, days_back, trends)
                    if random.random() < trend_mult / 1.5:  # Normalize to keep reasonable selection rate
                        adjusted_variants.append(v)
                
                # If all filtered out, take at least 1
                if not adjusted_variants:
                    adjusted_variants = [selected_variants[0]]
                
                # Create line items
                for variant in adjusted_variants:
                    quantity = random.choices([1, 2, 3], weights=[70, 25, 5])[0]
                    
                    # Price with occasional discount
                    discount = random.uniform(0, 0.25) if random.random() < 0.2 else 0
                    price = float(variant['price']) * (1 - discount)
                    
                    cur.execute("""
                        INSERT INTO line_items (
                            variant_id, reference_id, reference_type, quantity,
                            price, product_name, variant_title, requires_shipping, grams
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        variant['id'], order_id, 'order', quantity,
                        price, variant['product_name'], variant.get('variant_title', ''),
                        True, random.randint(100, 2000)
                    ))
                    
                    line_items_created += 1
            
            # Generate cart interactions (browsing behavior)
            # User-defined ratio of users with active carts
            if random.random() < cart_ratio:
                cart_date = end_date - timedelta(days=random.randint(0, 7))  # Recent carts
                
                cur.execute("""
                    INSERT INTO carts (
                        token, customer_id, status, store_id, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    f"cart_{customer_id}_{random.randint(1000, 9999)}",
                    customer_id, 'active', store_id, cart_date, cart_date
                ))
                
                cart_id = cur.fetchone()[0]
                carts_created += 1
                
                # Cart has 1-3 items
                num_cart_items = random.randint(1, 3)
                cart_variants = select_items_for_user(
                    persona, clusters, variants, popularity, num_cart_items
                )
                
                for variant in cart_variants:
                    quantity = 1
                    price = float(variant['price'])
                    
                    cur.execute("""
                        INSERT INTO line_items (
                            variant_id, reference_id, reference_type, quantity,
                            price, product_name, variant_title, requires_shipping, grams
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        variant['id'], cart_id, 'cart', quantity,
                        price, variant['product_name'], variant.get('variant_title', ''),
                        True, random.randint(100, 2000)
                    ))
                    
                    line_items_created += 1
            
            # Progress update every 50 users
            if (orders_created + 1) % 50 == 0:
                print(f"   ⏳ Progress: {orders_created} orders, {carts_created} carts...")
        
        conn.commit()
    
    print(f"\n✅ GENERATION COMPLETE!")
    print("=" * 80)
    print(f"   📦 Orders created: {orders_created:,}")
    print(f"   📝 Line items created: {line_items_created:,}")
    print(f"   🛒 Carts created: {carts_created:,}")
    print(f"   📊 Avg items/order: {line_items_created/(orders_created+carts_created):.2f}")
    
    return orders_created, line_items_created, carts_created


def check_data_status(conn, store_id=None):
    """Kiểm tra số liệu hiện tại"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if store_id:
            stores = [{'id': store_id}]
            cur.execute("SELECT id, name FROM stores WHERE id = %s AND deleted_at IS NULL", (store_id,))
            store_info = cur.fetchone()
            if store_info:
                stores[0]['name'] = store_info.get('name', f'Store {store_id}')
        else:
            cur.execute("SELECT id, name FROM stores WHERE deleted_at IS NULL ORDER BY id")
            stores = cur.fetchall()
        
        print("\n📊 CURRENT DATA STATUS:")
        print("=" * 80)
        
        for store in stores:
            sid = store['id']
            sname = store.get('name', f'Store {sid}')
            
            # Count data
            cur.execute("SELECT COUNT(*) as cnt FROM customers WHERE store_id = %s AND deleted_at IS NULL", (sid,))
            num_customers = cur.fetchone()['cnt']
            
            cur.execute("SELECT COUNT(*) as cnt FROM orders WHERE store_id = %s", (sid,))
            num_orders = cur.fetchone()['cnt']
            
            cur.execute("SELECT COUNT(*) as cnt FROM carts WHERE store_id = %s AND deleted_at IS NULL", (sid,))
            num_carts = cur.fetchone()['cnt']
            
            cur.execute("""
                SELECT COUNT(DISTINCT li.variant_id) as cnt
                FROM line_items li
                JOIN orders o ON li.reference_id = o.id AND li.reference_type = 'order'
                WHERE o.store_id = %s
            """, (sid,))
            num_variants = cur.fetchone()['cnt']
            
            print(f"\n🏪 {sname} (ID: {sid}):")
            print(f"   - Customers: {num_customers:,}")
            print(f"   - Orders: {num_orders:,}")
            print(f"   - Carts: {num_carts:,}")
            print(f"   - Variants sold: {num_variants:,}")
            
            # Check readiness for AI
            ready = num_customers >= 50 and num_orders >= 100
            status = "✅ Ready for AI" if ready else "❌ Need more data"
            print(f"   - Status: {status}")


def generate_customers(conn, store_id, num_customers):
    """Generate thêm customers"""
    from faker import Faker
    fake = Faker(['vi_VN'])
    
    print(f"\n👥 Generating {num_customers} customers for store {store_id}...")
    
    with conn.cursor() as cur:
        for i in range(num_customers):
            first_name = fake.first_name()
            last_name = fake.last_name()
            email = f"{fake.user_name()}_{store_id}_{i}_{random.randint(1000,9999)}@example.com"
            phone = fake.phone_number()
            
            cur.execute("""
                INSERT INTO customers (
                    first_name, last_name, email, phone, store_id,
                    status, verified_email, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                first_name, last_name, email, phone, store_id,
                'active', True, datetime.now(), datetime.now()
            ))
        
        conn.commit()
    
    print(f"   ✅ Created {num_customers} customers")


def main():
    print("=" * 80)
    print("🚀 REALISTIC ORDER GENERATOR - AI Training Data")
    print("=" * 80)
    
    conn = connect_db()
    
    try:
        # Get available stores
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, name FROM stores WHERE deleted_at IS NULL ORDER BY id")
            stores = cur.fetchall()
        
        if not stores:
            print("❌ No stores found")
            return
        
        print(f"\n📍 Available Stores ({len(stores)}):")
        for store in stores:
            print(f"   - ID: {store['id']}, Name: {store.get('name', 'N/A')}")
        
        # Check current status
        check_data_status(conn)
        
        print("\n" + "=" * 80)
        print("\nChọn action:")
        print("1. Generate thêm customers")
        print("2. Generate realistic orders (with behavior patterns)")
        print("3. Cả hai (customers + orders)")
        print("4. Chỉ check status")
        
        choice = input("\nLựa chọn (1-4): ")
        
        if choice == '4':
            return
        
        # Ask for store scope
        print("\nGenerate cho:")
        print("  a. Tất cả stores")
        print("  b. Một store cụ thể")
        scope = input("Lựa chọn (a/b): ").lower()
        
        target_stores = stores
        if scope == 'b':
            store_id = int(input(f"Nhập Store ID: "))
            target_stores = [s for s in stores if s['id'] == store_id]
            if not target_stores:
                print(f"❌ Store ID {store_id} not found")
                return
        
        # Ask about clearing old data
        print("\n📋 Xử lý dữ liệu cũ:")
        print("  a. Giữ nguyên và thêm mới (append)")
        print("  b. Xóa orders/carts cũ, giữ customers")
        print("  c. Xóa tất cả và tạo mới hoàn toàn")
        clear_choice = input("Lựa chọn (a/b/c): ").lower()
        
        # Get parameters based on choice
        if choice in ['1', '3']:
            num_customers = int(input("\nSố customers mỗi store (default 50): ") or "50")
        
        if choice in ['2', '3']:
            num_orders_per_user = int(input("\nAvg orders per user (default 8): ") or "8")
            days_back = int(input("Spread over how many days? (default 90): ") or "90")
            
            # Cart ratio configuration
            print("\n🛒 Cart Configuration:")
            print(f"   Current: {num_customers if choice == '3' else 'existing'} customers")
            print(f"   Expected orders: ~{num_orders_per_user * (num_customers if choice == '3' else 550):,}")
            print("\n   Cart ratio examples:")
            print("     0.3 = 30% customers có cart (~165 carts)")
            print("     0.5 = 50% customers có cart (~275 carts)")
            print("     0.8 = 80% customers có cart (~440 carts)")
            print("     1.0 = 100% customers có cart (~550 carts)")
            cart_ratio = float(input("\n   Cart ratio (0.0-1.0, default 0.5): ") or "0.5")
        
        print("\n⚠️  WARNING: This will INSERT data into your REAL database!")
        confirm = input("Continue? (yes/no): ").lower()
        
        if confirm != 'yes':
            print("❌ Cancelled")
            return
        
        # Execute generation
        for store in target_stores:
            store_id = store['id']
            store_name = store.get('name', f'Store {store_id}')
            
            print(f"\n{'='*80}")
            print(f"🏪 Processing: {store_name} (ID: {store_id})")
            print('='*80)
            
            # Clear old data if requested
            if clear_choice == 'b':
                clear_store_data(conn, store_id, clear_customers=False)
            elif clear_choice == 'c':
                clear_store_data(conn, store_id, clear_customers=True)
            
            if choice in ['1', '3']:
                generate_customers(conn, store_id, num_customers)
            
            if choice in ['2', '3']:
                generate_realistic_orders(conn, store_id, num_orders_per_user, days_back, cart_ratio)
        
        # Show final status
        print("\n" + "=" * 80)
        print("🎉 GENERATION COMPLETE!")
        print("=" * 80)
        check_data_status(conn, target_stores[0]['id'] if len(target_stores) == 1 else None)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()


if __name__ == '__main__':
    main()
