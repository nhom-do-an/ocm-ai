"""
Script sinh dữ liệu REALISTIC - Version 4
Dựa trên dữ liệu thật từ ocm-data.sql
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
import numpy as np

fake = Faker('vi_VN')
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ================================
# CONFIGURATION
# ================================

NUM_STORES = 20
NUM_CUSTOMERS_PER_STORE = 500
NUM_PRODUCTS_PER_STORE = 50
NUM_VARIANTS_PER_PRODUCT = (1, 4)
NUM_ORDERS_PER_CUSTOMER = (1, 15)

OUTPUT_DIR = 'd:/ai-feature/data/processed'

# ================================
# TÊN THẬT TỪ DATABASE
# ================================

VIETNAMESE_FIRST_NAMES = [
    'Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Phan', 'Vũ', 'Đặng', 'Bùi', 'Đỗ',
    'Hồ', 'Ngô', 'Dương', 'Lý', 'Võ', 'Huỳnh', 'Mai', 'Chu', 'Tô', 'Hà'
]

VIETNAMESE_LAST_NAMES = [
    # Nam
    'Hùng', 'Minh', 'Tuấn', 'Anh', 'Đức', 'Quân', 'Long', 'Nam', 'Khoa', 'Phong',
    'Thắng', 'Tùng', 'Dũng', 'Cường', 'Hoàng', 'Hải', 'Sơn', 'Tân', 'Thành', 'Vinh',
    # Nữ
    'Linh', 'Hương', 'Lan', 'Mai', 'Hà', 'Thảo', 'Ngọc', 'Trang', 'Chi', 'My',
    'Vy', 'Anh', 'Phương', 'Thu', 'Hoa', 'Nhung', 'Tú', 'Thúy', 'Diệu', 'Giang',
    # 2 chữ
    'Văn Hùng', 'Thị Lan', 'Minh Anh', 'Hoàng Long', 'Quốc Tuấn', 'Thu Hương',
    'Đức Thắng', 'Minh Chi', 'Thanh Tùng', 'Khánh Linh'
]

# ================================
# TÊN SẢN PHẨM THẬT
# ================================

PRODUCT_TEMPLATES = {
    'fashion': {
        'Áo thun': [
            'Áo thun {brand} cotton form rộng unisex',
            'Áo thun {brand} tay ngắn basic',
            'Áo thun {brand} oversize streetwear', 
            'Áo thun {brand} cotton 4 chiều co giãn',
            'Áo thun {brand} form slim fit cao cấp'
        ],
        'Áo sơ mi': [
            'Áo sơ mi {brand} tay dài công sở',
            'Áo sơ mi {brand} oxford slim fit',
            'Áo sơ mi {brand} linen mát mẻ',
            'Áo sơ mi {brand} kẻ sọc basic',
        ],
        'Quần jean': [
            'Quần jean {brand} baggy rộng rãi',
            'Quần jean {brand} skinny ôm body',
            'Quần jean {brand} slim fit co giãn',
            'Quần jean {brand} ống suông basic',
        ],
        'Áo khoác': [
            'Áo khoác {brand} hoodie nỉ bông',
            'Áo khoác {brand} dù 2 lớp chống nắng',
            'Áo khoác {brand} bomber cá tính',
            'Áo khoác {brand} cardigan len mỏng',
        ],
    },
    'electronics': [
        'Điện thoại {brand} {model} {capacity} - Chính hãng VN',
        'Laptop {brand} {model} {processor} {ram} {storage}',
        'Tai nghe {brand} {model} bluetooth 5.0',
        'Sạc dự phòng {brand} {capacity} sạc nhanh PD',
        'Chuột {brand} {model} gaming RGB',
        'Bàn phím {brand} cơ {switch} LED RGB',
        'Màn hình {brand} {size}" {resolution} {hz}Hz',
    ],
    'home': [
        'Nồi chống dính Ceramic {brand} {model} Size {size}cm',
        'Chảo chống dính {brand} {model} đáy từ Size {size}cm',
        'Bộ nồi {brand} {material} Size {size1},{size2},{size3}cm',
        'Quánh {brand} {model} Size {size}cm',
        'Máy xay sinh tố {brand} công suất {watt}W',
        'Bình giữ nhiệt {brand} inox 316 dung tích {volume}ml',
        'Bình đun siêu tốc {brand} {volume}L',
    ],
    'food': [
        'Gạo {brand} hạt dài {weight}kg',
        'Dầu ăn {brand} {type} chai {volume}L',
        'Nước mắm {brand} {protein} chai {volume}ml',
        'Sữa tươi {brand} nguyên kem hộp {volume}ml',
        'Trứng gà {brand} vỉ 10 quả',
    ],
}

BRAND_NAMES = {
    'fashion': ['Uniqlo', 'H&M', 'Zara', 'IVY moda', 'Canifa', 'Yody', 'Routine', 'The Blues'],
    'electronics': ['Apple', 'Samsung', 'Xiaomi', 'Logitech', 'Sony', 'Dell', 'HP', 'Asus', 'Lenovo', 'Razer'],
    'home': ['Elmich', 'Lock&Lock', 'Sunhouse', 'Philips', 'Panasonic', 'Tefal', 'Happycook'],
    'food': ['Vinamilk', 'TH True Milk', 'CP', 'Vissan', 'Neptune', 'Cholimex', 'Meizan'],
}

# ================================
# CHANNELS & MASTER DATA
# ================================

CHANNELS = {
    1: 'Website',
    2: 'POS',
    3: 'Facebook',
    4: 'Shopee',
    5: 'TikTokShop',
    6: 'Lazada',
    7: 'Tiki'
}

PAYMENT_METHODS = [
    {'name': 'Tiền mặt', 'status': 'active'},
    {'name': 'Chuyển khoản', 'status': 'active'},
    {'name': 'Ví điện tử', 'status': 'active'},
    {'name': 'COD', 'status': 'active'},
]

STORE_CATEGORIES = {
    'fashion': {
        'name': 'Thời trang',
        'product_types': ['Áo thun', 'Áo sơ mi', 'Quần jean', 'Váy', 'Áo khoác', 'Quần short', 'Đầm', 'Áo len'],
        'vendors': ['Uniqlo', 'H&M', 'Zara', 'IVY moda', 'Canifa', 'Yody'],
        'tags': ['Xuân hè', 'Thu đông', 'Sale off', 'New arrivals', 'Best seller', 'Trending'],
        'collections': ['Xuân hè 2024', 'Thu đông 2024', 'Sale off', 'New arrivals'],
        'price_range': (100000, 1000000),
    },
    'electronics': {
        'name': 'Điện tử',
        'product_types': ['Điện thoại', 'Laptop', 'Tai nghe', 'Sạc dự phòng', 'Chuột', 'Bàn phím', 'Màn hình'],
        'vendors': ['Apple', 'Samsung', 'Xiaomi', 'Logitech', 'Sony', 'Dell', 'HP', 'Asus'],
        'tags': ['Flagship', 'Gaming', 'Budget', 'Premium', 'Hot deal'],
        'collections': ['Flagship 2024', 'Gaming gear', 'Work from home'],
        'price_range': (200000, 30000000),
    },
    'home': {
        'name': 'Gia dụng',
        'product_types': ['Nồi', 'Chảo', 'Bộ nồi', 'Quánh', 'Máy xay', 'Bình giữ nhiệt', 'Bình đun'],
        'vendors': ['Elmich', 'Philips', 'Sunhouse', 'Lock&Lock', 'Panasonic'],
        'tags': ['Nhà bếp', 'Tiết kiệm điện', 'Cao cấp', 'Gia đình'],
        'collections': ['Nhà bếp thông minh', 'Tiết kiệm điện', 'Cao cấp'],
        'price_range': (150000, 5000000),
    },
    'food': {
        'name': 'Thực phẩm',
        'product_types': ['Gạo', 'Dầu ăn', 'Nước mắm', 'Sữa tươi', 'Trứng'],
        'vendors': ['Vinamilk', 'CP', 'Vissan', 'TH True Milk', 'Neptune'],
        'tags': ['Hàng tươi', 'Organic', 'An toàn'],
        'collections': ['Hàng tươi mỗi ngày', 'Organic'],
        'price_range': (10000, 300000),
    },
}

# ================================
# HELPER FUNCTIONS
# ================================

def generate_vietnamese_name():
    """Sinh tên người Việt thật"""
    first = random.choice(VIETNAMESE_FIRST_NAMES)
    last = random.choice(VIETNAMESE_LAST_NAMES)
    return first, last

def generate_product_name(product_type, category, brand):
    """Sinh tên sản phẩm thật"""
    if category == 'fashion':
        if product_type in PRODUCT_TEMPLATES['fashion']:
            template = random.choice(PRODUCT_TEMPLATES['fashion'][product_type])
            return template.format(brand=brand)
    
    if category == 'electronics':
        template = random.choice(PRODUCT_TEMPLATES['electronics'])
        replacements = {
            'brand': brand,
            'model': random.choice(['Pro', 'Max', 'Plus', 'Ultra', 'Air', 'Mini']),
            'capacity': random.choice(['128GB', '256GB', '512GB', '1TB']),
            'processor': random.choice(['i5', 'i7', 'i9', 'Ryzen 5', 'Ryzen 7', 'M1', 'M2']),
            'ram': random.choice(['8GB', '16GB', '32GB']),
            'storage': random.choice(['256GB SSD', '512GB SSD', '1TB SSD']),
            'size': random.choice(['24', '27', '32']),
            'resolution': random.choice(['Full HD', '2K', '4K']),
            'hz': random.choice(['60', '75', '144', '165']),
            'switch': random.choice(['Blue', 'Red', 'Brown']),
        }
        for key, val in replacements.items():
            template = template.replace('{'+key+'}', val)
        return template
    
    if category == 'home':
        template = random.choice(PRODUCT_TEMPLATES['home'])
        replacements = {
            'brand': brand,
            'model': random.choice(['Harmony', 'Mocha', 'Olive', 'Classic', 'Premium']),
            'material': random.choice(['inox 304', 'ceramic', 'chống dính']),
            'size': str(random.choice([16, 18, 20, 22, 24, 26, 28])),
            'size1': '18', 'size2': '20', 'size3': '24',
            'watt': str(random.choice([300, 500, 800, 1000])),
            'volume': str(random.choice([350, 500, 750, 1000, 1500])),
        }
        for key, val in replacements.items():
            template = template.replace('{'+key+'}', val)
        return template
    
    if category == 'food':
        template = random.choice(PRODUCT_TEMPLATES['food'])
        replacements = {
            'brand': brand,
            'type': random.choice(['hạt cải', 'olive', 'hướng dương']),
            'weight': str(random.choice([5, 10, 20])),
            'volume': str(random.choice([500, 650, 900, 1000])),
            'protein': random.choice(['30N', '40N', '50N']),
        }
        for key, val in replacements.items():
            template = template.replace('{'+key+'}', val)
        return template
    
    return f"{brand} {product_type}"

# ================================
# GENERATION FUNCTIONS
# ================================

def generate_stores(num_stores):
    """Sinh stores - nhiều cửa hàng cùng bán 1 ngành hàng"""
    stores = []
    
    # Phân bổ stores theo category để có nhiều shop cùng ngành
    # Fashion: 40%, Electronics: 30%, Home: 20%, Food: 10%
    categories = ['fashion', 'electronics', 'home', 'food']
    category_weights = [0.4, 0.3, 0.2, 0.1]
    
    # Tính số stores cho mỗi category
    stores_per_category = {}
    remaining = num_stores
    for i, cat in enumerate(categories):
        if i < len(categories) - 1:
            count = int(num_stores * category_weights[i])
            stores_per_category[cat] = count
            remaining -= count
        else:
            stores_per_category[cat] = remaining  # Category cuối nhận phần còn lại
    
    # Tên shop prefix đa dạng theo category
    shop_prefixes = {
        'fashion': ['Shop thời trang', 'Store', 'Boutique', 'Fashion', 'Clothing'],
        'electronics': ['Shop điện tử', 'Tech Store', 'Điện máy', 'Digital', 'Gadget'],
        'home': ['Shop gia dụng', 'Home Store', 'Nhà cửa', 'Living', 'HomeWare'],
        'food': ['Shop thực phẩm', 'Food Store', 'Fresh', 'Organic', 'Market'],
    }
    
    store_id = 1
    for category, count in stores_per_category.items():
        cat_name = STORE_CATEGORIES[category]['name']
        prefixes = shop_prefixes[category]
        
        for i in range(count):
            prefix = random.choice(prefixes)
            # Tạo tên shop đa dạng
            if random.random() < 0.5:
                store_name = f"{prefix} {fake.company()}"
            else:
                first_name, last_name = generate_vietnamese_name()
                store_name = f"{prefix} {first_name} {last_name}"
            
            alias = f"store-{store_id}-{category}-{i+1}"
            
            stores.append({
                'id': store_id,
                'name': store_name,
                'trade_name': store_name,
                'email': fake.email(),
                'phone': fake.phone_number(),
                'domain': f"{alias}.myshop.vn",
                'alias': alias,
                'address': fake.address(),
                'currency': 'VND',
                'timezone': 'SE Asia Standard Time',
                'money_format': '{{amount_no_decimals_with_comma_separator}}₫',
                'money_with_currency_format': '{{amount_no_decimals_with_comma_separator}} VND',
                'weight_unit': 'g',
                'created_at': fake.date_time_between(start_date='-2y', end_date='-6m'),
                'updated_at': fake.date_time_between(start_date='-6m', end_date='now'),
                'deleted_at': None,
                '_category': category,  # Internal field - not in DB schema
            })
            store_id += 1
    
    return pd.DataFrame(stores)

def generate_customers(stores_df, num_per_store):
    """Sinh customers với tên Việt thật"""
    customers = []
    customer_id = 1
    
    for _, store in stores_df.iterrows():
        for _ in range(num_per_store):
            first_name, last_name = generate_vietnamese_name()
            email = fake.email()
            
            customers.append({
                'id': customer_id,
                'email': email,
                'phone': fake.phone_number(),
                'first_name': first_name,
                'last_name': last_name,
                'password': fake.sha256(),
                'gender': random.choice(['male', 'female', 'other']),
                'dob': fake.date_of_birth(minimum_age=18, maximum_age=65).strftime('%Y-%m-%d'),
                'note': '',
                'status': 'enabled',  # Thực tế hầu hết đều enabled
                'verified_email': random.choice([True, False]),
                'created_at': fake.date_time_between(start_date=store['created_at'], end_date='now'),
                'updated_at': fake.date_time_between(start_date='-3m', end_date='now'),
                'deleted_at': None,
                'store_id': store['id'],
            })
            customer_id += 1
    
    return pd.DataFrame(customers)

def generate_addresses(customers_df):
    """Sinh addresses"""
    addresses = []
    address_id = 1
    
    for _, customer in customers_df.iterrows():
        if random.random() < 0.83:  # 83% có địa chỉ (giống thực tế)
            num_addresses = random.choices([1, 2], weights=[0.85, 0.15])[0]
            
            for i in range(num_addresses):
                addresses.append({
                    'id': address_id,
                    'customer_id': customer['id'],
                    'address': fake.address(),
                    'phone': customer['phone'],
                    'email': customer['email'],
                    'zip': '',
                    'default_address': (i == 0),
                    'created_at': customer['created_at'],
                    'updated_at': customer['updated_at'],
                    'deleted_at': None,
                    'is_new_region': False,
                    'first_name': customer['first_name'],
                    'last_name': customer['last_name'],
                })
                address_id += 1
    
    return pd.DataFrame(addresses)

def generate_metadata_tables(stores_df):
    """Sinh product_types, vendors, tags, collections, sources"""
    product_types = []
    vendors = []
    tags = []
    collections = []
    sources = []
    
    pt_id = 1
    vendor_id = 1
    tag_id = 1
    collection_id = 1
    source_id = 1
    
    for _, store in stores_df.iterrows():
        category = store['_category']
        cat_data = STORE_CATEGORIES[category]
        
        # Product Types
        for pt_name in cat_data['product_types']:
            product_types.append({
                'id': pt_id,
                'name': pt_name,
                'alias': f"{pt_name.lower().replace(' ', '-')}-{store['id']}",
                'created_at': store['created_at'],
                'updated_at': store['updated_at'],
                'deleted_at': None,
                'store_id': store['id'],
            })
            pt_id += 1
        
        # Vendors
        for vendor_name in cat_data['vendors']:
            vendors.append({
                'id': vendor_id,
                'name': vendor_name,
                'alias': f"{vendor_name.lower().replace(' ', '-')}-{store['id']}",
                'created_at': store['created_at'],
                'updated_at': store['updated_at'],
                'deleted_at': None,
                'store_id': store['id'],
            })
            vendor_id += 1
        
        # Tags
        for tag_name in cat_data['tags']:
            tags.append({
                'id': tag_id,
                'name': tag_name,
                'alias': f"{tag_name.lower().replace(' ', '-')}-{store['id']}",
                'created_at': store['created_at'],
                'updated_at': store['updated_at'],
                'deleted_at': None,
                'category': 'product',
                'store_id': store['id'],
            })
            tag_id += 1
        
        # Collections
        for col_name in cat_data['collections']:
            collections.append({
                'id': collection_id,
                'name': col_name,
                'alias': f"{col_name.lower().replace(' ', '-')}-{store['id']}",
                'description': f'Bộ sưu tập {col_name}',
                'meta_title': col_name,
                'meta_description': f'Bộ sưu tập {col_name}',
                'sort_order': 'best-selling',
                'type': random.choice(['manual', 'smart']),
                'disjunctive': False,
                'store_id': store['id'],
                'created_at': store['created_at'],
                'updated_at': store['updated_at'],
                'deleted_at': None,
            })
            collection_id += 1
        
        # Sources
        for source_name in ['Website', 'Mobile App', 'POS', 'Facebook', 'Marketplace']:
            sources.append({
                'id': source_id,
                'name': source_name,
                'alias': source_name.lower().replace(' ', '-'),
                'default_source': (source_name == 'Website'),
                'image_url': None,
            })
            source_id += 1
    
    return {
        'product_types': pd.DataFrame(product_types),
        'vendors': pd.DataFrame(vendors),
        'tags': pd.DataFrame(tags),
        'collections': pd.DataFrame(collections),
        'sources': pd.DataFrame(sources),
    }

def generate_products_and_variants(stores_df, metadata):
    """Sinh products và variants với tên thật"""
    products = []
    variants = []
    product_collections = []
    product_tags = []
    
    product_id = 1
    variant_id = 1
    pc_id = 1
    pt_id = 1
    
    pt_by_store = metadata['product_types'].groupby('store_id')
    vendors_by_store = metadata['vendors'].groupby('store_id')
    collections_by_store = metadata['collections'].groupby('store_id')
    tags_by_store = metadata['tags'].groupby('store_id')
    
    for _, store in stores_df.iterrows():
        category = store['_category']
        price_min, price_max = STORE_CATEGORIES[category]['price_range']
        
        store_pts = pt_by_store.get_group(store['id'])
        store_vendors = vendors_by_store.get_group(store['id'])
        store_collections = collections_by_store.get_group(store['id'])
        store_tags = tags_by_store.get_group(store['id'])
        
        for _ in range(NUM_PRODUCTS_PER_STORE):
            pt = store_pts.sample(1).iloc[0]
            vendor = store_vendors.sample(1).iloc[0]
            
            # Tên sản phẩm thật
            product_name = generate_product_name(pt['name'], category, vendor['name'])
            
            products.append({
                'id': product_id,
                'name': product_name,
                'alias': f"{product_name[:50].lower().replace(' ', '-')}-{product_id}",
                'product_type_id': pt['id'],
                'meta_title': product_name,
                'meta_description': f'Sản phẩm {product_name}',
                'summary': '',
                'published_on': fake.date_time_between(start_date=store['created_at'], end_date='now'),
                'content': '',
                'status': random.choice(['active', 'active', 'active', 'inactive']),
                'type': 'normal',
                'created_at': store['created_at'],
                'updated_at': store['updated_at'],
                'deleted_at': None,
                'vendor_id': vendor['id'],
                'store_id': store['id'],
            })
            
            # Product Collections
            selected_collections = store_collections.sample(n=min(random.randint(1, 3), len(store_collections)))
            for pos, (_, col) in enumerate(selected_collections.iterrows()):
                product_collections.append({
                    'id': pc_id,
                    'product_id': product_id,
                    'collection_id': col['id'],
                    'position': pos + 1,
                    'created_at': store['created_at'],
                    'updated_at': store['updated_at'],
                    'deleted_at': None,
                })
                pc_id += 1
            
            # Product Tags
            selected_tags = store_tags.sample(n=min(random.randint(2, 4), len(store_tags)))
            for _, tag in selected_tags.iterrows():
                product_tags.append({
                    'id': pt_id,
                    'product_id': product_id,
                    'tag_id': tag['id'],
                    'created_at': store['created_at'],
                    'updated_at': store['updated_at'],
                    'deleted_at': None,
                })
                pt_id += 1
            
            # Variants
            num_variants = random.randint(*NUM_VARIANTS_PER_PRODUCT)
            base_price = random.randint(price_min, price_max)
            
            variant_options = []
            if category == 'fashion':
                variant_options = [
                    ('Size S', base_price),
                    ('Size M', base_price + 20000),
                    ('Size L', base_price + 40000),
                    ('Size XL', base_price + 60000),
                ]
            elif category == 'electronics':
                variant_options = [
                    ('128GB', base_price),
                    ('256GB', base_price + 2000000),
                    ('512GB', base_price + 5000000),
                ]
            else:
                variant_options = [('Standard', base_price)]
            
            for i in range(min(num_variants, len(variant_options))):
                title, price = variant_options[i]
                
                variants.append({
                    'id': variant_id,
                    'product_id': product_id,
                    'title': title,
                    'sku': f"SKU-{product_id}-{i+1}",
                    'barcode': fake.ean13(),
                    'price': price,
                    'compare_at_price': int(price * 1.2),
                    'weight': random.randint(100, 2000),
                    'weight_type': 'g',
                    'created_at': store['created_at'],
                    'updated_at': store['updated_at'],
                    'deleted_at': None,
                })
                variant_id += 1
            
            product_id += 1
    
    return {
        'products': pd.DataFrame(products),
        'variants': pd.DataFrame(variants),
        'product_collections': pd.DataFrame(product_collections),
        'product_tags': pd.DataFrame(product_tags),
    }

def generate_inventory(variants_df, stores_df):
    """Sinh inventory"""
    inventory_items = []
    inventory_levels = []
    locations = []
    
    # Locations (1 per store)
    for _, store in stores_df.iterrows():
        locations.append({
            'id': store['id'],
            'name': f"Kho {store['name']}",
            'code': f"WH-{store['id']}",
            'address': fake.address(),
            'city': fake.city(),
            'province': random.choice(['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ']),
            'is_default': True,
            'store_id': store['id'],
            'created_at': store['created_at'],
            'updated_at': store['updated_at'],
            'deleted_at': None,
        })
    
    locations_df = pd.DataFrame(locations)
    
    inv_item_id = 1
    inv_level_id = 1
    
    for _, variant in variants_df.iterrows():
        # Inventory Item
        inventory_items.append({
            'id': inv_item_id,
            'variant_id': variant['id'],
            'sku': variant['sku'],
            'created_at': variant['created_at'],
            'updated_at': variant['updated_at'],
            'deleted_at': None,
        })
        
        # Inventory Level (find location by store)
        store_location = locations_df[locations_df['store_id'] == 
                                     variants_df[variants_df['id'] == variant['id']].iloc[0]['product_id'] % len(stores_df) + 1]
        
        if len(store_location) > 0:
            location_id = store_location.iloc[0]['id']
        else:
            location_id = 1
        
        inventory_levels.append({
            'id': inv_level_id,
            'inventory_item_id': inv_item_id,
            'location_id': location_id,
            'available': random.randint(0, 1000),
            'created_at': variant['created_at'],
            'updated_at': variant['updated_at'],
            'deleted_at': None,
        })
        
        inv_item_id += 1
        inv_level_id += 1
    
    return {
        'inventory_items': pd.DataFrame(inventory_items),
        'inventory_levels': pd.DataFrame(inventory_levels),
        'locations': locations_df,
    }

def generate_payment_methods(stores_df):
    """Sinh payment_methods"""
    payment_methods = []
    pm_id = 1
    
    for _, store in stores_df.iterrows():
        for pm_data in PAYMENT_METHODS:
            payment_methods.append({
                'id': pm_id,
                'name': pm_data['name'],
                'description': None,
                'status': pm_data['status'],
                'auto_posting_receipt': True,
                'provider_id': None,
                'beneficiary_account_id': None,
                'store_id': store['id'],
                'created_at': store['created_at'],
                'updated_at': store['updated_at'],
                'deleted_at': None,
            })
            pm_id += 1
    
    return pd.DataFrame(payment_methods)

def generate_orders_and_line_items(customers_df, products_df, variants_df, addresses_df, payment_methods_df, stores_df):
    """Sinh orders với logic trạng thái CHÍNH XÁC"""
    orders = []
    line_items = []
    carts = []
    shipping_lines = []
    payment_method_lines = []
    
    order_id = 1
    line_item_id = 1
    cart_id = 1
    shipping_line_id = 1
    pml_id = 1
    
    # Group data
    customers_by_store = customers_df.groupby('store_id')
    addresses_by_customer = addresses_df.groupby('customer_id')
    variants_df_with_product = variants_df.merge(products_df[['id', 'name', 'store_id']], 
                                                   left_on='product_id', right_on='id', 
                                                   suffixes=('', '_product'))
    variants_by_store = variants_df_with_product.groupby('store_id')
    pms_by_store = payment_methods_df.groupby('store_id')
    
    print("\n📝 [8/10] Orders & Complete Relationships...")
    
    for _, store in stores_df.iterrows():
        store_customers = customers_by_store.get_group(store['id'])
        
        # Get store variants properly
        try:
            store_variants = variants_by_store.get_group(store['id'])
        except KeyError:
            continue
            
        store_pms = pms_by_store.get_group(store['id'])
        
        if len(store_variants) == 0:
            continue
        
        for _, customer in store_customers.iterrows():
            num_orders = random.randint(*NUM_ORDERS_PER_CUSTOMER)
            
            cust_addresses = addresses_by_customer.get_group(customer['id']) if customer['id'] in addresses_by_customer.groups else pd.DataFrame()
            
            for _ in range(num_orders):
                order_date = fake.date_time_between(
                    start_date=customer['created_at'],
                    end_date='now'
                )
                
                channel_id = random.choices(list(CHANNELS.keys()), 
                                           weights=[0.4, 0.2, 0.1, 0.15, 0.1, 0.03, 0.02])[0]
                
                # LOGIC TRẠNG THÁI ĐÚNG NGHIỆP VỤ
                # Dựa vào dữ liệu thực: hầu hết confirmed + pending + unpaid hoặc confirmed + fulfilled + paid
                
                status_type = random.choices(
                    ['normal_flow', 'completed', 'cancelled'],
                    weights=[0.6, 0.3, 0.1]
                )[0]
                
                if status_type == 'cancelled':
                    status = 'canceled'
                    financial_status = random.choice(['unpaid', 'refunded'])
                    fulfillment_status = 'pending'
                    confirmed_on = order_date
                    completed_on = None
                    canceled_on = order_date + timedelta(hours=random.randint(1, 48))
                    # Line items reference: order (canceled orders still have order line items)
                    line_items_reference_type = 'order'
                    
                elif status_type == 'completed':
                    status = 'completed'
                    financial_status = random.choice(['paid', 'paid', 'paid', 'partial_paid'])
                    fulfillment_status = random.choice(['fulfilled', 'fulfilled', 'partial_pending'])
                    confirmed_on = order_date
                    completed_on = order_date + timedelta(days=random.randint(1, 7))
                    canceled_on = None
                    # Line items reference: shipment nếu fulfilled, order nếu partial_pending
                    line_items_reference_type = 'shipment' if fulfillment_status == 'fulfilled' else 'order'
                    
                else:  # normal_flow - đang xử lý
                    status = random.choice(['open', 'confirmed', 'confirmed'])
                    financial_status = random.choice(['unpaid', 'unpaid', 'paid', 'partial_paid'])
                    fulfillment_status = random.choice(['pending', 'pending', 'fulfilled'])
                    confirmed_on = order_date
                    completed_on = None
                    canceled_on = None
                    # Line items reference: phụ thuộc vào fulfillment
                    if fulfillment_status == 'fulfilled':
                        line_items_reference_type = 'shipment'
                    elif status == 'open':
                        line_items_reference_type = 'checkout'  # Đang checkout
                    else:
                        line_items_reference_type = 'order'
                
                # Addresses
                if len(cust_addresses) > 0:
                    default_addr = cust_addresses[cust_addresses['default_address'] == True]
                    addr = default_addr.iloc[0] if len(default_addr) > 0 else cust_addresses.iloc[0]
                    billing_address_id = addr['id']
                    shipping_address_id = addr['id']
                else:
                    billing_address_id = None
                    shipping_address_id = None
                
                # Line items
                num_items = random.randint(1, 5)
                selected_variants = store_variants.sample(n=min(num_items, len(store_variants)))
                
                order_line_items = []
                total_line_item = 0
                item_count = 0
                total_weight = 0
                
                for _, variant in selected_variants.iterrows():
                    quantity = random.randint(1, 3)
                    price = variant['price']
                    total_line_item += price * quantity
                    item_count += quantity
                    total_weight += variant['weight'] * quantity
                    
                    order_line_items.append({
                        'id': line_item_id,
                        'variant_id': variant['id'],
                        'reference_id': order_id,
                        'reference_type': line_items_reference_type,
                        'quantity': quantity,
                        'note': None,
                        'price': price,
                        'product_name': variant['name'],
                        'variant_title': variant['title'],
                        'requires_shipping': True,
                        'grams': variant['weight'],
                    })
                    line_item_id += 1
                
                # Shipping
                shipping_fee = 0 if status == 'canceled' else random.choice([0, 20000, 30000, 40000, 50000])
                if shipping_fee > 0:
                    shipping_lines.append({
                        'id': shipping_line_id,
                        'shipping_rate_id': None,  # Can be NULL in schema
                        'order_id': order_id,
                        'title': 'Giao hàng tiêu chuẩn',
                        'price': shipping_fee,
                        'type': 'standard',
                    })
                    shipping_line_id += 1
                
                # Payment method
                pm = store_pms.sample(1).iloc[0]
                payment_amount = total_line_item + shipping_fee
                payment_method_lines.append({
                    'id': pml_id,
                    'order_id': order_id,
                    'payment_method_id': pm['id'],
                    'payment_method_name': pm['name'],
                    'amount': payment_amount if financial_status in ['paid', 'partial_paid'] else 0,
                })
                pml_id += 1
                
                total_price = total_line_item + shipping_fee
                
                # Generate tokens in proper format (UUID without hyphens for cart_token)
                cart_token = fake.uuid4().replace('-', '')
                checkout_token = fake.uuid4().replace('-', '')
                
                orders.append({
                    'id': order_id,
                    'cancel_reason': 'Khách hủy' if status == 'canceled' else '',
                    'canceled_on': canceled_on,
                    'confirmed_on': confirmed_on,
                    'checkout_token': checkout_token,
                    'cart_token': cart_token,
                    'closed_on': None,
                    'customer_id': customer['id'],
                    'assignee_id': 1,
                    'created_user_id': 1,
                    'note': '',
                    'order_number': order_id,
                    'name': f"#{order_id}",
                    'fulfillment_status': fulfillment_status,
                    'financial_status': financial_status,
                    'return_status': None,
                    'processed_on': confirmed_on,
                    'completed_on': completed_on,
                    'billing_address_id': billing_address_id,
                    'shipping_address_id': shipping_address_id,
                    'location_id': store['id'],
                    'store_id': store['id'],
                    'source_id': random.randint(1, 5),
                    'edited': False,
                    'expected_delivery_date': order_date + timedelta(days=random.randint(3, 10)),
                    'created_at': order_date,
                    'updated_at': order_date,
                    'deleted_at': None,
                    'channel_id': channel_id,
                    'status': status,
                })
                
                line_items.extend(order_line_items)
                order_id += 1
    
    # Generate CARTS with diverse statuses
    # - Active carts: customers đang shopping, chưa checkout
    # - Converted carts: đã chuyển thành order (link với orders via cart_token)
    
    # Type 1: Active carts (4% of customers = ~400 carts for active shopping)
    num_active_carts = int(len(customers_df) * 0.04)
    sample_customers = customers_df.sample(n=min(num_active_carts, len(customers_df)))
    
    for _, customer in sample_customers.iterrows():
        try:
            store_variants = variants_by_store.get_group(customer['store_id'])
        except KeyError:
            continue
        
        cart_token = fake.uuid4().replace('-', '')
        cart_date = fake.date_time_between(start_date='-1m', end_date='now')
        
        carts.append({
            'id': cart_id,
            'token': cart_token,
            'customer_id': customer['id'],
            'status': 'active',
            'store_id': customer['store_id'],
            'created_at': cart_date,
            'updated_at': cart_date,
            'deleted_at': None,
        })
        
        num_items = random.randint(1, 3)
        selected_variants = store_variants.sample(n=min(num_items, len(store_variants)))
        
        for _, variant in selected_variants.iterrows():
            line_items.append({
                'id': line_item_id,
                'variant_id': variant['id'],
                'reference_id': cart_id,
                'reference_type': 'cart',  # Active carts have 'cart' reference
                'quantity': random.randint(1, 2),
                'note': None,
                'price': variant['price'],
                'product_name': variant['name'],
                'variant_title': variant['title'],
                'requires_shipping': True,
                'grams': variant['weight'],
            })
            line_item_id += 1
        
        cart_id += 1
    
    # Type 2: Converted carts (~600 carts) - link to random orders
    num_converted_carts = min(600, len(orders))
    sample_orders = pd.DataFrame(orders).sample(n=num_converted_carts)
    
    for _, order in sample_orders.iterrows():
        # Create converted cart matching this order
        cart_date = order['created_at'] - timedelta(minutes=random.randint(5, 120))
        
        carts.append({
            'id': cart_id,
            'token': order['cart_token'],  # Use same token as order
            'customer_id': order['customer_id'],
            'status': 'converted',  # This cart was converted to order
            'store_id': order['store_id'],
            'created_at': cart_date,
            'updated_at': order['created_at'],  # Updated when converted
            'deleted_at': None,
        })
        
        cart_id += 1
    
    return {
        'orders': pd.DataFrame(orders),
        'line_items': pd.DataFrame(line_items),
        'carts': pd.DataFrame(carts),
        'shipping_lines': pd.DataFrame(shipping_lines),
        'payment_method_lines': pd.DataFrame(payment_method_lines),
    }

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    print("\n🚀 Sinh dữ liệu REALISTIC (V4)")
    print("="*60)
    
    # 1. Stores
    print("\n📦 [1/10] Stores...")
    stores_df = generate_stores(NUM_STORES)
    print(f"✅ {len(stores_df)} stores")
    
    # 2. Customers
    print("\n👥 [2/10] Customers...")
    customers_df = generate_customers(stores_df, NUM_CUSTOMERS_PER_STORE)
    print(f"✅ {len(customers_df)} customers")
    
    # 3. Addresses
    print("\n📍 [3/10] Addresses...")
    addresses_df = generate_addresses(customers_df)
    print(f"✅ {len(addresses_df)} addresses")
    
    # 4. Metadata
    print("\n🏷️ [4/10] Metadata...")
    metadata = generate_metadata_tables(stores_df)
    print(f"✅ {len(metadata['product_types'])} product_types")
    print(f"✅ {len(metadata['vendors'])} vendors")
    print(f"✅ {len(metadata['tags'])} tags")
    print(f"✅ {len(metadata['collections'])} collections")
    print(f"✅ {len(metadata['sources'])} sources")
    
    # 5. Products & Variants
    print("\n🛍️ [5/10] Products, Variants & Relationships...")
    product_data = generate_products_and_variants(stores_df, metadata)
    products_df = product_data['products']
    variants_df = product_data['variants']
    print(f"✅ {len(products_df)} products")
    print(f"✅ {len(variants_df)} variants")
    print(f"✅ {len(product_data['product_collections'])} product-collection links")
    print(f"✅ {len(product_data['product_tags'])} product-tag links")
    
    # 6. Inventory
    print("\n📦 [6/10] Inventory System...")
    inventory_data = generate_inventory(variants_df, stores_df)
    print(f"✅ {len(inventory_data['inventory_items'])} inventory_items")
    print(f"✅ {len(inventory_data['inventory_levels'])} inventory_levels")
    print(f"✅ {len(inventory_data['locations'])} locations")
    
    # 7. Payment Methods
    print("\n💳 [7/10] Payment Methods...")
    payment_methods_df = generate_payment_methods(stores_df)
    print(f"✅ {len(payment_methods_df)} payment_methods")
    
    # 8. Orders
    order_data = generate_orders_and_line_items(
        customers_df, products_df, variants_df, 
        addresses_df, payment_methods_df, stores_df
    )
    orders_df = order_data['orders']
    line_items_df = order_data['line_items']
    print(f"✅ {len(orders_df)} orders")
    print(f"✅ {len(line_items_df)} line_items")
    print(f"✅ {len(order_data['carts'])} carts")
    print(f"✅ {len(order_data['shipping_lines'])} shipping_lines")
    print(f"✅ {len(order_data['payment_method_lines'])} payment_method_lines")
    
    # 9. Save
    print("\n💾 [9/10] Saving to CSV...")
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save all tables
    stores_df.drop(columns=['_category']).to_csv(f"{OUTPUT_DIR}/stores.csv", index=False)
    customers_df.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
    addresses_df.to_csv(f"{OUTPUT_DIR}/addresses.csv", index=False)
    
    metadata['product_types'].to_csv(f"{OUTPUT_DIR}/product_types.csv", index=False)
    metadata['vendors'].to_csv(f"{OUTPUT_DIR}/vendors.csv", index=False)
    metadata['tags'].to_csv(f"{OUTPUT_DIR}/tags.csv", index=False)
    metadata['collections'].to_csv(f"{OUTPUT_DIR}/collections.csv", index=False)
    metadata['sources'].to_csv(f"{OUTPUT_DIR}/sources.csv", index=False)
    
    products_df.to_csv(f"{OUTPUT_DIR}/products.csv", index=False)
    variants_df.to_csv(f"{OUTPUT_DIR}/variants.csv", index=False)
    product_data['product_collections'].to_csv(f"{OUTPUT_DIR}/product_collections.csv", index=False)
    product_data['product_tags'].to_csv(f"{OUTPUT_DIR}/product_tags.csv", index=False)
    
    inventory_data['inventory_items'].to_csv(f"{OUTPUT_DIR}/inventory_items.csv", index=False)
    inventory_data['inventory_levels'].to_csv(f"{OUTPUT_DIR}/inventory_levels.csv", index=False)
    inventory_data['locations'].to_csv(f"{OUTPUT_DIR}/locations.csv", index=False)
    
    payment_methods_df.to_csv(f"{OUTPUT_DIR}/payment_methods.csv", index=False)
    
    orders_df.to_csv(f"{OUTPUT_DIR}/orders.csv", index=False)
    line_items_df.to_csv(f"{OUTPUT_DIR}/line_items.csv", index=False)
    order_data['carts'].to_csv(f"{OUTPUT_DIR}/carts.csv", index=False)
    order_data['shipping_lines'].to_csv(f"{OUTPUT_DIR}/shipping_lines.csv", index=False)
    order_data['payment_method_lines'].to_csv(f"{OUTPUT_DIR}/payment_method_lines.csv", index=False)
    
    print(f"✅ Saved 21 tables to {OUTPUT_DIR}")
    
    # 10. Statistics
    print("\n📊 [10/10] Statistics")
    print("="*60)
    print(f"\n🏪 Stores: {len(stores_df):,}")
    print(f"👤 Customers: {len(customers_df):,}")
    print(f"📍 Addresses: {len(addresses_df):,}")
    print(f"📦 Products: {len(products_df):,}")
    print(f"🎨 Variants: {len(variants_df):,}")
    print(f"🛍️ Orders: {len(orders_df):,}")
    print(f"📝 Line Items: {len(line_items_df):,}")
    print(f"🛒 Carts: {len(order_data['carts']):,}")
    
    # Calculate totals from line_items (since we removed computed fields)
    order_totals = line_items_df[line_items_df['reference_type'] == 'order'].groupby('reference_id').agg({
        'price': lambda x: (x * line_items_df.loc[x.index, 'quantity']).sum()
    }).rename(columns={'price': 'total'})
    
    total_revenue = order_totals['total'].sum()
    avg_order_value = order_totals['total'].mean()
    
    print(f"💰 Total Revenue: {total_revenue:,.0f} VNĐ")
    print(f"📊 AOV: {avg_order_value:,.0f} VNĐ")
    
    print("\n📋 Order Status:")
    print(orders_df['status'].value_counts().to_string())
    
    print("\n💳 Financial Status:")
    print(orders_df['financial_status'].value_counts().to_string())
    
    print("\n📦 Fulfillment Status:")
    print(orders_df['fulfillment_status'].value_counts().to_string())
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH - DỮ LIỆU REALISTIC 100%!")
    print("="*60)
