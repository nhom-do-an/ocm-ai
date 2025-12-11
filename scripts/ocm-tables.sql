CREATE SCHEMA IF NOT EXISTS "public";

CREATE  TABLE "public".address_regions ( 
	id                   serial  NOT NULL  ,
	address_id           integer  NOT NULL  ,
	old_region_id        integer  NOT NULL  ,
	"type"               integer    ,
	CONSTRAINT address_regions_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".addresses ( 
	id                   serial  NOT NULL  ,
	customer_id          integer    ,
	address              varchar    ,
	phone                varchar    ,
	email                varchar    ,
	zip                  varchar    ,
	default_address      boolean    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	is_new_region        boolean DEFAULT false   ,
	first_name           varchar(100)    ,
	last_name            varchar(100)    ,
	CONSTRAINT addresses_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".attachments ( 
	id                   serial  NOT NULL  ,
	filename             varchar    ,
	file_path            varchar    ,
	file_size            double precision    ,
	mime_type            varchar    ,
	status               integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	category             varchar    ,
	store_id             integer    ,
	CONSTRAINT attachments_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".attribute_options ( 
	id                   serial  NOT NULL  ,
	option_id            integer    ,
	attribute_id         integer    ,
	"position"           integer    ,
	product_id           integer    ,
	CONSTRAINT attribute_options_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".attributes ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	"values"             jsonb    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT attributes_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".banks ( 
	id                   serial  NOT NULL  ,
	name                 varchar(255)  NOT NULL  ,
	bin                  varchar(20)  NOT NULL  ,
	logo                 varchar(255)    ,
	short_name           varchar(100)    ,
	CONSTRAINT banks_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".beneficiary_accounts ( 
	id                   serial  NOT NULL  ,
	account_name         varchar    ,
	account_number       varchar    ,
	note                 text    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	bank_id              integer    ,
	CONSTRAINT beneficiary_accounts_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".carts ( 
	id                   serial  NOT NULL  ,
	token                varchar(255)  NOT NULL  ,
	customer_id          integer    ,
	status               varchar(50)  NOT NULL  ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT carts_pkey PRIMARY KEY ( id ),
	CONSTRAINT carts_token_key UNIQUE ( token ) 
 );

CREATE  TABLE "public".catalog_targets ( 
	id                   serial  NOT NULL  ,
	catalog_id           integer    ,
	reference_id         integer    ,
	"position"           integer    ,
	CONSTRAINT catalog_targets_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".catalogs ( 
	id                   serial  NOT NULL  ,
	code                 varchar    ,
	title                varchar    ,
	"type"               varchar    ,
	status               varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT catalogs_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".channels ( 
	id                   serial  NOT NULL  ,
	name                 varchar(255)    ,
	description          varchar(500)    ,
	short_name           varchar(100)    ,
	"type"               varchar(100)    ,
	image_url            varchar(500)    ,
	CONSTRAINT channels_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".checkout_settings ( 
	id                   serial  NOT NULL  ,
	customer_account     varchar    ,
	phone                varchar    ,
	email                varchar    ,
	address              varchar    ,
	district             varchar    ,
	ward                 varchar    ,
	zip                  varchar    ,
	country              varchar    ,
	same_billing_and_shipping boolean    ,
	order_closing        boolean    ,
	refund_policy        text    ,
	privacy_policy       text    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT checkout_settings_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".checkouts ( 
	id                   serial  NOT NULL  ,
	token                varchar(255)  NOT NULL  ,
	cart_token           varchar(255)    ,
	customer_id          integer    ,
	status               varchar(50)  NOT NULL  ,
	completed_at         timestamp    ,
	note                 text    ,
	email_state          varchar    ,
	email_sent_on        timestamp    ,
	billing_address_id   integer    ,
	shipping_address_id  integer    ,
	shipping_rate_id     integer    ,
	payment_method_id    integer    ,
	order_id             integer    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	channel_id           integer    ,
	CONSTRAINT checkouts_pkey PRIMARY KEY ( id ),
	CONSTRAINT checkouts_token_key UNIQUE ( token ) 
 );

CREATE  TABLE "public".collection_attachments ( 
	id                   serial  NOT NULL  ,
	collection_id        integer    ,
	attachment_id        integer    ,
	"position"           integer    ,
	CONSTRAINT collection_attachments_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".collection_attributes ( 
	id                   serial  NOT NULL  ,
	collection_id        integer    ,
	attribute_id         integer    ,
	CONSTRAINT collection_attributes_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".collection_product_types ( 
	id                   serial  NOT NULL  ,
	collection_id        integer    ,
	product_type_id      integer    ,
	CONSTRAINT collection_product_types_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".collections ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	"alias"              varchar    ,
	description          text    ,
	meta_title           varchar    ,
	meta_description     text    ,
	sort_order           varchar    ,
	"type"               varchar    ,
	disjunctive          boolean    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT collections_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".combinations ( 
	id                   serial  NOT NULL  ,
	order_discount       boolean    ,
	product_discount     boolean    ,
	shipping_discount    boolean    ,
	CONSTRAINT combinations_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".combo_items ( 
	id                   serial  NOT NULL  ,
	combo_id             integer    ,
	variant_id           integer    ,
	quantity             integer    ,
	"position"           integer    ,
	price                double precision    ,
	product_name         varchar    ,
	title                varchar    ,
	CONSTRAINT combo_items_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".combos ( 
	id                   serial  NOT NULL  ,
	variant_id           integer    ,
	price                double precision    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT combos_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".customer_group_rules ( 
	id                   serial  NOT NULL  ,
	customer_group_id    integer    ,
	rule_id              integer    ,
	"position"           integer    ,
	CONSTRAINT customer_group_rules_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".customer_groups ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	disjunctive          boolean    ,
	note                 text    ,
	"type"               varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT customer_groups_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".customers ( 
	id                   serial  NOT NULL  ,
	email                varchar    ,
	phone                varchar    ,
	first_name           varchar    ,
	last_name            varchar    ,
	"password"           varchar    ,
	gender               varchar    ,
	dob                  varchar    ,
	note                 text    ,
	status               varchar    ,
	verified_email       boolean    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	store_id             integer    ,
	CONSTRAINT customers_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".delivery_providers ( 
	id                   serial  NOT NULL  ,
	code                 varchar    ,
	name                 varchar    ,
	address              varchar    ,
	"type"               varchar    ,
	phone                varchar    ,
	email                varchar    ,
	note                 text    ,
	status               varchar    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT delivery_providers_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".discounts ( 
	id                   serial  NOT NULL  ,
	title                varchar    ,
	starts_on            timestamp    ,
	ends_on              timestamp    ,
	status               varchar    ,
	value_type           varchar    ,
	"value"              integer    ,
	customer_selection   varchar    ,
	location_selection   varchar    ,
	target_selection     varchar    ,
	target_type          varchar    ,
	usage_limit          integer    ,
	once_per_customer    boolean    ,
	value_limit_amount   integer    ,
	allocation_limit     integer    ,
	summary              text    ,
	rule_type            varchar    ,
	prerequisite_sale_total_range integer    ,
	combines_with        integer    ,
	store_id             integer    ,
	prerequisite_to_entitlement_type varchar    ,
	prerequisite_to_entitlement_id integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT discounts_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".events ( 
	id                   serial  NOT NULL  ,
	description          text    ,
	"path"               varchar    ,
	message              text    ,
	subject_type         varchar    ,
	author_type          varchar    ,
	user_id              integer    ,
	verb                 varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	subject_id           integer    ,
	CONSTRAINT events_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".fulfillment_counters ( 
	store_id             integer  NOT NULL  ,
	last_fulfillment_number integer DEFAULT 0 NOT NULL  ,
	CONSTRAINT fulfillment_counters_pkey PRIMARY KEY ( store_id )
 );

CREATE  TABLE "public".fulfillment_line_items ( 
	id                   serial  NOT NULL  ,
	fulfillment_id       integer    ,
	line_item_id         integer    ,
	quantity             integer    ,
	CONSTRAINT fulfillment_line_items_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".fulfillments ( 
	id                   serial  NOT NULL  ,
	name                 varchar(255)    ,
	store_id             integer    ,
	order_id             integer    ,
	location_id          integer    ,
	channel_id           integer    ,
	source_id            integer    ,
	delivery_method      varchar(255)    ,
	shipment_status      varchar(50)    ,
	status               varchar(50)    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT fulfillments_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".inventory_adjustments ( 
	id                   serial  NOT NULL  ,
	store_id             integer    ,
	location_id          integer    ,
	inventory_item_id    integer    ,
	changes              jsonb    ,
	reference_document_type varchar    ,
	reference_document_name varchar    ,
	reference_document_id integer    ,
	actor_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT inventory_adjustments_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".inventory_items ( 
	id                   serial  NOT NULL  ,
	variant_id           integer    ,
	tracked              boolean    ,
	requires_shipping    boolean    ,
	cost_price           double precision    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	lot_management       boolean    ,
	CONSTRAINT inventory_items_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".inventory_levels ( 
	id                   serial  NOT NULL  ,
	inventory_item_id    integer    ,
	location_id          integer    ,
	store_id             integer    ,
	on_hand              integer    ,
	available            integer    ,
	"committed"          integer    ,
	incoming             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT inventory_levels_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".line_items ( 
	id                   serial  NOT NULL  ,
	variant_id           integer    ,
	reference_id         integer    ,
	reference_type       varchar    ,
	quantity             integer    ,
	note                 text    ,
	price                double precision    ,
	product_name         varchar    ,
	variant_title        varchar    ,
	requires_shipping    boolean    ,
	grams                integer    ,
	CONSTRAINT line_items_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".locations ( 
	id                   serial  NOT NULL  ,
	code                 varchar    ,
	name                 varchar    ,
	email                varchar    ,
	phone                varchar    ,
	address              varchar    ,
	zip                  varchar    ,
	fulfill_order        boolean    ,
	inventory_management boolean    ,
	default_location     boolean    ,
	status               varchar    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT locations_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".notification_readers ( 
	id                   bigserial  NOT NULL  ,
	notification_id      bigint    ,
	user_id              bigint    ,
	created_at           timestamptz    ,
	updated_at           timestamptz    ,
	deleted_at           timestamptz    ,
	CONSTRAINT notification_readers_pkey PRIMARY KEY ( id )
 );

CREATE INDEX idx_notification_readers_deleted_at ON "public".notification_readers USING  btree ( deleted_at );

CREATE  TABLE "public".notification_templates ( 
	id                   serial  NOT NULL  ,
	"template"           varchar(255)  NOT NULL  ,
	name                 varchar(255)  NOT NULL  ,
	description          text    ,
	subject              varchar(255)  NOT NULL  ,
	content              text  NOT NULL  ,
	category             varchar(100)  NOT NULL  ,
	active               boolean DEFAULT true NOT NULL  ,
	can_edit_active      boolean DEFAULT true NOT NULL  ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT notification_templates_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".notifications ( 
	id                   bigserial  NOT NULL  ,
	title                text    ,
	content              text    ,
	detail_content       text    ,
	image                text    ,
	topic                text    ,
	"type"               bigint    ,
	html_content         text    ,
	created_user_id      text    ,
	metadata             jsonb    ,
	status               bigint    ,
	schedule_time        timestamptz    ,
	created_at           timestamptz    ,
	updated_at           timestamptz    ,
	deleted_at           timestamptz    ,
	CONSTRAINT notifications_pkey PRIMARY KEY ( id )
 );

CREATE INDEX idx_notifications_deleted_at ON "public".notifications USING  btree ( deleted_at );

CREATE  TABLE "public".object_vouchers ( 
	id                   serial  NOT NULL  ,
	object_type          varchar    ,
	object_name          varchar    ,
	object_id            integer    ,
	CONSTRAINT object_vouchers_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".old_regions ( 
	id                   serial  NOT NULL  ,
	name                 varchar  NOT NULL  ,
	code                 varchar    ,
	"type"               integer    ,
	parent_code          varchar    ,
	priority             integer    ,
	CONSTRAINT old_regions_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public"."options" ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT options_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".order_counters ( 
	store_id             integer  NOT NULL  ,
	last_order_number    integer  NOT NULL  ,
	CONSTRAINT order_counters_pkey PRIMARY KEY ( store_id )
 );

CREATE  TABLE "public".orders ( 
	id                   serial  NOT NULL  ,
	cancel_reason        text    ,
	canceled_on          timestamp    ,
	confirmed_on         timestamp    ,
	checkout_token       varchar    ,
	cart_token           varchar    ,
	closed_on            timestamp    ,
	customer_id          integer    ,
	assignee_id          integer    ,
	created_user_id      integer    ,
	note                 text    ,
	order_number         integer    ,
	name                 varchar    ,
	fulfillment_status   varchar    ,
	financial_status     varchar    ,
	return_status        varchar    ,
	processed_on         timestamp    ,
	completed_on         timestamp    ,
	billing_address_id   integer    ,
	shipping_address_id  integer    ,
	location_id          integer    ,
	store_id             integer    ,
	source_id            integer    ,
	edited               boolean    ,
	expected_delivery_date timestamp    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	channel_id           integer    ,
	status               varchar(20)    ,
	CONSTRAINT orders_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".packsizes ( 
	id                   serial  NOT NULL  ,
	packsize_variant_id  integer    ,
	variant_id           integer    ,
	quantity             varchar    ,
	unit                 varchar    ,
	price                double precision    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT packsizes_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".payment_method_lines ( 
	id                   serial  NOT NULL  ,
	order_id             integer    ,
	payment_method_id    integer    ,
	payment_method_name  varchar(255)    ,
	amount               decimal(10,2)    ,
	CONSTRAINT payment_method_lines_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".payment_methods ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	description          text    ,
	status               varchar    ,
	auto_posting_receipt boolean    ,
	provider_id          integer    ,
	beneficiary_account_id integer    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT payment_methods_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".payment_providers ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	description          text    ,
	CONSTRAINT payment_providers_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".prerequisite_to_entitlement_purchases ( 
	id                   serial  NOT NULL  ,
	prerequisite_amount  double precision    ,
	entitled_amount      double precision    ,
	CONSTRAINT prerequisite_to_entitlement_purchases_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".prerequisite_to_entitlement_quantity_radios ( 
	id                   serial  NOT NULL  ,
	prerequisite_quantity integer    ,
	entitled_quantity    integer    ,
	CONSTRAINT prerequisite_to_entitlement_quantity_radios_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".price_lists ( 
	id                   serial  NOT NULL  ,
	catalog_id           integer    ,
	title                varchar    ,
	adjustment_type      varchar    ,
	adjustment_value     integer    ,
	CONSTRAINT price_lists_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".product_attachments ( 
	id                   serial  NOT NULL  ,
	attachment_id        integer    ,
	product_id           integer    ,
	"position"           integer    ,
	CONSTRAINT product_attachments_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".product_attributes ( 
	id                   serial  NOT NULL  ,
	product_id           integer    ,
	attribute_id         integer    ,
	"position"           integer    ,
	CONSTRAINT product_attributes_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".product_collections ( 
	id                   serial  NOT NULL  ,
	product_id           integer    ,
	collection_id        integer    ,
	auto                 boolean    ,
	"position"           integer    ,
	CONSTRAINT product_collections_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".product_publications ( 
	id                   serial  NOT NULL  ,
	publication_id       integer    ,
	product_id           integer    ,
	"position"           integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT product_publications_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".product_tags ( 
	id                   serial  NOT NULL  ,
	product_id           integer    ,
	tag_id               integer    ,
	"position"           integer    ,
	CONSTRAINT product_tags_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".product_types ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	"alias"              varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	store_id             integer    ,
	CONSTRAINT product_types_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".products ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	"alias"              varchar    ,
	product_type_id      integer    ,
	meta_title           varchar    ,
	meta_description     text    ,
	summary              text    ,
	published_on         timestamp    ,
	content              text    ,
	status               varchar    ,
	"type"               varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	vendor_id            integer    ,
	store_id             integer    ,
	CONSTRAINT products_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".provinces ( 
	id                   bigserial  NOT NULL  ,
	province_code        varchar(2)  NOT NULL  ,
	name                 varchar(255)  NOT NULL  ,
	short_name           varchar(255)  NOT NULL  ,
	code                 varchar(5)  NOT NULL  ,
	place_type           varchar(255)  NOT NULL  ,
	country_code         varchar(10)  NOT NULL  ,
	CONSTRAINT provinces_pkey PRIMARY KEY ( id )
 );

CREATE UNIQUE INDEX provinces_province_code_unique ON "public".provinces ( province_code );

CREATE  TABLE "public".publications ( 
	id                   serial  NOT NULL  ,
	channel_id           integer    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	status               varchar DEFAULT 'active'::character varying   ,
	CONSTRAINT publications_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".region_locations ( 
	id                   serial  NOT NULL  ,
	region_id            integer    ,
	location_id          integer    ,
	"type"               integer    ,
	CONSTRAINT region_locations_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".region_stores ( 
	id                   serial  NOT NULL  ,
	region_id            integer    ,
	store_id             integer    ,
	"type"               integer    ,
	CONSTRAINT region_stores_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".regions ( 
	id                   serial  NOT NULL  ,
	name                 varchar  NOT NULL  ,
	code                 varchar    ,
	"type"               integer    ,
	parent_code          varchar    ,
	priority             integer    ,
	CONSTRAINT regions_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".rules ( 
	id                   serial  NOT NULL  ,
	"column_name"        varchar    ,
	relation             varchar    ,
	condition            varchar    ,
	collection_id        integer    ,
	CONSTRAINT rules_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".schema_migrations ( 
	"version"            bigint  NOT NULL  ,
	dirty                boolean  NOT NULL  ,
	CONSTRAINT schema_migrations_pkey PRIMARY KEY ( "version" )
 );

CREATE  TABLE "public".selection_targets ( 
	id                   serial  NOT NULL  ,
	"type"               varchar    ,
	reference_type       varchar    ,
	reference_id         integer    ,
	discount_id          integer    ,
	CONSTRAINT selection_targets_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".shipment_counters ( 
	store_id             integer  NOT NULL  ,
	last_shipment_number integer DEFAULT 1000 NOT NULL  ,
	CONSTRAINT shipment_counters_pkey PRIMARY KEY ( store_id )
 );

CREATE  TABLE "public".shipments ( 
	id                   serial  NOT NULL  ,
	order_id             integer    ,
	fulfillment_id       integer    ,
	tracking_info_id     integer    ,
	shipping_info_id     integer    ,
	note                 text    ,
	location_id          integer    ,
	channel_id           integer    ,
	source_id            integer    ,
	shipping_address_id  integer    ,
	printed              boolean    ,
	cancelled_on         timestamp    ,
	delivered_on         timestamp    ,
	picked_up_on         timestamp    ,
	status               varchar(50)    ,
	delivery_status      varchar(50)    ,
	payment_status       varchar(50)    ,
	delivery_method      varchar(50)    ,
	service_fee          double precision    ,
	cod_amount           double precision    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	name                 varchar(255)    ,
	store_id             integer    ,
	CONSTRAINT shipments_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".shipping_area_locations ( 
	id                   serial  NOT NULL  ,
	shipping_area_id     integer    ,
	region_id            integer    ,
	CONSTRAINT shipping_area_locations_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".shipping_areas ( 
	id                   serial  NOT NULL  ,
	group_name           varchar    ,
	store_id             integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT shipping_areas_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".shipping_infos ( 
	id                   serial  NOT NULL  ,
	insurance_value      double precision    ,
	freight_payer        varchar(255)    ,
	weight               integer    ,
	"length"             integer    ,
	width                integer    ,
	height               integer    ,
	note                 text    ,
	weight_type          varchar(50)    ,
	requirement          integer    ,
	CONSTRAINT shipping_infos_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".shipping_lines ( 
	id                   serial  NOT NULL  ,
	shipping_rate_id     integer    ,
	order_id             integer    ,
	title                varchar(255)    ,
	price                decimal(10,2)    ,
	"type"               varchar(50)    ,
	CONSTRAINT shipping_lines_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".shipping_rate_conditions ( 
	id                   serial  NOT NULL  ,
	shipping_rate_id     integer    ,
	"type"               varchar    ,
	min_value            double precision    ,
	max_value            double precision    ,
	CONSTRAINT shipping_rate_conditions_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".shipping_rates ( 
	id                   serial  NOT NULL  ,
	shipping_area_id     integer    ,
	name                 varchar    ,
	price                double precision    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	store_id             integer    ,
	CONSTRAINT shipping_rates_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".shipping_settings ( 
	id                   serial  NOT NULL  ,
	weight               integer    ,
	"length"             integer    ,
	width                integer    ,
	height               integer    ,
	note                 text    ,
	weight_type          varchar    ,
	requirement          integer    ,
	store_id             integer    ,
	CONSTRAINT shipping_settings_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".sources ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	"alias"              varchar    ,
	image_url            varchar    ,
	default_source       boolean    ,
	CONSTRAINT sources_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".store_attachments ( 
	id                   serial  NOT NULL  ,
	store_id             integer  NOT NULL  ,
	attachment_id        integer  NOT NULL  ,
	CONSTRAINT store_attachments_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".stores ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	trade_name           varchar    ,
	email                varchar    ,
	phone                varchar    ,
	"domain"             varchar    ,
	"alias"              varchar    ,
	address              varchar    ,
	currency             varchar    ,
	timezone             varchar    ,
	money_format         varchar    ,
	money_with_currency_format varchar    ,
	weight_unit          varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT stores_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".tags ( 
	id                   serial  NOT NULL  ,
	"type"               varchar    ,
	name                 varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	store_id             integer    ,
	CONSTRAINT tags_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".tracking_infos ( 
	id                   serial  NOT NULL  ,
	tracking_number      varchar(255)    ,
	tracking_reference   varchar(255)    ,
	delivery_provider_id integer    ,
	CONSTRAINT tracking_infos_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".transactions ( 
	id                   serial  NOT NULL  ,
	reference_id         integer    ,
	reference_type       varchar    ,
	amount               double precision    ,
	description          text    ,
	payment_method_id    integer    ,
	status               integer    ,
	paid_on              timestamp    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	kind                 varchar(50)    ,
	cause_type           varchar(50)    ,
	CONSTRAINT transactions_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".user_stores ( 
	id                   serial  NOT NULL  ,
	store_id             integer    ,
	user_id              integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	is_owner             boolean    ,
	last_login           timestamp    ,
	CONSTRAINT user_stores_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".users ( 
	id                   serial  NOT NULL  ,
	email                varchar    ,
	first_name           varchar    ,
	last_name            varchar    ,
	phone                varchar    ,
	"password"           varchar    ,
	last_login           timestamp    ,
	active               boolean    ,
	verified_email       boolean    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	login_method         integer    ,
	name                 varchar    ,
	username             varchar    ,
	CONSTRAINT users_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".variants ( 
	id                   serial  NOT NULL  ,
	title                varchar    ,
	sku                  varchar    ,
	barcode              varchar    ,
	price                double precision    ,
	compare_at_price     double precision    ,
	weight               double precision    ,
	unit                 varchar    ,
	weight_unit          varchar    ,
	image_id             integer    ,
	"position"           integer    ,
	option1              varchar    ,
	option2              varchar    ,
	option3              varchar    ,
	"type"               varchar    ,
	sold                 integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	store_id             integer    ,
	product_id           integer    ,
	CONSTRAINT variants_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".vendors ( 
	id                   serial  NOT NULL  ,
	name                 varchar    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	store_id             integer    ,
	CONSTRAINT vendors_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".voucher_attchments ( 
	id                   serial  NOT NULL  ,
	voucher_id           integer    ,
	attachment_id        integer    ,
	"position"           integer    ,
	CONSTRAINT voucher_attchments_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".vouchers ( 
	id                   serial  NOT NULL  ,
	store_id             integer    ,
	location_id          integer    ,
	transaction_id       integer    ,
	object_voucher_id    integer    ,
	code                 varchar    ,
	"type"               varchar    ,
	reason               varchar    ,
	memo                 varchar    ,
	voucher_date         timestamp    ,
	reference_document_type varchar    ,
	reference_document_name varchar    ,
	reference_document_id integer    ,
	auto                 boolean    ,
	create_by            integer    ,
	created_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	updated_at           timestamp DEFAULT CURRENT_TIMESTAMP   ,
	deleted_at           timestamp    ,
	CONSTRAINT vouchers_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".ward_mappings ( 
	id                   serial  NOT NULL  ,
	old_ward_code        varchar    ,
	old_ward_name        varchar    ,
	old_district_name    varchar    ,
	old_province_name    varchar    ,
	new_ward_code        varchar    ,
	new_ward_name        varchar    ,
	new_province_name    varchar    ,
	created_at           timestamp    ,
	updated_at           timestamp    ,
	CONSTRAINT ward_mappings_pkey PRIMARY KEY ( id )
 );

CREATE  TABLE "public".wards ( 
	id                   bigserial  NOT NULL  ,
	ward_code            varchar(6)  NOT NULL  ,
	name                 varchar(255)  NOT NULL  ,
	province_code        varchar(2)  NOT NULL  ,
	CONSTRAINT wards_pkey PRIMARY KEY ( id )
 );

CREATE UNIQUE INDEX wards_ward_code_unique ON "public".wards ( ward_code );

CREATE INDEX wards_province_code_index ON "public".wards USING  btree ( province_code );

